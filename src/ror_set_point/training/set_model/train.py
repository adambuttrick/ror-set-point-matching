"""Train the set model, calibrate it, and answer the evaluation rows of the data folder.

    python -m ror_set_point.training.set_model.train --data set_data --out set_model --base jhu-clsp/mmBERT-base

Stage 1 trains on stage1.jsonl.gz (teacher and synthetic rows) for one epoch at --lr. Stage 2 trains on
stage2.jsonl.gz (labeled rows) for --epochs2 epochs at --lr / 2, from the stage 1 weights: once for each fold, to
get out-of-fold scores, and once on all rows (the final model). Platt scaling (a * score + b) is fitted on the
out-of-fold scores of every candidate. The loss is binary cross-entropy against the targets, over the real
candidates. AdamW (weight decay 0.01), linear warm-up for 5% of the steps then linear decay, bf16 autocast on CUDA,
gradient clipping 1.0.

--stage1 FILE writes the stage 1 weights to FILE. When FILE exists, they are read from it and stage 1 is not
trained; the stage 1 rows and settings must be the same. Stage 1 rows have no lexical candidates, so runs whose
stage 2 rows come from other lexical indexes can use the same stage 1.

Writes into --out (which must not exist): model/ (the encoder in bf16, head.safetensors, the tokenizer and
set_model_config.json, as ror_set_point.inference.set_model reads them), calibration.json, oof.jsonl.gz ({k, c, y, z}),
probabilities.jsonl.gz ({set, k, c, z, p} for the evaluation rows), predictions/<set>.jsonl ({id, ror_ids}),
train_log.txt and summary.json.
"""
import argparse
import gzip
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from ror_set_point.common.io import check_manifest, read_jsonl, write_jsonl
from ror_set_point.inference.set_model import Packer, SetModel, calibrate, load_encoder, scores
from ror_set_point.training.retriever.train import fix_gemma_config


def targets(rows, n, dev):
    y = torch.zeros((len(rows), n))
    real = torch.zeros((len(rows), n), dtype=torch.bool)
    for i, r in enumerate(rows):
        y[i, :len(r["y"])] = torch.tensor(r["y"])
        real[i, :len(r["c"])] = True
    return y.to(dev), real.to(dev)


def train(model, packer, rows, epochs, lr, steps, rng, a, dev, log, tag):
    order = []
    for _ in range(epochs):
        o = list(range(len(rows)))
        rng.shuffle(o)
        order += o
    total = math.ceil(len(order) / a.batch_size) if steps < 0 else steps
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    model.train()
    t0, run_loss, n_log = time.time(), 0.0, 0
    for step in range(total):
        B = [rows[i] for i in order[(step * a.batch_size) % len(order):][:a.batch_size]]
        ids, mask, seg, n = packer([(r["s"], r["c"]) for r in B], dev)
        y, real = targets(B, n, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            z = model(ids, mask, seg, n)
        loss = F.binary_cross_entropy_with_logits(z[real], y[real])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run_loss += loss.item()
        n_log += 1
        if (step + 1) % a.log_every == 0 or step + 1 == total:
            el = time.time() - t0
            log(f"{tag} step {step + 1}/{total} loss {run_loss / n_log:.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{el / (step + 1):.3f} s/step eta {(total - step - 1) * el / (step + 1) / 60:.1f} min")
            run_loss, n_log = 0.0, 0
    return {"steps": total, "seconds": round(time.time() - t0)}


def fit_platt(oof):
    """a, b that minimize the log loss of sigmoid(a * z + b) on the out-of-fold scores, and the fit's measures."""
    zs = torch.tensor([v for r in oof for v in r["z"]])
    ys = torch.tensor([v for r in oof for v in r["y"]])
    ab = torch.zeros(2, requires_grad=True)
    opt = torch.optim.LBFGS([ab], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits((1 + ab[0]) * zs + ab[1], ys)
        loss.backward()
        return loss
    opt.step(closure)
    a, b = 1 + ab[0].item(), ab[1].item()
    p = torch.sigmoid(a * zs + b)
    hit = ((p >= 0.5) & (ys >= 0.5)).sum().item()
    exact = sum(all((calibrate(z, a, b) >= 0.5) == (y >= 0.5) for z, y in zip(r["z"], r["y"])) for r in oof)
    return {"a": a, "b": b, "candidates": len(zs),
            "log_loss_raw": F.binary_cross_entropy_with_logits(zs, ys).item(),
            "log_loss_calibrated": F.binary_cross_entropy(p, ys).item(),
            "precision": hit / max(1, (p >= 0.5).sum().item()), "recall": hit / max(1, (ys >= 0.5).sum().item()),
            "exact_match": exact / max(1, len(oof))}


def save(model, tok, cal, a, path):
    from safetensors.torch import save_file
    model.enc.to(torch.bfloat16).save_pretrained(path, safe_serialization=True)
    model.enc.float()
    fix_gemma_config(path, a.base, a.base_revision)
    save_file({"weight": model.head.weight.detach().float().cpu().contiguous(),
               "bias": model.head.bias.detach().float().cpu().contiguous()}, os.path.join(path, "head.safetensors"))
    tok.save_pretrained(path)
    with open(os.path.join(path, "set_model_config.json"), "w", encoding="utf-8") as f:
        json.dump({"base_model": a.base, "base_revision": a.base_revision, "string_max_tokens": a.string_length,
                   "card_max_tokens": a.record_length, "max_tokens": a.max_length, "threshold": a.threshold,
                   "calibration": {"a": cal["a"], "b": cal["b"]}}, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="a folder from ror_set_point.training.set_model.build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True, help="for example jhu-clsp/mmBERT-base")
    ap.add_argument("--base-revision")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs2", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--string-length", type=int, default=96)
    ap.add_argument("--record-length", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1363)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="read only this many rows of each file (a smoke test)")
    ap.add_argument("--steps1", type=int, default=-1, help="stage 1 steps (-1: one epoch; a smoke test sets it)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--stage1", help="a file of the stage 1 weights: read when it exists, else written")
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    os.makedirs(a.out)
    os.makedirs(os.path.join(a.out, "predictions"))
    logf = open(os.path.join(a.out, "train_log.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t_start = time.time()
    dev = torch.device(a.device)
    settings = {k: v for k, v in vars(a).items() if k not in ("data", "out")}
    log(f"device {torch.cuda.get_device_name(0) if dev.type == 'cuda' else dev.type}; settings {json.dumps(settings)}")
    manifest = check_manifest(a.data, ("records.jsonl.gz", "stage1.jsonl.gz", "stage2.jsonl.gz", "eval.jsonl.gz"))
    folds = manifest["folds"]
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.base_revision)
    packer = Packer(tok, {r["ror_id"]: r["text"] for r in read_jsonl(os.path.join(a.data, "records.jsonl.gz"))},
                    a.string_length, a.record_length, a.max_length)
    log(f"{len(packer.ctok):,} records tokenized; start {packer.start}, sep {packer.sep}, pad {packer.pad}")
    S1 = list(read_jsonl(os.path.join(a.data, "stage1.jsonl.gz"), a.limit))
    S2 = list(read_jsonl(os.path.join(a.data, "stage2.jsonl.gz"), a.limit))
    EV = list(read_jsonl(os.path.join(a.data, "eval.jsonl.gz"), a.limit))
    log(f"stage 1 {len(S1):,} rows, stage 2 {len(S2):,}, evaluation {len(EV):,}")
    model = SetModel(load_encoder(a.base, a.base_revision)).to(dev)
    summary = {"settings": settings, "params": sum(p.numel() for p in model.parameters()),
               "data_sha256": manifest["sha256"]}

    key = {"stage1.jsonl.gz": manifest["sha256"]["stage1.jsonl.gz"],
           **{x: settings[x] for x in ("base", "base_revision", "lr", "batch_size", "string_length", "record_length",
                                       "max_length", "seed", "limit", "steps1")}}
    if a.stage1 and os.path.exists(a.stage1):
        saved = torch.load(a.stage1, map_location="cpu")
        if saved["key"] != key:
            raise SystemExit(f"{a.stage1} has other stage 1 rows or settings: {saved['key']}")
        model.load_state_dict(saved["state"])
        rng.shuffle(list(range(len(S1))))  # as train() does, so that stage 2 gets the same order
        summary["stage1"] = {"from": a.stage1}
        log(f"stage 1 weights from {a.stage1}")
    else:
        summary["stage1"] = train(model, packer, S1, 1, a.lr, a.steps1, rng, a, dev, log, "s1")
    state1 = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    if a.stage1 and not os.path.exists(a.stage1):
        torch.save({"key": key, "state": state1}, a.stage1)
        log(f"stage 1 weights to {a.stage1}")
    steps2 = -1 if a.steps1 < 0 else max(1, a.steps1 // 4)
    oof = []
    for f in range(folds):
        model.load_state_dict(state1)
        tr = [r for r in S2 if r["fold"] != f]
        te = [r for r in S2 if r["fold"] == f]
        summary[f"stage2_fold{f}"] = train(model, packer, tr, a.epochs2, a.lr / 2, steps2, rng, a, dev, log,
                                           f"s2 fold {f}")
        z = scores(model, packer, [(r["s"], r["c"]) for r in te], dev)
        oof += [{"k": r["k"], "c": r["c"], "y": r["y"], "z": zz} for r, zz in zip(te, z)]
    cal = fit_platt(oof)
    log(f"calibration {json.dumps(cal)}")
    with open(os.path.join(a.out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=1)
    write_jsonl(os.path.join(a.out, "oof.jsonl.gz"), oof)
    model.load_state_dict(state1)
    summary["stage2_full"] = train(model, packer, S2, a.epochs2, a.lr / 2, steps2, rng, a, dev, log, "s2 full")

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    z = scores(model, packer, [(r["s"], r["c"]) for r in EV], dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    summary["evaluation_strings_per_second"] = round(len(EV) / max(time.time() - t0, 1e-9), 1)
    answers = {}
    with gzip.open(os.path.join(a.out, "probabilities.jsonl.gz"), "wt", encoding="utf-8") as fp:
        for r, zz in zip(EV, z):
            p = [calibrate(v, cal["a"], cal["b"]) for v in zz]
            fp.write(json.dumps({"set": r["set"], "k": r["k"], "c": r["c"], "z": zz, "p": p}) + "\n")
            answers.setdefault(r["set"], []).append(
                {"id": r["k"], "ror_ids": [x for x, v in zip(r["c"], p) if v >= a.threshold]})
    for name, rows in answers.items():
        write_jsonl(os.path.join(a.out, "predictions", f"{name}.jsonl"), rows)
    save(model, tok, cal, a, os.path.join(a.out, "model"))
    summary["total_seconds"] = round(time.time() - t_start)
    log(json.dumps({"evaluation_strings_per_second": summary["evaluation_strings_per_second"],
                    "total_seconds": summary["total_seconds"]}))
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    logf.close()


if __name__ == "__main__":
    main()
