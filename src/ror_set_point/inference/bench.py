"""Measure the speed of the set model, and compare its scores with the reference way.

    python -m ror_set_point.inference.bench --set-model adambuttrick/ror-set-point-set-model-mmbert-base-mask \
        --records records --input candidates.jsonl --batch-tokens 32768,65536,131072 --reference

--input holds candidate rows: {text, candidates} (match candidates) or {s, c} (the eval rows of the set model's
training folder). Each time covers SetMatcher.scores from the strings and the candidate ids to the scores on the
CPU: tokenizing, batching, the model and the copy back. A first pass on --warmup rows (which also compiles, with
--compile) is not timed.

--reference also runs the reference way: the encoder in fp32 with SDPA attention, padded batches of --batch-size
strings and bf16 autocast, as in training. It gives its speed, and the
differences in the scores, the probabilities and the answers.

TFLOP/s counts the useful work: the encoder's matrix products on the real tokens, and attention inside each input
(a window of local_attention + 1 tokens in the local layers). The bf16 dense peak of an H200 is about 989 TFLOP/s.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from ror_set_point.common.io import read_jsonl
from ror_set_point.common.records import read_records
from .set_model import Packer, SetMatcher, SetModel, calibrate, load_encoder, model_folder, scores


def useful_flops(conf, lengths):
    D, I, L = conf.hidden_size, conf.intermediate_size, conf.num_hidden_layers
    T = np.asarray(lengths, dtype=np.float64)
    flops = 2 * L * (D * 3 * D + D * D + D * 2 * I + I * D) * T.sum()
    every = getattr(conf, "global_attn_every_n_layers", 1)
    n_global = len(range(0, L, every))
    window = getattr(conf, "local_attention", None)
    flops += 4 * D * n_global * (T * T).sum()
    if window and L > n_global:
        flops += 4 * D * (L - n_global) * (T * np.minimum(T, window + 1)).sum()
    return float(flops)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, repeat):
    """The result of fn and its best time over repeat runs."""
    best, out = float("inf"), None
    for _ in range(repeat):
        sync(device)
        t = time.perf_counter()
        out = fn()
        sync(device)
        best = min(best, time.perf_counter() - t)
    return out, best


def report(name, n, seconds, flops, peak):
    tf = flops / seconds / 1e12
    row = {"run": name, "strings": n, "seconds": round(seconds, 3), "strings_per_second": round(n / seconds, 1),
           "tflops": round(tf, 1), "of_peak": round(tf / peak, 3)}
    print(json.dumps(row), flush=True)
    return row


def compare(items, z, zr, a, b, t):
    """The differences between scores z and the reference scores zr."""
    dz = np.array([abs(u - v) for x, y in zip(z, zr) for u, v in zip(x, y)])
    dp, flips, strings = [], 0, 0
    for x, y in zip(z, zr):
        p = [calibrate(v, a, b) for v in x]
        pr = [calibrate(v, a, b) for v in y]
        dp += [abs(u - v) for u, v in zip(p, pr)]
        f = sum((u >= t) != (v >= t) for u, v in zip(p, pr))
        flips += f
        strings += f > 0
    return {"candidates": len(dz), "max_abs_score_diff": round(float(dz.max()), 4),
            "mean_abs_score_diff": round(float(dz.mean()), 5), "max_abs_p_diff": round(max(dp), 4),
            "decisions_changed": int(flips), "answers_changed": int(strings), "strings": len(items)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set-model", required=True, help="a set-model folder or Hub repository")
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--input", required=True, help="candidate rows")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--batch-tokens", default="65536", help="tokens in a packed batch (a comma list)")
    ap.add_argument("--batch-size", type=int, default=32, help="strings in a padded batch")
    ap.add_argument("--compile", default="on", help="on, off or both (a comma list)")
    ap.add_argument("--reference", action="store_true", help="also run and compare the reference way")
    ap.add_argument("--warmup", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=3, help="timed runs of each setting; the best is reported")
    ap.add_argument("--peak-tflops", type=float, default=989.0)
    ap.add_argument("--out", help="write the results here as JSON")
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    rows = list(read_jsonl(a.input, a.limit))
    items = [(r["text"], r["candidates"]) if "candidates" in r else (r["s"], r["c"]) for r in rows]
    texts = {r["ror_id"]: r["text"] for r in read_records(a.records)}
    path = model_folder(a.set_model)
    out = {"runs": []}
    z_of = {}
    for comp in a.compile.split(","):
        m = SetMatcher(path, texts, a.device, compile={"off": False, "on": True}[comp])
        dev, conf = m.device, m.model.enc.config
        if "env" not in out:
            p = m.packer.plan(items)
            lengths = p.length[p.n > 0]
            flops = useful_flops(conf, lengths)
            out["env"] = {"device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type,
                          "torch": torch.__version__, "packed": m.packed, "strings": len(items),
                          "tokens_per_string": round(float(lengths.mean()), 1),
                          "useful_tflop_per_string": round(flops / max(1, len(lengths)) / 1e12, 3)}
            print(json.dumps(out["env"]), flush=True)
        name = f"{'packed' if m.packed else 'padded'} compile {comp}"
        t = time.perf_counter()
        m.scores(items[:a.warmup], a.batch_size, int(a.batch_tokens.split(",")[0]))
        sync(dev)
        print(json.dumps({"run": name, "warmup_seconds": round(time.perf_counter() - t, 1),
                          "attention": conf._attn_implementation}), flush=True)
        for bt in (int(x) for x in a.batch_tokens.split(",")):
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            z, s = timed(lambda: m.scores(items, a.batch_size, bt), dev, a.repeat)
            row = report(f"{name} tokens {bt}" if m.packed else f"{name} batch {a.batch_size}", len(items), s,
                         flops, a.peak_tflops)
            if dev.type == "cuda":
                row["peak_memory_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 1)
            out["runs"].append(row)
            z_of[name] = z
            if not m.packed:
                break
        cfg = m.cfg
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if a.reference:
        from safetensors.torch import load_file
        ref = SetModel(load_encoder(path))
        ref.head.load_state_dict({k: v.float() for k, v in load_file(os.path.join(path, "head.safetensors")).items()})
        ref.to(dev).eval()
        packer = Packer(AutoTokenizer.from_pretrained(path), texts, cfg["string_max_tokens"],
                        cfg["card_max_tokens"], cfg["max_tokens"])
        scores(ref, packer, items[:64], dev, a.batch_size)
        zr, s = timed(lambda: scores(ref, packer, items, dev, a.batch_size), dev, 1)
        out["runs"].append(report(f"reference batch {a.batch_size}", len(items), s, flops, a.peak_tflops))
        out["agreement"] = {}
        for name, z in z_of.items():
            cal = cfg["calibration"]
            out["agreement"][name] = compare(items, z, zr, cal["a"], cal["b"], cfg["threshold"])
            print(json.dumps({"run": name, **out["agreement"][name]}), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
