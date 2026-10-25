"""Train the retriever, save it as a sentence-transformers model, and search the queries of the data folder.

    python -m ror_set_point.training.retriever.train --data retriever_data --out retriever

The model is one shared encoder for strings and record texts. Strings get the query prompt and record texts the
passage prompt. With --loader transformers, the encoder is a transformers model with mean pooling; with --loader
sentence-transformers, the base model's own pooling and dense layers are kept. Vectors have length 1.

The loss is InfoNCE at temperature --temperature over every record of the batch. Each string has its positive and
--negatives hard negatives (filled with random records when it has fewer). The other positives of the string are not
negatives. AdamW, linear warm-up for 5% of the steps then linear decay, bf16 autocast on CUDA, gradient clipping 1.0.
--steps 0 skips training (to search with the base model).

Writes into --out (which must not exist): model/ (the sentence-transformers model, fp16, or bf16 with --loader
sentence-transformers), topk.jsonl.gz ({set, id, ids, scores}: the top --k records of each query; --k 0 skips the
search), train_log.txt and summary.json.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from ror_set_point.common.io import check_manifest, read_jsonl, write_jsonl


class Encoder(torch.nn.Module):
    def __init__(self, base, revision, loader):
        super().__init__()
        self.st = loader == "sentence-transformers"
        if self.st:
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer(base, revision=revision, device="cpu")
            self.tok = self.m.tokenizer
        else:
            from transformers import AutoModel, AutoTokenizer
            self.m = AutoModel.from_pretrained(base, revision=revision, add_pooling_layer=False)
            self.tok = AutoTokenizer.from_pretrained(base, revision=revision)

    def hf(self):
        return self.m[0].auto_model if self.st else self.m

    def forward(self, ids, mask):
        if self.st:
            return F.normalize(self.m({"input_ids": ids, "attention_mask": mask})["sentence_embedding"].float(), dim=-1)
        h = self.m(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        m = mask.unsqueeze(-1).float()
        return F.normalize((h * m).sum(1) / m.sum(1).clamp_min(1.0), dim=-1)


def train(model, rows, record_ids, record_text, a, dev, log):
    """One pass over the rows (or --steps steps). Returns the step count and the seconds."""
    rng = random.Random(a.seed)
    tok = model.tok
    total = len(rows) // a.batch_size if a.steps < 0 else a.steps
    log(f"{len(rows):,} training rows; {total:,} steps")
    order = list(range(len(rows)))
    rng.shuffle(order)
    if a.gradient_checkpointing:
        model.hf().gradient_checkpointing_enable()
    model.hf().config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    model.train()
    t0, run_loss, run_acc, n = time.time(), 0.0, 0.0, 0
    for step in range(total):
        B = [rows[i] for i in order[(step * a.batch_size) % len(rows):][:a.batch_size]]
        pid = []
        for r in B:
            negs = rng.sample(r["negs"], a.negatives) if len(r["negs"]) >= a.negatives else list(r["negs"])
            while len(negs) < a.negatives:
                x = record_ids[rng.randrange(len(record_ids))]
                if x not in r["pos_all"]:
                    negs.append(x)
            pid += [r["pos"]] + negs
        eq = tok([a.query_prompt + r["q"] for r in B], padding=True, truncation=True, max_length=a.query_length,
                 return_tensors="pt")
        ep = tok([a.record_prompt + record_text[x] for x in pid], padding=True, truncation=True,
                 max_length=a.record_length, return_tensors="pt")
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            q = model(eq["input_ids"].to(dev), eq["attention_mask"].to(dev))
            p = model(ep["input_ids"].to(dev), ep["attention_mask"].to(dev))
        scores = (q @ p.T) / a.temperature
        where = {}
        for j, x in enumerate(pid):
            where.setdefault(x, []).append(j)
        gold = torch.arange(len(B), device=dev) * (1 + a.negatives)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for i, r in enumerate(B):
            for x in r["pos_all"]:
                for j in where.get(x, ()):
                    if j != i * (1 + a.negatives):
                        mask[i, j] = True
        scores = scores.masked_fill(mask, float("-inf"))
        loss = F.cross_entropy(scores, gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run_loss += loss.item()
        run_acc += (scores.argmax(1) == gold).float().mean().item()
        n += 1
        if (step + 1) % a.log_every == 0 or step + 1 == total:
            el = time.time() - t0
            log(f"step {step + 1}/{total} loss {run_loss / n:.4f} in-batch acc {run_acc / n:.3f} "
                f"lr {sched.get_last_lr()[0]:.2e} {el / (step + 1):.2f} s/step "
                f"eta {(total - step - 1) * el / (step + 1) / 60:.1f} min")
            run_loss, run_acc, n = 0.0, 0.0, 0
    if a.gradient_checkpointing:
        model.hf().gradient_checkpointing_disable()
    return {"steps": total, "train_seconds": round(time.time() - t0)}


def fix_gemma_config(path, base, revision):
    """transformers 4.57 halves sliding_window when a bidirectional Gemma 3 config loads (512 -> 257), and saves the
    halved value. Write the base model's raw value back, so that a reload gives the same model."""
    from huggingface_hub import hf_hub_download
    p = os.path.join(path, "config.json")
    with open(p, encoding="utf-8") as f:
        c = json.load(f)
    if c.get("model_type") == "gemma3_text" and c.get("use_bidirectional_attention"):
        with open(hf_hub_download(base, "config.json", revision=revision), encoding="utf-8") as f:
            c["sliding_window"] = json.load(f)["sliding_window"]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(c, f, indent=2)


def to_sentence_transformer(model, a):
    """The trained encoder as a sentence-transformers model with the prompts "query" and "passage"."""
    from sentence_transformers import SentenceTransformer, models
    if model.st:
        st = model.m
    else:
        rev = {"revision": a.base_revision}
        tr = models.Transformer(a.base, max_seq_length=a.search_query_length, model_args=rev, tokenizer_args=rev,
                                config_args=rev)
        missing, unexpected = tr.auto_model.load_state_dict(model.m.state_dict(), strict=False)
        assert not unexpected and all(x.startswith("pooler.") for x in missing), (missing, unexpected)
        st = SentenceTransformer(modules=[tr, models.Pooling(tr.get_word_embedding_dimension(), pooling_mode="mean"),
                                          models.Normalize()], device="cpu")
    st.prompts = dict(st.prompts or {}, query=a.query_prompt, passage=a.record_prompt)
    st.default_prompt_name = None
    return st


@torch.no_grad()
def reload_check(model, path, queries, records, a, dev):
    """The smallest cosine between the training encoder's vectors and the saved model's vectors."""
    from sentence_transformers import SentenceTransformer
    model.eval()
    saved = SentenceTransformer(path, device=str(dev))
    out = {}
    for name, texts, prompt, prefix, maxlen in (("queries", queries, "query", a.query_prompt, a.search_query_length),
                                                ("records", records, "passage", a.record_prompt, a.record_length)):
        e = model.tok([prefix + t for t in texts], padding=True, truncation=True, max_length=maxlen,
                      return_tensors="pt").to(dev)
        ref = model(e["input_ids"], e["attention_mask"]).float().cpu()
        saved.max_seq_length = maxlen
        got = saved.encode(texts, prompt_name=prompt, convert_to_tensor=True, normalize_embeddings=True).float().cpu()
        out[f"min_cosine_{name}"] = round(float((ref * got).sum(1).min()), 6)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="a folder from ror_set_point.training.retriever.build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="intfloat/multilingual-e5-base")
    ap.add_argument("--base-revision")
    ap.add_argument("--loader", choices=["transformers", "sentence-transformers"], default="transformers")
    ap.add_argument("--query-prompt", default="query: ")
    ap.add_argument("--record-prompt", default="passage: ")
    ap.add_argument("--steps", type=int, default=-1, help="-1: one epoch; 0: no training")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--negatives", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--query-length", type=int, default=96)
    ap.add_argument("--record-length", type=int, default=128)
    ap.add_argument("--search-query-length", type=int, default=512)
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--k", type=int, default=64, help="search depth for the queries (0: no search)")
    ap.add_argument("--seed", type=int, default=1363)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="read only this many rows of each file (a smoke test)")
    ap.add_argument("--log-every", type=int, default=100)
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(a.out)
    logf = open(os.path.join(a.out, "train_log.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t_start = time.time()
    dev = torch.device(a.device)
    settings = {k: v for k, v in vars(a).items() if k not in ("data", "out")}
    log(f"device {torch.cuda.get_device_name(0) if dev.type == 'cuda' else dev.type}; settings {json.dumps(settings)}")
    manifest = check_manifest(a.data, ("records.jsonl.gz", "train.jsonl.gz", "queries.jsonl.gz"))
    records = list(read_jsonl(os.path.join(a.data, "records.jsonl.gz")))
    record_ids = [r["ror_id"] for r in records]
    record_text = {r["ror_id"]: r["text"] for r in records}
    torch.manual_seed(a.seed)
    model = Encoder(a.base, a.base_revision, a.loader).to(dev)
    summary = {"settings": settings, "records": len(records), "data_sha256": manifest["sha256"]}
    if a.steps != 0:
        rows = list(read_jsonl(os.path.join(a.data, "train.jsonl.gz"), a.limit))
        summary.update(train(model, rows, record_ids, record_text, a, dev, log), train_rows=len(rows))

    path = os.path.join(a.out, "model")
    st = to_sentence_transformer(model, a)
    (st.to(torch.bfloat16) if model.st else st.half()).save(path)
    fix_gemma_config(path, a.base, a.base_revision)
    queries = list(read_jsonl(os.path.join(a.data, "queries.jsonl.gz"), a.limit))
    sample = [q["text"] for q in queries[:32]] + sorted((q["text"] for q in queries), key=len)[-32:]
    st.to(torch.float32)
    summary["reload_check"] = reload_check(model, path, sample, [r["text"] for r in records[:64]], a, dev)
    log(f"saved {path}; {json.dumps(summary['reload_check'])}")

    if a.k:
        from ror_set_point.inference.retriever import Retriever
        del model, st
        r = Retriever(path, a.data, a.device, query_length=a.search_query_length, record_length=a.record_length)
        t0 = time.time()
        res = r.search([q["text"] for q in queries], a.k)
        write_jsonl(os.path.join(a.out, "topk.jsonl.gz"),
                    ({"set": q["set"], "id": q["id"], "ids": ids, "scores": sc} for q, (ids, sc, _) in zip(queries, res)))
        summary["search_seconds"] = round(time.time() - t0)
    summary["total_seconds"] = round(time.time() - t_start)
    log(json.dumps(summary["reload_check"] | {"total_seconds": summary["total_seconds"]}))
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    logf.close()


if __name__ == "__main__":
    main()
