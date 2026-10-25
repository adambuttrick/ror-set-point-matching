"""The retriever: a dual encoder that finds candidate ROR records for an affiliation string. The model is a
sentence-transformers model with the prompts "query" (strings) and "passage" (record texts). A high dot product
means a good candidate.

    python -m ror_set_point.inference.retriever --model adambuttrick/ror-set-point-retriever-e5-base \
        --records records --queries strings.jsonl --k 32 --out top32.jsonl.gz

Each output row is {id, ids, scores}, with "set" when the query row has one, and "positive_scores" (the scores of
the records in --positives-field, in that order) when that option is given.

Without --queries, the command only encodes the record texts into --record-vectors (the matcher's index):

    python -m ror_set_point.inference.retriever --model adambuttrick/ror-set-point-retriever-e5-base \
        --records records --record-vectors record_vectors.pt
"""
import argparse
import os
import sys
import time

import torch

from ror_set_point.common.io import read_jsonl, sha256, write_jsonl
from ror_set_point.common.records import read_records, records_file


def load_model(name_or_path, device):
    """The model in its saved precision on CUDA (fp16, or bf16 for EmbeddingGemma, which fails in fp16), and in fp32
    on other devices."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name_or_path, device=str(device), model_kwargs={"dtype": "auto"})
    if device.type != "cuda":
        m.float()
    return m.eval()


def encode(model, texts, prompt_name, max_length, batch_size):
    """Unit vectors for texts, as a tensor on the model's device."""
    model.max_seq_length = max_length
    v = model.encode(texts, prompt_name=prompt_name, batch_size=batch_size, convert_to_tensor=True,
                     normalize_embeddings=True, show_progress_bar=False)
    return v.half() if v.device.type == "cuda" else v.float()


class Retriever:
    def __init__(self, model, records_dir, device=None, batch_size=256, record_length=128, query_length=512,
                 vectors_path=None):
        """model: a model name or folder, or a loaded SentenceTransformer. vectors_path: a file that keeps the
        record vectors between runs (written when it does not exist)."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_model(model, self.device) if isinstance(model, str) else model
        self.batch_size, self.query_length = batch_size, query_length
        rows = list(read_records(records_dir))
        self.ids = [r["ror_id"] for r in rows]
        self.index = {x: j for j, x in enumerate(self.ids)}
        key = {"model": model if isinstance(model, str) else None, "records": sha256(records_file(records_dir)),
               "record_length": record_length}
        saved = torch.load(vectors_path, map_location="cpu") if vectors_path and os.path.exists(vectors_path) else None
        if saved and saved["key"] == key:
            self.vectors = saved["vectors"].to(self.device)
        else:
            self.vectors = encode(self.model, [r["text"] for r in rows], "passage", record_length, batch_size)
            if vectors_path:
                torch.save({"key": key, "vectors": self.vectors.cpu()}, vectors_path)

    def search(self, texts, k, positives=None, chunk=1024):
        """Yield (ids, scores, positive scores) for each text: the top k records by dot product. positives: one
        list of ROR ids for each text, or None."""
        q = encode(self.model, texts, "query", self.query_length, self.batch_size)
        for s in range(0, len(texts), chunk):
            S = q[s:s + chunk] @ self.vectors.T
            v, i = S.float().topk(k, dim=1)
            for n, (vv, ii) in enumerate(zip(v.cpu().tolist(), i.cpu().tolist())):
                pos = None
                if positives is not None:
                    pos = [round(float(S[n, self.index[x]]), 4) for x in positives[s + n] if x in self.index]
                yield [self.ids[x] for x in ii], [round(x, 4) for x in vv], pos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="a model folder or Hub repository")
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--queries", help="a JSON Lines file of strings (without it, only write --record-vectors)")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--key-field", default="id", help='a field, or "line" for the line number')
    ap.add_argument("--positives-field", help="also give the scores of these records (for example ror_ids)")
    ap.add_argument("--k", type=int)
    ap.add_argument("--out")
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--query-length", type=int, default=512)
    ap.add_argument("--record-length", type=int, default=128)
    ap.add_argument("--record-vectors", help="a file that keeps the record vectors between runs")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not a.queries:
        if not a.record_vectors:
            ap.error("give --queries, or --record-vectors to only encode the records")
        t = time.perf_counter()
        r = Retriever(a.model, a.records, a.device, a.batch_size, a.record_length, a.query_length, a.record_vectors)
        print(f"{len(r.ids)} record vectors in {a.record_vectors} ({time.perf_counter() - t:.0f} s)", file=sys.stderr)
        return
    if a.k is None or not a.out:
        ap.error("--queries needs --k and --out")
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    r = Retriever(a.model, a.records, a.device, a.batch_size, a.record_length, a.query_length, a.record_vectors)
    Q = list(read_jsonl(a.queries, a.limit))
    pos = [q[a.positives_field] for q in Q] if a.positives_field else None

    def rows():
        for n, (q, (ids, sc, ps)) in enumerate(zip(Q, r.search([q[a.text_field] for q in Q], a.k, pos))):
            out = {"set": q["set"]} if "set" in q else {}
            out.update({"id": n if a.key_field == "line" else q[a.key_field], "ids": ids, "scores": sc})
            if ps is not None:
                out["positive_scores"] = ps
            yield out
    write_jsonl(a.out, rows())


if __name__ == "__main__":
    main()
