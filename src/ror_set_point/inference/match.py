"""Match affiliation strings to ROR records. The retriever and the lexical index give the candidates. The set model
selects the records that each string names.

    python -m ror_set_point.inference.match run --input strings.jsonl --records records \
        --retriever adambuttrick/ror-set-point-retriever-e5-base \
        --set-model adambuttrick/ror-set-point-set-model-mmbert-base --out answers.jsonl

Modes:
  candidates  writes {id, text, candidates}: the retriever's top --k, then the lexical top --lexical-k that are not
              already there, cut at --cap, in a fixed random order for each string
  answer      reads candidate rows (--input) and writes {id, ror_ids, p}: the candidates with p >= the threshold
  run         candidates, then answer

--input is a JSON Lines file (fields --text-field and --id-field) or a .txt file with one string on each line (the
id is the line number). The input is done in chunks of --chunk-size rows, and the rows of each chunk are added to
--out, in the order of the input. Each distinct string of a chunk is done once. --resume continues a run that
stopped: it keeps the complete rows of --out and does the rest of the input.
"""
import argparse
import contextlib
import itertools
import json
import os
import random
import sys
import time

from ror_set_point.common.io import open_text
from ror_set_point.common.records import read_records


def order(cands, text, seed):
    """The candidates in a random order that depends only on the seed and the string."""
    c = list(cands)
    random.Random(f"{seed}:{text}").shuffle(c)
    return c


def unique(keys):
    """The distinct keys in order, and the place of each key among them."""
    u = {}
    at = [u.setdefault(x, len(u)) for x in keys]
    return list(u), at


class Matcher:
    def __init__(self, records_dir, retriever, set_model, device=None, k=32, lexical_k=10, cap=42, seed=1363,
                 record_vectors=None, **set_model_kw):
        """records_dir: a records folder. retriever, set_model: a model folder or Hub repository (set_model may be
        None for candidates only). set_model_kw: compile for SetMatcher."""
        from .lexical import LexicalIndex
        from .retriever import Retriever
        from .set_model import SetMatcher
        self.k, self.lexical_k, self.cap, self.seed = k, lexical_k, cap, seed
        self.lexical = LexicalIndex(records_dir)
        self.retriever = Retriever(retriever, records_dir, device, vectors_path=record_vectors)
        texts = {r["ror_id"]: r["text"] for r in read_records(records_dir)}
        self.set_model = SetMatcher.from_pretrained(set_model, texts, device, **set_model_kw) if set_model else None

    def candidates(self, texts, workers=1, pool=None):
        """The candidates of each string. Each distinct string is done once. workers: processes for the lexical
        index. pool: processes from self.lexical.pool(), kept between calls."""
        u, at = unique(texts)
        lexical = self.lexical.top_many(u, self.lexical_k, workers, pool=pool)
        out = []
        for text, lex, (ids, _, _) in zip(u, lexical, self.retriever.search(u, self.k)):
            out.append(order(list(dict.fromkeys(ids + lex))[:self.cap], text, self.seed))
        return [out[i] for i in at]

    def match(self, texts, batch_size=32, workers=1, batch_tokens=65536, pool=None):
        """[(ROR id, p)] for each string: the records that it names. Each distinct string is done once."""
        u, at = unique(texts)
        out = self.set_model.predict(list(zip(u, self.candidates(u, workers, pool))), batch_size, batch_tokens)
        return [out[i] for i in at]


def read_rows(path, skip=0):
    with open_text(path) as f:
        for line in itertools.islice(f, skip, None):
            yield json.loads(line)


def read_input(path, text_field, id_field, skip=0):
    if path.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(itertools.islice(f, skip, None), skip):
                yield str(n), line.rstrip("\n")
        return
    for r in read_rows(path, skip):
        yield r[id_field], r[text_field]


def chunks(rows, size, done=0):
    """Lists of up to size rows, cut at multiples of size from the start of the input. done: the number of rows
    before rows, so that a resumed run has the chunks of a run that did not stop."""
    it = iter(rows)
    n = size - done % size
    while part := list(itertools.islice(it, n)):
        yield part
        n = size


def complete_rows(path):
    """Cut a partial last line off path, and return the number of lines left (0 when there is no file)."""
    if not os.path.exists(path):
        return 0
    n = size = 0
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            n, size = n + 1, size + len(line)
    os.truncate(path, size)
    return n


def answer_row(key, p):
    return {"id": key, "ror_ids": [x for x, _ in p], "p": [v for _, v in p]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["candidates", "answer", "run"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--retriever", help="candidates, run: a retriever folder or Hub repository")
    ap.add_argument("--set-model", help="answer, run: a set-model folder or Hub repository")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lexical-k", type=int, default=10)
    ap.add_argument("--cap", type=int, default=42)
    ap.add_argument("--seed", type=int, default=1363)
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=32, help="strings in a padded batch (CPU, MPS, other encoders)")
    ap.add_argument("--batch-tokens", type=int, default=65536, help="tokens in a packed batch (CUDA, flash-attn)")
    ap.add_argument("--no-compile", action="store_true", help="do not torch.compile the set model (packed batches)")
    ap.add_argument("--record-vectors", help="a file that keeps the retriever's record vectors between runs")
    ap.add_argument("--workers", type=int, default=1, help="processes for the lexical index")
    ap.add_argument("--chunk-size", type=int, default=100000, help="input rows that are done and written at a time")
    ap.add_argument("--resume", action="store_true", help="keep the complete rows of --out, and do the rest")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.resume:
        raise SystemExit(f"{a.out} exists (--resume continues it)")
    if a.resume and a.out.endswith(".gz"):
        ap.error("--resume needs an --out that is not gzip")
    if a.chunk_size < 1:
        ap.error("--chunk-size must be at least 1")
    if a.mode != "answer" and not a.retriever:
        ap.error("candidates and run need --retriever")
    if a.mode != "candidates" and not a.set_model:
        ap.error("answer and run need --set-model")
    done = complete_rows(a.out) if a.resume else 0

    if a.mode == "answer":
        from .set_model import SetMatcher
        texts = {r["ror_id"]: r["text"] for r in read_records(a.records)}
        m = SetMatcher.from_pretrained(a.set_model, texts, a.device, compile=not a.no_compile)
        rows, pool = read_rows(a.input, done), contextlib.nullcontext()

        def work(part, pool):
            u, at = unique((r["text"], tuple(r["candidates"])) for r in part)
            pred = m.predict([(s, list(c)) for s, c in u], a.batch_size, a.batch_tokens)
            return [answer_row(r["id"], pred[i]) for r, i in zip(part, at)]
    else:
        m = Matcher(a.records, a.retriever, a.set_model if a.mode == "run" else None, a.device, a.k, a.lexical_k,
                    a.cap, a.seed, a.record_vectors, compile=not a.no_compile)
        rows = read_input(a.input, a.text_field, a.id_field, done)
        pool = m.lexical.pool(a.workers) if a.workers > 1 else contextlib.nullcontext()

        def work(part, pool):
            keys, texts = [k for k, _ in part], [s for _, s in part]
            if a.mode == "candidates":
                return [{"id": k, "text": s, "candidates": c}
                        for k, s, c in zip(keys, texts, m.candidates(texts, a.workers, pool))]
            pred = m.match(texts, a.batch_size, a.workers, a.batch_tokens, pool)
            return [answer_row(k, p) for k, p in zip(keys, pred)]

    with pool as pool, open_text(a.out, "a" if a.resume else "w") as f:
        for part in chunks(rows, a.chunk_size, done):
            t = time.perf_counter()
            f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in work(part, pool)))
            f.flush()
            os.fsync(f.fileno())
            done, t = done + len(part), time.perf_counter() - t
            print(f"{done} rows done: {len(part)} in {t:.1f} s, {len(part) / t:.0f} each second", file=sys.stderr,
                  flush=True)


if __name__ == "__main__":
    main()
