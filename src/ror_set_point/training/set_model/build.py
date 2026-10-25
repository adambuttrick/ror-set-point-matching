"""Make the set model's rows: each row holds a string and its candidate records, in a random order.

    python -m ror_set_point.training.set_model.build --records records --retrieved retriever/topk.jsonl.gz \
        --teacher teacher_judgments/train.jsonl --labelled unseen/train.jsonl \
        --synthetic synthetic.jsonl --held-out unseen/held_out_records.jsonl \
        --eval unseen_validation unseen/validation.jsonl --eval unseen_test unseen/test.jsonl \
        --eval affiliations_validation affiliations/validation.jsonl --out set_data

  stage1.jsonl.gz  {s, c, y, src}      teacher: the retriever's top --k, then the records the teacher judged that are
                                       not in it (positives first), to --cap; y = the teacher's p, 0 where it gave none.
                                       synthetic: the retriever's top --k; y = 1 for the labels, 0 for other records
  stage2.jsonl.gz  {k, s, c, y, fold}  the labeled train split: the retriever's top --k and the lexical top
                                       --lexical-k; y = 1 for the labels; fold = sha256(fold seed|id) mod --folds
  eval.jsonl.gz    {set, k, s, c}      each --eval split, as at run time, with no labels
In training rows, a label that is not among the candidates replaces a random candidate that is not a label (or is
added, when the list is shorter than --cap). Teacher strings that ror_set_point.common.exclusions leaves out are not
used. Labeled and synthetic strings that are equal to an evaluation string are left out. --strict also leaves out
the synthetic strings that name a held-out record, so that no training row has a positive for a held-out record.

--retrieved is the training run's topk.jsonl.gz (ror_set_point.training.retriever.train), which has the queries of
ror_set_point.training.retriever.build. Writes into --out (which must not exist): those files, records.jsonl.gz {ror_id,
text} and manifest.json.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

from ror_set_point.common.exclusions import evaluation_texts, held_out_records, teacher_exclusion
from ror_set_point.common.io import read_jsonl, sha256, write_jsonl, write_manifest
from ror_set_point.common.records import read_records
from ror_set_point.common.text import normalize


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--retrieved", required=True, help="the retriever's topk.jsonl.gz")
    ap.add_argument("--teacher", required=True, help="the teacher_judgments train split")
    ap.add_argument("--labelled", required=True, help="the affiliations_unseen_institutions train split")
    ap.add_argument("--synthetic", required=True, help="synthetic strings (the synthetic_non_latin format)")
    ap.add_argument("--held-out", required=True, help="held_out_records.jsonl")
    ap.add_argument("--eval", nargs=2, action="append", required=True, metavar=("NAME", "PATH"))
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lexical-k", type=int, default=10)
    ap.add_argument("--cap", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1363)
    ap.add_argument("--fold-seed", default="20261022")
    ap.add_argument("--workers", type=int, default=1, help="processes for the lexical index")
    ap.add_argument("--strict", action="store_true", help="leave out synthetic strings with a held-out record")
    ap.add_argument("--out", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    from ror_set_point.inference.lexical import LexicalIndex
    os.makedirs(a.out)
    rng = random.Random(a.seed)
    counts = collections.Counter()
    held = held_out_records(a.held_out)
    eval_norm = evaluation_texts(p for _, p in a.eval)
    write_jsonl(os.path.join(a.out, "records.jsonl.gz"),
                ({"ror_id": r["ror_id"], "text": r["text"]} for r in read_records(a.records)))
    top = {(d["set"], d["id"]): [sys.intern(x) for x in d["ids"][:a.k]] for d in read_jsonl(a.retrieved)}
    index = LexicalIndex(a.records)
    texts = [r["text"] for r in read_jsonl(a.labelled)] + [r["text"] for _, p in a.eval for r in read_jsonl(p)]
    lexical = dict(zip(texts, index.top_many(texts, a.lexical_k, a.workers)))

    def cands(base, extra, gold=()):
        c = list(dict.fromkeys(base + extra))[:a.cap]
        for g in gold:
            if g not in c:
                if len(c) < a.cap:
                    c.append(g)
                else:
                    c[rng.choice([i for i, x in enumerate(c) if x not in gold])] = g
                counts["labels_inserted"] += 1
        rng.shuffle(c)
        return c

    def keep(s, src):
        if normalize(s) in eval_norm:
            counts[f"{src}_left_out: equal to an evaluation string"] += 1
            return False
        return True

    def stage1():
        for r in read_jsonl(a.teacher):
            why = teacher_exclusion(r, held, eval_norm)
            if why:
                counts[f"teacher_left_out: {why}"] += 1
                continue
            base = top[("teacher", r["id"])]
            p = {j["ror_id"]: j["p"] for j in r["judgments"]}
            outside = [x for x in p if x not in base]
            rng.shuffle(outside)
            c = cands(base, sorted(outside, key=lambda x: p[x] < 0.5), [x for x in p if p[x] >= 0.5])
            counts["stage1_teacher"] += 1
            yield {"s": r["text"], "c": c, "y": [p.get(x, 0.0) for x in c], "src": "teacher"}
        for n, t in enumerate(read_jsonl(a.synthetic)):
            if a.strict and set(t["ror_ids"]) & held:
                counts["synthetic_left_out: names a held-out record"] += 1
                continue
            if not keep(t["text"], "synthetic"):
                continue
            gold = set(t["ror_ids"])
            c = cands(top[("synthetic", n)], [], t["ror_ids"])
            counts["stage1_synthetic"] += 1
            yield {"s": t["text"], "c": c, "y": [float(x in gold) for x in c], "src": "synthetic"}

    def stage2():
        for r in read_jsonl(a.labelled):
            if not keep(r["text"], "labelled"):
                continue
            gold = set(r["ror_ids"])
            c = cands(top[("labelled", r["id"])], lexical[r["text"]], r["ror_ids"])
            fold = int(hashlib.sha256(f"{a.fold_seed}|{r['id']}".encode()).hexdigest(), 16) % a.folds
            counts[f"stage2_fold{fold}"] += 1
            yield {"k": r["id"], "s": r["text"], "c": c, "y": [float(x in gold) for x in c], "fold": fold}

    def evaluation():
        for name, path in a.eval:
            for r in read_jsonl(path):
                c = cands(top[(name, r["id"])], lexical[r["text"]])
                counts[f"eval_{name}"] += 1
                counts[f"eval_{name}_candidates"] += len(c)
                yield {"set": name, "k": r["id"], "s": r["text"], "c": c}

    write_jsonl(os.path.join(a.out, "stage1.jsonl.gz"), stage1())
    write_jsonl(os.path.join(a.out, "stage2.jsonl.gz"), stage2())
    write_jsonl(os.path.join(a.out, "eval.jsonl.gz"), evaluation())
    write_manifest(a.out, eval_sets=[n for n, _ in a.eval], folds=a.folds,
                   settings={x: getattr(a, x) for x in ("k", "lexical_k", "cap", "seed", "fold_seed", "strict")},
                   inputs={x: sha256(getattr(a, x)) for x in ("retrieved", "teacher", "labelled", "synthetic",
                                                              "held_out")},
                   counts=dict(sorted(counts.items())))
    if a.verbose:
        print(json.dumps(dict(sorted(counts.items())), indent=1))


if __name__ == "__main__":
    main()
