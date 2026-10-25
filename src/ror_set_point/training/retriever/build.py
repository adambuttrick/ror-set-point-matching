"""Make the retriever's training rows and search queries.

    python -m ror_set_point.training.retriever.build --records records \
        --teacher teacher_judgments/train.jsonl --labelled unseen/train.jsonl --held-out unseen/held_out_records.jsonl \
        --eval unseen_validation unseen/validation.jsonl --eval unseen_test unseen/test.jsonl \
        --eval affiliations_validation affiliations/validation.jsonl \
        --synthetic synthetic.jsonl --synthetic-negatives mined.jsonl.gz --out retriever_data

Training rows {q, pos, negs, pos_all, src}, one for each (string, positive record). pos_all holds every positive of
the string; training does not use them as negatives of that string.
  teacher    teacher strings with p >= 0.5 as positives and p < 0.1 as hard negatives. Strings that
             ror_set_point.common.exclusions leaves out are not used.
  labelled   the labeled train split. Hard negatives are the candidates in --labelled-negatives ({id, ror_ids}),
             or else the lexical top 10, without the labels. Each row is repeated --labelled-repeat times.
  synthetic  (--synthetic) positives are the labels. Hard negatives are the mined ones (--synthetic-negatives), or 3
             random records of the same country and type. --strict leaves out strings with a held-out record.
Labeled and synthetic strings that are equal to an evaluation string are left out.

Queries {set, id, text}: the teacher, labeled and synthetic strings (sets "teacher", "labelled", "synthetic"; the id
of a synthetic string is its line number) and each --eval split under its name. No label of an evaluation string is
written.

Writes into --out (which must not exist): records.jsonl.gz {ror_id, text}, train.jsonl.gz, queries.jsonl.gz and
manifest.json.
"""
import argparse
import collections
import json
import os
import random

from ror_set_point.common.exclusions import evaluation_texts, held_out_records, teacher_exclusion
from ror_set_point.common.io import read_jsonl, sha256, write_jsonl, write_manifest
from ror_set_point.common.records import read_records
from ror_set_point.common.text import normalize


def labelled_negatives(a, rows):
    """{id: candidate ROR ids} for the labeled strings."""
    if a.labelled_negatives:
        return {r["id"]: r["ror_ids"] for r in read_jsonl(a.labelled_negatives)}
    from ror_set_point.inference.lexical import LexicalIndex
    ix = LexicalIndex(a.records)
    return {r["id"]: ix.top(r["text"], 10) for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--teacher", required=True, help="the teacher_judgments train split")
    ap.add_argument("--labelled", required=True, help="the affiliations_unseen_institutions train split")
    ap.add_argument("--held-out", required=True, help="held_out_records.jsonl")
    ap.add_argument("--eval", nargs=2, action="append", required=True, metavar=("NAME", "PATH"))
    ap.add_argument("--labelled-negatives", help="candidates for the labeled strings: {id, ror_ids} (default: the "
                    "lexical top 10)")
    ap.add_argument("--synthetic", help="synthetic strings (the synthetic_non_latin format)")
    ap.add_argument("--synthetic-negatives", help="mined negatives: {id (line number), negs}")
    ap.add_argument("--strict", action="store_true", help="leave out synthetic strings with a held-out record")
    ap.add_argument("--labelled-repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1363)
    ap.add_argument("--out", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out)
    rng = random.Random(a.seed)
    counts = collections.Counter()
    held = held_out_records(a.held_out)
    eval_norm = evaluation_texts(p for _, p in a.eval)
    records = list(read_records(a.records))
    write_jsonl(os.path.join(a.out, "records.jsonl.gz"), ({"ror_id": r["ror_id"], "text": r["text"]} for r in records))
    train, queries = [], []

    def add(q, pos_all, negs, src):
        negs = [x for x in dict.fromkeys(negs) if x not in pos_all]
        for p in pos_all:
            counts[f"rows_{src}"] += 1
            train.append({"q": q, "pos": p, "negs": negs, "pos_all": pos_all, "src": src})

    for r in read_jsonl(a.teacher):
        why = teacher_exclusion(r, held, eval_norm)
        if why:
            counts[f"teacher_left_out: {why}"] += 1
            continue
        counts["teacher_strings"] += 1
        queries.append({"set": "teacher", "id": r["id"], "text": r["text"]})
        pos = [j["ror_id"] for j in r["judgments"] if j["p"] >= 0.5]
        if pos:
            add(r["text"], pos, [j["ror_id"] for j in r["judgments"] if j["p"] < 0.1], "teacher")
        else:
            counts["teacher_strings_without_positive"] += 1

    rows = [r for r in read_jsonl(a.labelled)]
    kept = [r for r in rows if normalize(r["text"]) not in eval_norm]
    counts["labelled_left_out: equal to an evaluation string"] = len(rows) - len(kept)
    negs = labelled_negatives(a, kept)
    for r in kept:
        counts["labelled_strings"] += 1
        queries.append({"set": "labelled", "id": r["id"], "text": r["text"]})
        if r["ror_ids"]:
            for _ in range(a.labelled_repeat):
                add(r["text"], r["ror_ids"], negs.get(r["id"], []), "labelled")

    if a.synthetic:
        first = lambda r: ((r["places"] or [{}])[0].get("cc"), r["types"][0] if r["types"] else "")
        by, group = collections.defaultdict(list), {}
        for r in records:
            by[first(r)].append(r["ror_id"])
            group[r["ror_id"]] = first(r)
        mined = {r["id"]: r["negs"] for r in read_jsonl(a.synthetic_negatives)} if a.synthetic_negatives else None
        for n, t in enumerate(read_jsonl(a.synthetic)):
            if a.strict and set(t["ror_ids"]) & held:
                counts["synthetic_left_out: names a held-out record"] += 1
                continue
            if normalize(t["text"]) in eval_norm:
                counts["synthetic_left_out: equal to an evaluation string"] += 1
                continue
            if mined is not None:
                sn = mined[n]
            else:
                pool = [x for x in by[group[t["ror_id"]]] if x not in t["ror_ids"]]
                sn = rng.sample(pool, min(3, len(pool)))
            add(t["text"], t["ror_ids"], sn, "synthetic")
            counts["synthetic_strings"] += 1
            queries.append({"set": "synthetic", "id": n, "text": t["text"]})

    for name, path in a.eval:
        for r in read_jsonl(path):
            queries.append({"set": name, "id": r["id"], "text": r["text"]})
            counts[f"queries_{name}"] += 1
    write_jsonl(os.path.join(a.out, "train.jsonl.gz"), train)
    write_jsonl(os.path.join(a.out, "queries.jsonl.gz"), queries)
    write_manifest(a.out, eval_sets=[n for n, _ in a.eval], seed=a.seed, labelled_repeat=a.labelled_repeat,
                   strict=a.strict, labelled_negatives="file" if a.labelled_negatives else "lexical top 10",
                   synthetic_negatives="mined" if a.synthetic_negatives else "random" if a.synthetic else None,
                   inputs={x: sha256(getattr(a, x)) for x in ("teacher", "labelled", "held_out", "synthetic",
                                                              "synthetic_negatives", "labelled_negatives")
                           if getattr(a, x)},
                   counts=dict(sorted(counts.items())))
    if a.verbose:
        print(json.dumps(dict(sorted(counts.items())), indent=1))


if __name__ == "__main__":
    main()
