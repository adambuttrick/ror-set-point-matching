"""Mine hard negatives for the synthetic strings from a first retriever's search.

    python -m ror_set_point.inference.retriever --model first/model --records records --queries synthetic.jsonl \
        --key-field line --positives-field ror_ids --k 64 --out synthetic_top64.jsonl.gz
    python -m ror_set_point.training.retriever.mine --synthetic synthetic.jsonl --records records \
        --search synthetic_top64.jsonl.gz --out mined.jsonl.gz

For each synthetic string, the search's top 64 records are read in rank order. These records are dropped:
  - the labels of the string;
  - each record that is related to a label (parent, child, related, predecessor or successor);
  - each record with the same normalized name as the source record;
  - each record with a score of --ratio x the score of the source record or more (probably a match without a label).
The first --keep records that remain are the negatives. Writes {id (line number), negs} for each string.
"""
import argparse
import collections
import json
import os

from ror_set_point.common.io import read_jsonl, write_jsonl
from ror_set_point.common.records import read_records
from ror_set_point.common.text import normalize


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--synthetic", required=True, help="synthetic strings (the synthetic_non_latin format)")
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--search", required=True, help="the retriever's output for the synthetic strings, with "
                                                    "positive_scores and the line number as id")
    ap.add_argument("--ratio", type=float, default=0.95)
    ap.add_argument("--keep", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    records = {r["ror_id"]: r for r in read_records(a.records)}
    by_name = collections.defaultdict(set)
    for x, r in records.items():
        by_name[normalize(r["name"])].add(x)
    synthetic = list(read_jsonl(a.synthetic))
    top = {d["id"]: d for d in read_jsonl(a.search)}
    if len(top) != len(synthetic):
        raise SystemExit(f"{len(top)} searched strings for {len(synthetic)} synthetic strings")
    counts = collections.Counter()

    def rows():
        for n, t in enumerate(synthetic):
            d = top[n]
            labels = set(t["ror_ids"])
            related = {x["id"] for y in labels for xs in records[y]["relations"].values() for x in xs}
            same = by_name[normalize(records[t["ror_id"]]["name"])]
            bar = a.ratio * d["positive_scores"][0]
            negs = []
            for x, s in zip(d["ids"], d["scores"]):
                why = ("label" if x in labels else "related" if x in related else "same name" if x in same
                       else "score" if s >= bar else None)
                if why:
                    counts[f"dropped: {why}"] += 1
                    continue
                negs.append(x)
                if len(negs) == a.keep:
                    break
            counts["strings with fewer than 3 negatives"] += len(negs) < 3
            counts["negatives"] += len(negs)
            yield {"id": n, "negs": negs}
    counts["strings"] = write_jsonl(a.out, rows())
    if a.verbose:
        print(json.dumps(dict(sorted(counts.items())), indent=1))


if __name__ == "__main__":
    main()
