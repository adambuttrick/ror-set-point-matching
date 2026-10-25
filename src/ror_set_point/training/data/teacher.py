"""Change the teacher judgments of openalex-affiliation-matcher v3.0.0 (train_v1_jev_labels.jsonl.gz, one row per
string and OpenAlex institution) into the teacher_judgments config: one row per string, with ROR ids.

    python -m ror_set_point.training.data.teacher --judgments train_v1_jev_labels.jsonl.gz --records records \
        --out teacher

An OpenAlex id goes to the ROR id of its institution, resolved in the records folder (a withdrawn record goes to its
successor). A judgment with no ROR id is dropped. When two judgments of a string get the same ROR id, the higher p
stays. Writes --out/train.jsonl and --out/validation.jsonl ({id, text, judgments [{ror_id, p}]}) and manifest.json.
"""
import argparse
import collections
import json
import os

from ror_set_point.common.io import read_jsonl, sha256, write_jsonl, write_manifest
from ror_set_point.common.records import Records, records_file


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judgments", required=True, help="rows {split, set, k, s, i (OpenAlex id), p}")
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out)
    ids = Records(a.records)
    counts = collections.Counter()
    by = {}  # (split, set, k) -> [text, {ror_id: p}]; the pairs of one string are not together in the file
    for r in read_jsonl(a.judgments):
        key = (r["split"], r["set"], r["k"])
        if key not in by:
            by[key] = [r["s"], {}]
        counts["pairs"] += 1
        x = ids.from_openalex(r["i"], counts, "pairs")
        if x is None:
            continue
        pairs = by[key][1]
        counts["pairs_merged"] += x in pairs
        pairs[x] = max(pairs.get(x, 0.0), r["p"])
    split_name = {"train": "train", "val": "validation"}
    for split in ("train", "val"):
        rows = ({"id": k, "text": s, "judgments": [{"ror_id": x, "p": p} for x, p in pairs.items()]}
                for (sp, _, k), (s, pairs) in by.items() if sp == split)
        counts[f"strings_{split_name[split]}"] = write_jsonl(os.path.join(a.out, f"{split_name[split]}.jsonl"), rows)
    write_manifest(a.out, inputs={"judgments": sha256(a.judgments),
                                  "records": sha256(records_file(a.records))},
                   counts=dict(sorted(counts.items())))
    if a.verbose:
        print(json.dumps(dict(sorted(counts.items())), indent=1))


if __name__ == "__main__":
    main()
