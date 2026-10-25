"""Divide the labeled training strings by institution: the affiliations_unseen_institutions config.

    python -m ror_set_point.training.data.split --records records \
        --affiliations-train affiliations/train.jsonl --openalex-sample-train openalex_sample/train.jsonl \
        --affiliations-validation affiliations/validation.jsonl \
        --teacher teacher_judgments/train.jsonl teacher_judgments/validation.jsonl --seed 20261022 --out unseen

The labeled strings are the affiliations train split and the openalex_sample train split. A label on a withdrawn
record goes to its successor (a label with no successor is dropped and listed in excluded_labels.jsonl).

1. Each ROR record of the labels goes into a stratum: type x script x frequency.
2. In each stratum, --share of the records are held out, in sha256 order of seed and ROR id. The count is rounded up
   or down at random, so that small strata are not all rounded down.
3. Every string that names a held-out record leaves the train split. So does --share of the strings that name no
   record (in each source x script stratum).
4. Strings that name the same held-out record make one group. In each source x script stratum, the groups go in
   sha256 order to the validation or the test split, whichever has fewer strings at that time.
5. Duplicates go out. A string that is equal to a moved string leaves the train split. A moved string that is equal
   to an affiliations validation string or to a teacher string is dropped.

Writes into --out (which must not exist): train.jsonl, validation.jsonl, test.jsonl ({id, text, ror_ids, origin,
source, script}), held_out_records.jsonl ({ror, type, script, freq, stratum}), excluded.jsonl (strings in no split,
with the reason), excluded_labels.jsonl and manifest.json.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random

from ror_set_point.common.io import read_jsonl, sha256, write_jsonl, write_manifest
from ror_set_point.common.records import Records, read_records
from ror_set_point.common.text import normalize, script


def frequency_bucket(n):
    return "1" if n == 1 else "2-3" if n <= 3 else "4-9" if n <= 9 else "10+"


def labelled_rows(a, ids, bad):
    """The labeled strings in a fixed order: affiliations train by row number, then the openalex_sample train
    rows as they are in the file (sample v1, then v2)."""
    rows = []
    for r in read_jsonl(a.affiliations_train):
        rows.append({"origin": "train", "row": r["id"], "text": r["text"], "labels": r["ror_ids"],
                     "source": r["source"]})
    for r in read_jsonl(a.openalex_sample_train):
        origin = f"dev_{r['sample']}"
        rows.append({"origin": origin, "row": r["id"], "text": r["text"], "labels": r["ror_ids"], "source": origin})
    order = {"train": 0, "dev_v1": 1, "dev_v2": 2}
    rows.sort(key=lambda r: (order[r["origin"]], int(r["row"]) if r["origin"] == "train" else 0))
    counts = collections.Counter()
    for r in rows:
        r["id"] = f"{r['origin']}:{r['row']}"
        r["script"] = script(r["text"])
        r["ror_ids"] = []
        for x in r.pop("labels"):
            y = ids.resolve(x, counts, "labels")
            if y is None:
                bad.append({"id": r["id"], "text": r["text"], "ror_id": x})
            elif y not in r["ror_ids"]:
                r["ror_ids"].append(y)
    return rows, counts


def assign_halves(dev, held, h):
    """Strings that name the same held-out record are one group. In each source x script stratum, the groups go in
    sha256 order to the validation or the test split, whichever has fewer strings at that time."""
    parent = list(range(len(dev)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first = {}
    for i, r in enumerate(dev):
        for x in set(r["ror_ids"]) & held:
            if x in first:
                parent[find(i)] = find(first[x])
            else:
                first[x] = i
    groups = collections.defaultdict(list)
    for i in range(len(dev)):
        groups[find(i)].append(i)
    by_stratum = collections.defaultdict(list)
    for g in groups.values():
        by_stratum[(dev[g[0]]["source"], dev[g[0]]["script"])].append(g)
    for _, gs in sorted(by_stratum.items()):
        size = {"validation": 0, "test": 0}
        for g in sorted(gs, key=lambda g: h(normalize(dev[g[0]]["text"]))):
            half = min(size, key=lambda k: (size[k], k == "validation"))
            size[half] += len(g)
            for i in g:
                dev[i]["half"] = half
    return len(groups)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--affiliations-train", required=True)
    ap.add_argument("--openalex-sample-train", required=True)
    ap.add_argument("--affiliations-validation", required=True, help="read for its texts only")
    ap.add_argument("--teacher", nargs="+", required=True, help="teacher_judgments splits, read for their texts only")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--share", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out)
    h = lambda x: hashlib.sha256(f"{a.seed}:{x}".encode()).hexdigest()

    def take(keys, stratum):
        u = random.Random(h(f"stratum:{stratum}")).random()
        return sorted(keys, key=h)[:math.floor(len(keys) * a.share + u)]

    ids = Records(a.records)
    types = {r["ror_id"]: r["types"] for r in read_records(a.records)}
    bad = []
    rows, counts = labelled_rows(a, ids, bad)

    kinds = ("education", "healthcare", "facility", "government", "company", "nonprofit", "funder", "archive", "other")
    freq, nl = collections.Counter(), collections.Counter()
    for r in rows:
        for x in r["ror_ids"]:
            freq[x] += 1
            nl[x] += r["script"] == "non-Latin"
    rec = {}
    for x, n in freq.items():
        t = next((t for t in kinds if t in types.get(x, ())), "other")
        sc = "non-Latin" if nl[x] * 2 > n else "Latin"
        rec[x] = {"ror": x, "type": t, "script": sc, "freq": n, "stratum": f"{t}|{sc}|{frequency_bucket(n)}"}
    strata = collections.defaultdict(list)
    for x, d in rec.items():
        strata[d["stratum"]].append(x)
    held = set()
    for st, xs in sorted(strata.items()):
        held |= set(take(xs, st))

    nil_strata = collections.defaultdict(list)
    for r in rows:
        if not r["ror_ids"]:
            nil_strata[f"nil|{r['source']}|{r['script']}"].append(r["id"])
    nil_moved = set()
    for st, ks in sorted(nil_strata.items()):
        nil_moved |= set(take(ks, st))
    teacher_norm = {normalize(r["text"]) for p in a.teacher for r in read_jsonl(p)}
    val_norm = {normalize(r["text"]) for r in read_jsonl(a.affiliations_validation)}

    dev, train, excluded, dev_norm = [], [], [], set()
    for r in rows:
        if set(r["ror_ids"]) & held or r["id"] in nil_moved:
            r["moved"] = True
            n = normalize(r["text"])
            reason = ("duplicate of a moved string" if n in dev_norm else
                      "equal to a teacher string" if n in teacher_norm else
                      "equal to an affiliations validation string" if n in val_norm else None)
            if reason:
                excluded.append((r, reason))
            else:
                dev.append(r)
                dev_norm.add(n)
    for r in rows:
        if r.get("moved"):
            continue
        if normalize(r["text"]) in dev_norm:
            excluded.append((r, "equal to a validation or test string"))
        else:
            train.append(r)
    labelled = {x for r in train for x in r["ror_ids"]}
    assert not labelled & held
    groups = assign_halves(dev, held, h)

    keep = ("id", "text", "ror_ids", "origin", "source", "script")
    counts["train"] = write_jsonl(os.path.join(a.out, "train.jsonl"), ({k: r[k] for k in keep} for r in train))
    for half in ("validation", "test"):
        counts[half] = write_jsonl(os.path.join(a.out, f"{half}.jsonl"),
                                   ({k: r[k] for k in keep} for r in dev if r["half"] == half))
    counts["held_out_records"] = write_jsonl(os.path.join(a.out, "held_out_records.jsonl"),
                                             (rec[x] for x in sorted(held)))
    write_jsonl(os.path.join(a.out, "excluded.jsonl"), ({**{k: r[k] for k in keep}, "reason": why} for r, why in excluded))
    write_jsonl(os.path.join(a.out, "excluded_labels.jsonl"), bad)
    counts.update({"records": len(rec), "labelled_strings": len(rows), "groups": groups,
                   "train_records": len(labelled), "excluded": len(excluded)})
    counts.update(collections.Counter(f"excluded: {why}" for _, why in excluded))
    counts.update(collections.Counter(f"held out: {rec[x]['type']}" for x in held))
    write_manifest(a.out, seed=a.seed, share=a.share,
                   inputs={x: sha256(getattr(a, x)) for x in
                           ("affiliations_train", "openalex_sample_train", "affiliations_validation")},
                   counts=dict(sorted(counts.items())))
    if a.verbose:
        print(json.dumps(dict(sorted(counts.items())), indent=1))


if __name__ == "__main__":
    main()
