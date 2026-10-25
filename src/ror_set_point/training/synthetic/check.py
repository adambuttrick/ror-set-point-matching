"""A blind check of synthetic strings by judges of another model family (or by people). Two modes:

  sample  --per strings from each script group, in sha256 order of --seed, ROR id and text. Writes
          --dir/check/items_<group>.json for the judges (the string, the source record's names, place, types and
          related records, and the names of the labeled records; no form, confidence or batch) and
          --dir/check/key.jsonl (the rest).
  score   reads --dir/check/verdicts_<group>.jsonl ({id, org, form, labels, note}) and prints, for each group and
          weighted by the group's share of strings, the rates of wrong organization (org = "wrong"), unnatural form
          (form = "unnatural") and wrong labels (labels other than "right"), with Wilson 95% intervals.

    python -m ror_set_point.training.synthetic.check sample --dir synthetic --synthetic synthetic/synthetic.jsonl \
        --records records
    python -m ror_set_point.training.synthetic.check score --dir synthetic --synthetic synthetic/synthetic.jsonl \
        --records records

The ids of the synthetic strings must be "<batch>:<card key>:<n>", as ror_set_point.training.synthetic.generate writes
them, so that the batch files give each string's record.
"""
import argparse
import collections
import hashlib
import json
import math
import os

from ror_set_point.common.io import read_jsonl
from ror_set_point.common.records import read_records


def script_groups():
    groups = {"han_kana_hangul": "CN TW HK MO JP KR KP",
              "cyrillic": "RU UA BY KZ KG TJ MN BG RS MK",
              "arabic": "IR AF EG SA AE IQ JO LB SY PS KW QA BH OM YE LY SD DZ MA TN MR",
              "other": "TH LA KH MM GR CY IL GE AM ET BD NP LK"}
    return {g: v.split() for g, v in groups.items()}


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sample(a, rows, group, out):
    os.makedirs(out)
    cards = {}
    for batch in sorted({r["id"].split(":")[0] for r in rows}):
        with open(os.path.join(a.dir, "batches", f"{batch}.json"), encoding="utf-8") as f:
            for c in json.load(f)["cards"]:
                cards[(batch, c["key"])] = c
    h = lambda r: hashlib.sha256(f"{a.seed}|{r['ror_id']}|{r['text']}".encode()).hexdigest()
    by = collections.defaultdict(list)
    for r in sorted(rows, key=h):
        by[group(r)].append(r)
    with open(os.path.join(out, "key.jsonl"), "w", encoding="utf-8") as fk:
        for g in script_groups():
            items = []
            for i, r in enumerate(by[g][:a.per]):
                batch, key, _ = r["id"].split(":")
                c = cards[(batch, key)]
                names = {c["ror"]: c["name"], **{x["id"]: x["name"] for x in c["related"]}}
                iid = f"{g[:3]}{i:03d}"
                card = {k: c[k] for k in ("name", "acronyms", "other_names", "place", "types", "status")}
                card["related"] = [{"relation": x["relation"], "name": x["name"]} for x in c["related"]]
                items.append({"id": iid, "string": r["text"], "card": card, "labels": [names[x] for x in r["ror_ids"]]})
                fk.write(json.dumps({"item": iid, "group": g} | r, ensure_ascii=False) + "\n")
            with open(os.path.join(out, f"items_{g}.json"), "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=1)


def score(rows, group, out):
    share = collections.Counter(group(r) for r in rows)
    key = {r["item"]: r for r in read_jsonl(os.path.join(out, "key.jsonl"))}
    v = {}
    for g in script_groups():
        for r in read_jsonl(os.path.join(out, f"verdicts_{g}.jsonl")):
            v[r["id"]] = r
    missing = sorted(set(key) - set(v))
    if missing:
        raise SystemExit(f"{len(missing)} items have no verdict, for example {missing[:5]}")
    tests = {"wrong organization": lambda x: x["org"] == "wrong", "unnatural form": lambda x: x["form"] == "unnatural",
             "wrong labels": lambda x: x["labels"] != "right"}
    total = sum(share[g] for g in script_groups())
    print("| Group | Strings | Judged | " + " | ".join(tests) + " |")
    print("|---|---:|---:|" + "---|" * len(tests))
    est = collections.defaultdict(lambda: [0.0, 0.0])
    for g in script_groups():
        ids = [i for i in key if key[i]["group"] == g]
        cells = []
        for t, f in tests.items():
            k = sum(f(v[i]) for i in ids)
            lo, hi = wilson(k, len(ids))
            cells.append(f"{k} ({100 * k / len(ids):.1f}%, {max(0.0, 100 * lo):.1f}–{100 * hi:.1f})")
            w, p = share[g] / total, k / len(ids)
            est[t][0] += w * p
            est[t][1] += w * w * p * (1 - p) / len(ids)
        print(f"| {g} | {share[g]:,} | {len(ids)} | " + " | ".join(cells) + " |")
    print(f"| weighted | {total:,} | {len(key)} | " + " | ".join(
        f"{100 * p:.1f}% ({100 * max(0, p - 1.96 * math.sqrt(var)):.1f}–{100 * (p + 1.96 * math.sqrt(var)):.1f})"
        for p, var in est.values()) + " |")
    by_conf = collections.defaultdict(collections.Counter)
    for i, r in key.items():
        by_conf[r["confidence"]]["n"] += 1
        by_conf[r["confidence"]]["wrong"] += tests["wrong organization"](v[i])
    print("\nWrong organization by the generator's confidence:",
          ", ".join(f"{c} {x['wrong']} of {x['n']}" for c, x in sorted(by_conf.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["sample", "score"])
    ap.add_argument("--dir", required=True, help="the working folder of ror_set_point.training.synthetic.generate")
    ap.add_argument("--synthetic", required=True, help="the synthetic strings to check")
    ap.add_argument("--records", required=True, help="a records folder (for the country of each record)")
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--seed", default="20261022")
    a = ap.parse_args()
    rows = list(read_jsonl(a.synthetic))
    group_of = {cc: g for g, ccs in script_groups().items() for cc in ccs}
    cc_of = {r["ror_id"]: (r["places"] or [{}])[0].get("cc") for r in read_records(a.records)}
    group = lambda r: group_of.get(cc_of.get(r["ror_id"]), "other")
    out = os.path.join(a.dir, "check")
    if a.mode == "sample":
        sample(a, rows, group, out)
    else:
        score(rows, group, out)


if __name__ == "__main__":
    main()
