"""Where a system errs: recall by class of gold link, and wrong links by class. Prints Markdown.

    python -m ror_set_point.evaluation.errors --gold unseen/validation.jsonl --records records \
        --train unseen/train.jsonl --system baseline baseline.jsonl --system "this model" answers.jsonl candidates.jsonl

Each gold link gets these classes:
  string     the string names one record, or several
  script     Latin or non-Latin
  record     known (a training string names it) or new; needs --train
  type       the ROR type of the record (healthcare first)
  named by   a name, label or alias of the record occurs in the normalized string; else one of its acronyms occurs
             as a token; else neither (a translation, a short form or another form)
A third value after the answers file of a system gives its candidates ({id, candidates}, or {k, c} with --set). Its
misses then split into "not a candidate" and "a candidate below the threshold". Each wrong named link is on a string
that names no record, on a parent or child of a gold record, on another related record, or on an unrelated record.
--watch puts the strings that name one record into a block of their own.
"""
import argparse
import collections
import re

from ror_set_point.common.io import read_jsonl
from ror_set_point.common.records import Records, read_records
from ror_set_point.common.text import normalize, script


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gold", required=True, help="a labeled split: {id, text, ror_ids}")
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--system", nargs="+", action="append", required=True, metavar="NAME ANSWERS [CANDIDATES]")
    ap.add_argument("--train", help="the train split that defines known records")
    ap.add_argument("--set", help="read only the candidate rows of this set")
    ap.add_argument("--watch", help="a ROR id: the strings that name it make a block of their own")
    a = ap.parse_args()
    kinds = ["healthcare", "education", "facility", "government", "company", "nonprofit", "funder", "archive", "other"]
    ids = Records(a.records)
    card = {}
    for r in read_records(a.records):
        names = [r["name"]] + [x["v"] for x in r["labels"]] + r["aliases"]
        card[r["ror_id"]] = {"names": [n for n in (normalize(x) for x in names) if len(n) >= 3],
                             "acr": [x for x in r["acronyms"] if len(x) >= 2],
                             "type": next((t for t in kinds if t in r["types"]), "other"),
                             "rel": {x["id"]: k for k, xs in r["relations"].items() for x in xs}}
    known = {x for r in read_jsonl(a.train) for x in r["ror_ids"]} if a.train else None
    blocks = collections.defaultdict(dict)
    for r in read_jsonl(a.gold):
        G = {y for y in (ids.resolve(x) for x in r["ror_ids"]) if y}
        title = ("all strings" if not a.watch else f"strings that name {a.watch}" if a.watch in G
                 else f"strings that do not name {a.watch}")
        blocks[title][str(r["id"])] = (r["text"], G)

    def answers(path):
        return {str(d["id"]): set(d["ror_ids"]) for d in read_jsonl(path)}

    def candidates(path):
        out = {}
        for d in read_jsonl(path):
            if a.set and d.get("set") != a.set:
                continue
            out[str(d.get("id", d.get("k")))] = set(d.get("candidates", d.get("c", [])))
        return out
    systems = [(x[0], answers(x[1]), candidates(x[2]) if len(x) > 2 else None) for x in a.system]

    def named_by(s, g):
        c = card.get(g)
        if not c:
            return "no record"
        ns = f" {normalize(s)} "
        if any(f" {n} " in ns for n in c["names"]):
            return "name"
        if any(re.search(rf"(?<![\w]){re.escape(x)}(?![\w])", s) for x in c["acr"]):
            return "acronym only"
        return "neither"

    names = [n for n, *_ in systems]
    for title, rows in sorted(blocks.items()):
        print(f"\n## {title}: {len(rows):,} strings, {sum(len(g) for _, g in rows.values()):,} gold links\n")
        tot, rec = collections.Counter(), {n: collections.Counter() for n in names}
        wrong = {n: collections.Counter() for n in names}
        for k, (s, G) in rows.items():
            for g in G:
                cls = [f"string: {'one record' if len(G) == 1 else 'several records'}", f"script: {script(s)}",
                       f"type: {card[g]['type'] if g in card else 'no record'}", f"named by: {named_by(s, g)}"]
                if known is not None:
                    cls.append(f"record: {'known' if g in known else 'new'}")
                tot.update(cls)
                for name, ans, cands in systems:
                    if g in ans.get(k, set()):
                        rec[name].update(cls)
                    elif cands is not None:
                        rec[name]["miss: not a candidate" if g not in cands.get(k, set())
                                  else "miss: a candidate below the threshold"] += 1
            for name, ans, _ in systems:
                for x in ans.get(k, set()) - G:
                    rel = {card[g]["rel"].get(x) for g in G if g in card} | {card[x]["rel"].get(g) for g in G if x in card}
                    rel.discard(None)
                    wrong[name]["on a string that names no record" if not G else
                                "parent or child of a gold record" if rel & {"parent", "child"} else
                                "other relation of a gold record" if rel else "unrelated"] += 1
        print("| Gold links | n | " + " | ".join(f"{n} recall" for n in names) + " | "
              + " | ".join(f"{n} misses" for n in names) + " |")
        print("|---|---:|" + "---:|" * (2 * len(names)))
        for c in sorted(tot, key=lambda c: (c.split(":")[0], -tot[c])):
            print(f"| {c} | {tot[c]:,} | " + " | ".join(f"{100 * rec[n][c] / tot[c]:.1f}" for n in names) + " | "
                  + " | ".join(f"{tot[c] - rec[n][c]:,}" for n in names) + " |")
        for name, _, cands in systems:
            if cands is not None:
                print(f"\n{name} misses: not a candidate {rec[name]['miss: not a candidate']:,}; a candidate below "
                      f"the threshold {rec[name]['miss: a candidate below the threshold']:,}.")
        print("\n| Wrong named links | " + " | ".join(names) + " |")
        print("|---|" + "---:|" * len(names))
        for c in ("on a string that names no record", "parent or child of a gold record",
                  "other relation of a gold record", "unrelated"):
            print(f"| {c} | " + " | ".join(f"{wrong[n][c]:,}" for n in names) + " |")
        print("| all | " + " | ".join(f"{sum(wrong[n].values()):,}" for n in names) + " |")


if __name__ == "__main__":
    main()
