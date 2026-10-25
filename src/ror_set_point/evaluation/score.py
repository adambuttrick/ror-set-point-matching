"""Score answer files against a labeled split: exact match, and micro precision and recall over ROR ids. Prints
Markdown.

    python -m ror_set_point.evaluation.score --gold unseen/test.jsonl --records records \
        --train unseen/train.jsonl --system baseline baseline.jsonl --system "this model" answers.jsonl --paired

Exact match is the share of strings whose answer is exactly the set of gold records. Gold labels are resolved in the
records folder (a withdrawn record goes to its successor; a label with no successor is dropped and counted). Answer
rows are {id, ror_ids}, resolved in the same way (an id with no successor stays, and counts as wrong); rows with
OpenAlex ids ({id, ids}) are resolved too, and an OpenAlex id with no ROR record counts as wrong. A string with no
answer row has an empty answer.

--train (a labeled train split) adds columns for "known" records (some training string names them) and "new"
records (none does). Recall splits by the gold record, precision by the named record; a string is new when one of
its gold records is new. Strings are non-Latin when fewer than half of their letters are Latin.

--exclude-equal-to adds a block without the strings that are equal (after normalization) to a string of those
files. --group-field adds a block for each value of a field of --gold. --paired compares each pair of systems on
the whole split with the exact McNemar test, and gives the minimum detectable effect (alpha 0.05, power 0.8).
--bootstrap gives percentile intervals. --record-group NAME FILE adds a table for a group of records (a file of ROR
ids, one on each line, or JSON Lines rows with "ror" or "ror_id"): the recall of the gold links to these records, and
the exact match of the strings that name one of them.
"""
import argparse
import collections
import json

from ror_set_point.common.io import read_jsonl
from ror_set_point.common.records import Records
from ror_set_point.common.text import normalize, script
from ror_set_point.evaluation.stats import bootstrap, mcnemar_exact, mde_exact


def load_answers(path, ids, sink):
    out = {}
    for d in read_jsonl(path):
        if "ror_ids" in d:
            out[str(d["id"])] = {ids.resolve(x) or x for x in d["ror_ids"]}
        else:
            out[str(d["id"])] = {ids.from_openalex(i, sink, "answers") or f"openalex:{i}" for i in d["ids"]}
    return out


def read_group(path):
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line)
                out.add(d.get("ror_id") or d["ror"])
            elif line:
                out.add(line)
    return out


def pct(x, n):
    return f"{100 * x / n:.1f}" if n else "–"


def p_value(p):
    return f"{p:.2g}" if p > 0 else "<1e-300"


def count(rows, ans, known, watch):
    c = collections.Counter()
    for k, (_, G, sc) in rows.items():
        P = ans.get(k, set())
        c["ex"] += P == G
        c["n_" + sc] += 1
        c["ex_" + sc] += P == G
        c["tp"] += len(P & G)
        c["named"] += len(P)
        c["gold"] += len(G)
        c["nil"] += not G
        c["nil_answered"] += not G and bool(P)
        if known is not None:
            grp = "nil" if not G else "new" if G - known else "known"
            c["n_" + grp] += 1
            c["ex_" + grp] += P == G
        for x in G:
            c["g_" + sc] += 1
            c["h_" + sc] += x in P
            if known is not None:
                kn = "known" if x in known else "new"
                c["g_" + kn] += 1
                c["h_" + kn] += x in P
            if x == watch:
                c["g_watch"] += 1
                c["h_watch"] += x in P
        if known is not None:
            for x in P:
                kn = "known" if x in known else "new"
                c["p_" + kn] += 1
                c["t_" + kn] += x in G
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gold", required=True, help="a labeled split: {id, text, ror_ids}")
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--system", nargs=2, action="append", required=True, metavar=("NAME", "ANSWERS"))
    ap.add_argument("--train", help="the train split that defines known records")
    ap.add_argument("--exclude-equal-to", nargs="+", default=[], help="training files (fields text, s or q)")
    ap.add_argument("--group-field", help="also score each value of this field")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=0, help="resamples for intervals (0: none)")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--watch", help="a ROR id whose recall is reported apart")
    ap.add_argument("--record-group", nargs=2, action="append", default=[], metavar=("NAME", "FILE"))
    a = ap.parse_args()
    ids, sink = Records(a.records), collections.Counter()
    known = {x for r in read_jsonl(a.train) for x in r["ror_ids"]} if a.train else None
    rows, group = {}, {}
    for r in read_jsonl(a.gold):
        k = str(r["id"])
        G = {y for y in (ids.resolve(x, sink, "gold") for x in r["ror_ids"]) if y}
        rows[k] = (r["text"], G, script(r["text"]))
        if a.group_field:
            group[k] = r.get(a.group_field)
    A = {name: load_answers(p, ids, sink) for name, p in a.system}

    blocks = [("all", list(rows))]
    if a.exclude_equal_to:
        seen = set()
        for p in a.exclude_equal_to:
            for d in read_jsonl(p):
                seen.add(normalize(d.get("text") or d.get("s") or d.get("q") or ""))
        blocks.append(("not equal to a training string", [k for k, (s, _, _) in rows.items() if normalize(s) not in seen]))
    for v in sorted({x for x in group.values() if x is not None}):
        blocks.append((f"{a.group_field} = {v}", [k for k in rows if group[k] == v]))

    head = ["Strings", "System", "Exact", "P", "R"]
    if known is not None:
        head += ["R known", "R new", "P known", "P new", "Exact known / new / NIL"]
    head += ["Exact Latin / non-Latin", "R Latin / non-Latin", "NIL answered"]
    print("| Block | " + " | ".join(head) + " |")
    print("|---|---:|---|" + "---:|" * 3 + ("---:|" * 4 + "---|" if known is not None else "") + "---|---|---:|")
    notes = []
    for title, keys in blocks:
        sub = {k: rows[k] for k in keys}
        for j, (name, ans) in enumerate(A.items()):
            c = count(sub, ans, known, a.watch)
            cells = [title if j == 0 else "", f"{len(keys):,}" if j == 0 else "", name, pct(c["ex"], len(keys)),
                     pct(c["tp"], c["named"]), pct(c["tp"], c["gold"])]
            if known is not None:
                cells += [pct(c["h_known"], c["g_known"]), pct(c["h_new"], c["g_new"]),
                          pct(c["t_known"], c["p_known"]), pct(c["t_new"], c["p_new"]),
                          f"{pct(c['ex_known'], c['n_known'])} / {pct(c['ex_new'], c['n_new'])} / "
                          f"{pct(c['ex_nil'], c['n_nil'])}"]
            cells += [f"{pct(c['ex_Latin'], c['n_Latin'])} / {pct(c['ex_non-Latin'], c['n_non-Latin'])}",
                      f"{pct(c['h_Latin'], c['g_Latin'])} / {pct(c['h_non-Latin'], c['g_non-Latin'])}",
                      pct(c["nil_answered"], c["nil"])]
            print("| " + " | ".join(cells) + " |")
            if j == 0:
                notes.append(f"- {title}: {len(keys):,} strings ({c['nil']:,} name no record, {c['n_non-Latin']:,} "
                             f"non-Latin), {c['gold']:,} gold links"
                             + (f" ({c['g_known']:,} known, {c['g_new']:,} new)" if known is not None else "") + ".")
            if c["g_watch"]:
                notes.append(f"  - {name}: recall of {a.watch} {c['h_watch']:,} of {c['g_watch']:,}.")
    print("\n" + "\n".join(notes))

    if a.record_group:
        print("\nRecord groups: recall of the gold links to the records of the group, and exact match of the strings "
              "that name one of them.\n")
        print("| Group | Links | Strings | " + " | ".join(f"{n} recall | {n} exact" for n in A) + " |")
        print("|---|---:|---:|" + "---:|---:|" * len(A))
        for name, path in a.record_group:
            grp = read_group(path)
            links = [(k, x) for k, (_, G, _) in rows.items() for x in G if x in grp]
            strings = [k for k, (_, G, _) in rows.items() if G & grp]
            cells = []
            for ans in A.values():
                cells.append(pct(sum(x in ans.get(k, set()) for k, x in links), len(links)))
                cells.append(pct(sum(ans.get(k, set()) == rows[k][1] for k in strings), len(strings)))
            print(f"| {name} | {len(links):,} | {len(strings):,} | " + " | ".join(cells) + " |")
    if a.paired and len(A) > 1:
        print(f"\nPaired exact match on all {len(rows):,} strings. psi is the share of strings where exactly one "
              "system is right. MDE is the smallest difference that the exact McNemar test detects (alpha 0.05, "
              "power 0.8).\n")
        print("| A | B | A only | B only | psi | MDE | p |")
        print("|---|---|---:|---:|---:|---:|---:|")
        names = list(A)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                ax = sum(A[x].get(k, set()) == G and A[y].get(k, set()) != G for k, (_, G, _) in rows.items())
                by = sum(A[y].get(k, set()) == G and A[x].get(k, set()) != G for k, (_, G, _) in rows.items())
                psi = (ax + by) / len(rows)
                print(f"| {x} | {y} | {ax:,} | {by:,} | {100 * psi:.1f}% | {100 * mde_exact(len(rows), psi):.1f} | "
                      f"{p_value(mcnemar_exact(ax, by))} |")
    if a.bootstrap:
        print(f"\n{a.bootstrap:,} bootstrap resamples of strings (seed {a.seed}), 95% percentile intervals:\n")
        print("| System | Exact | P | R |")
        print("|---|---|---|---|")
        for name, ans in A.items():
            per = []
            for k, (_, G, _) in rows.items():
                P = ans.get(k, set())
                per.append((int(P == G), len(P & G), len(P), len(G)))
            iv = bootstrap(per, a.bootstrap, a.seed)
            print(f"| {name} | " + " | ".join(f"{100 * iv[m][0]:.1f}–{100 * iv[m][1]:.1f}"
                                              for m in ("exact", "precision", "recall")) + " |")
    changed = {k: v for k, v in sorted(sink.items()) if not k.endswith(("_active", "_inactive"))}
    if changed:
        print(f"\nIds that were not active or inactive records: {changed}")


if __name__ == "__main__":
    main()
