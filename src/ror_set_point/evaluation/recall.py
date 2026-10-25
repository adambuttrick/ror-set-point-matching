"""Candidate recall: the share of gold links that a candidate list holds. Prints Markdown.

    python -m ror_set_point.evaluation.recall --gold unseen/test.jsonl --records records --train unseen/train.jsonl \
        --candidates retriever retriever/topk.jsonl.gz --set unseen_test --lexical

--candidates NAME FILE reads ranked lists: rows {id, ids} (ror_set_point.inference.retriever, or the topk.jsonl.gz of a
training run, with --set to choose its rows) or rows {id, candidates} (ror_set_point.inference.match candidates, which
are not ranked). For ranked lists, the table gives recall at each --ks cut. With --lexical, it also gives the
lexical top --lexical-k alone, and the top --union-k of each ranked list with the lexical top --lexical-k added (the
candidates of the matcher). Recall splits by known and new records (with --train) and by script. --record-group NAME
FILE adds a column with the recall of the gold links to a group of records (as ror_set_point.evaluation.score reads it).
"""
import argparse
import collections

from ror_set_point.common.io import read_jsonl
from ror_set_point.common.records import Records
from ror_set_point.common.text import script
from ror_set_point.evaluation.score import read_group


def recall(gold, cands, known, groups=()):
    """A Counter of gold links ("g_" + column) and hits ("h_" + column), and the mean list size."""
    c = collections.Counter()
    size = 0
    for k, (G, sc) in gold.items():
        C = cands.get(k, set())
        size += len(C)
        for x in G:
            cols = ["all", sc] + ([("known" if x in known else "new")] if known is not None else [])
            cols += [name for name, grp in groups if x in grp]
            for col in cols:
                c["g_" + col] += 1
                c["h_" + col] += x in C
    return c, size / max(len(gold), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gold", required=True, help="a labeled split: {id, text, ror_ids}")
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--candidates", nargs=2, action="append", required=True, metavar=("NAME", "FILE"))
    ap.add_argument("--set", help="read only the rows of this set (for a training run's topk.jsonl.gz)")
    ap.add_argument("--train", help="the train split that defines known records")
    ap.add_argument("--ks", default="1,10,32,64")
    ap.add_argument("--lexical", action="store_true", help="add the rows of the lexical index of --records")
    ap.add_argument("--lexical-k", type=int, default=10)
    ap.add_argument("--union-k", type=int, default=32)
    ap.add_argument("--workers", type=int, default=1, help="processes for the lexical index")
    ap.add_argument("--record-group", nargs=2, action="append", default=[], metavar=("NAME", "FILE"))
    a = ap.parse_args()
    ids = Records(a.records)
    groups = [(name, read_group(path)) for name, path in a.record_group]
    known = {x for r in read_jsonl(a.train) for x in r["ror_ids"]} if a.train else None
    gold, texts = {}, {}
    for r in read_jsonl(a.gold):
        k = str(r["id"])
        gold[k] = ({y for y in (ids.resolve(x) for x in r["ror_ids"]) if y}, script(r["text"]))
        texts[k] = r["text"]
    lexical = None
    if a.lexical:
        from ror_set_point.inference.lexical import LexicalIndex
        ix = LexicalIndex(a.records)
        keys = list(texts)
        lexical = dict(zip(keys, ix.top_many([texts[k] for k in keys], a.lexical_k, a.workers)))

    rows = []
    if lexical is not None:
        rows.append((f"lexical top {a.lexical_k}", recall(gold, {k: set(v) for k, v in lexical.items()}, known, groups)))
    for name, path in a.candidates:
        lists = {}
        for d in read_jsonl(path):
            if "set" in d and not a.set:
                raise SystemExit(f"{path} holds several sets; choose one with --set")
            if a.set and d.get("set") != a.set:
                continue
            lists[str(d["id"])] = d["ids"] if "ids" in d else d["candidates"]
        if not any("ids" in d for d in read_jsonl(path, 1)):
            rows.append((name, recall(gold, {k: set(v) for k, v in lists.items()}, known, groups)))
            continue
        for k in (int(x) for x in a.ks.split(",")):
            rows.append((f"{name} top {k}", recall(gold, {q: set(v[:k]) for q, v in lists.items()}, known, groups)))
        if lexical is not None:
            rows.append((f"{name} top {a.union_k} + lexical top {a.lexical_k}",
                         recall(gold, {q: set(v[:a.union_k]) | set(lexical.get(q, [])) for q, v in lists.items()},
                                known, groups)))
    cols = ["all"] + (["known", "new"] if known is not None else []) + ["Latin", "non-Latin"] + [n for n, _ in groups]
    print("| Candidates | Per string | " + " | ".join(f"{c} links" for c in cols) + " |")
    print("|---|---:|" + "---:|" * len(cols))
    for name, (c, size) in rows:
        print(f"| {name} | {size:.1f} | " + " | ".join(
            f"{100 * c['h_' + col] / c['g_' + col]:.1f}" if c["g_" + col] else "–" for col in cols) + " |")
    first = rows[0][1][0]
    print(f"\n{len(gold):,} strings, {first['g_all']:,} gold links"
          + (f" ({first['g_known']:,} known, {first['g_new']:,} new)" if known is not None else "") + ".")


if __name__ == "__main__":
    main()
