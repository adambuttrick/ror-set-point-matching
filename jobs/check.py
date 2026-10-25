"""Compare the outputs of the jobs with the published files.

    python jobs/check.py records RECORDS_DIR REFERENCE_DIR         # stops when they differ
    python jobs/check.py candidates CANDIDATES REFERENCE           # match candidates rows {id, candidates}
    python jobs/check.py answers ANSWERS REFERENCE                 # answer rows {id, ror_ids[, p]}

records compares two records folders: the record rows, withdrawn.json and openalex_to_ror.json. candidates and
answers print how many rows of the reference have the same list (or set) of ROR ids, matched by id.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ror_set_point.common.io import read_jsonl  # noqa: E402
from ror_set_point.common.records import read_records  # noqa: E402


def records(built, reference):
    a = {r["ror_id"]: r for r in read_records(built)}
    b = {r["ror_id"]: r for r in read_records(reference)}
    differ = [k for k in b if k in a and a[k] != b[k]]
    print(f"records: {len(a)} built, {len(b)} reference, {len(a.keys() - b.keys())} only built, "
          f"{len(b.keys() - a.keys())} only reference, {len(differ)} differ")
    ok = a.keys() == b.keys() and not differ
    for name in ("withdrawn.json", "openalex_to_ror.json"):
        with open(os.path.join(built, name), encoding="utf-8") as f, \
                open(os.path.join(reference, name), encoding="utf-8") as g:
            same = json.load(f) == json.load(g)
        print(f"{name}: {'equal' if same else 'differs'}")
        ok &= same
    if not ok:
        raise SystemExit("the records folders differ")


def rows(path, field):
    return {str(r["id"]): r[field] for r in read_jsonl(path)}


def candidates(path, reference):
    a, b = rows(path, "candidates"), rows(reference, "candidates")
    same_list = sum(a.get(k) == v for k, v in b.items())
    same_set = sum(set(a.get(k) or ()) == set(v) for k, v in b.items())
    print(f"candidates: {len(b)} reference rows, {len(a)} rows, {sum(k in a for k in b)} matched by id; "
          f"same list {same_list} ({100 * same_list / len(b):.2f}%), same set {same_set} "
          f"({100 * same_set / len(b):.2f}%)")


def answers(path, reference):
    a, b = rows(path, "ror_ids"), rows(reference, "ror_ids")
    same = sum(set(a.get(k) or ()) == set(v) for k, v in b.items())
    print(f"answers: {len(b)} reference rows, {len(a)} rows, {sum(k in a for k in b)} matched by id; "
          f"same ROR ids {same} ({100 * same / len(b):.2f}%), different {len(b) - same}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"records": records, "candidates": candidates, "answers": answers}[cmd](*args)
