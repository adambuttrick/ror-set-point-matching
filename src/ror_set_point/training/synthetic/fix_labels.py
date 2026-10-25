"""Apply two label rules of the annotation guidelines to synthetic strings, row for row.

    python -m ror_set_point.training.synthetic.fix_labels --synthetic synthetic.jsonl --records records \
        --out synthetic_fixed.jsonl

  Rule 1  A string whose source record is not healthcare, and which names the Russian Academy of Sciences (РАН as a
          word, or "Российской/Российская академи… наук"), gets the Academy as a label. A string that names the
          Siberian, Ural or Far Eastern branch (СО/УрО/ДВО РАН, or the branch in full) also gets that branch.
          (A named parent organization is labeled.)
  Rule 2  A string whose source record is healthcare keeps only the labels whose record is also healthcare.
          (An organization that is only a part of a hospital's name is not labeled.)
The output has the same rows in the same order, so line numbers do not change.
"""
import argparse
import collections
import json
import re

from ror_set_point.common.io import read_jsonl, write_jsonl
from ror_set_point.common.records import read_records


def fix(row, types, academy, branches, names_academy):
    """The row with its labels fixed, and the rule that changed them (or None)."""
    ids, s = list(row["ror_ids"]), row["text"]
    if "healthcare" in types[row["ror_id"]]:
        keep = [ids[0]] + [x for x in ids[1:] if "healthcare" in types[x]]
        return row | {"ror_ids": keep}, ("rule2" if len(keep) < len(ids) else None)
    if names_academy.search(s):
        new = [x for x in [academy] + [x for rx, x in branches if rx.search(s)] if x not in ids]
        return row | {"ror_ids": ids + new}, ("rule1" if new else None)
    return row, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    word = r"(?<![А-Яа-яЁё]){}(?![А-Яа-яЁё])"
    academy = "https://ror.org/05qrfxd25"
    branches = [(re.compile(word.format("СО РАН") + "|Сибирского отделения"), "https://ror.org/02frkq021"),
                (re.compile(word.format("УрО РАН") + "|Уральского отделения"), "https://ror.org/02s4h3z39"),
                (re.compile(word.format("ДВО РАН") + "|Дальневосточного отделения"), "https://ror.org/05t43vz03")]
    names_academy = re.compile(word.format("РАН") + r"|Российск(?:ой|ая) академи[яи] наук")
    types = {r["ror_id"]: set(r["types"]) for r in read_records(a.records)}
    counts = collections.Counter()

    def rows():
        for r in read_jsonl(a.synthetic):
            out, rule = fix(r, types, academy, branches, names_academy)
            counts[rule or "unchanged"] += 1
            yield out
    write_jsonl(a.out, rows())
    if a.verbose:
        print(json.dumps(dict(sorted(counts.items())), indent=1))


if __name__ == "__main__":
    main()
