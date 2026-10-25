"""Write record groups of the held-out institutions, for ror_set_point.evaluation.score --record-group.

    python -m ror_set_point.evaluation.groups --held-out unseen/held_out_records.jsonl \
        --synthetic synthetic_non_latin/train.jsonl --out groups

Writes into --out (which must not exist), one ROR id on each line:
  held_out.txt                    every held-out record
  held_out_in_synthetic.txt       held-out records that are a label of a synthetic string. A model that trained on all
                                  synthetic strings saw a positive for these records
  held_out_not_in_synthetic.txt   the other held-out records
"""
import argparse
import os

from ror_set_point.common.exclusions import held_out_records
from ror_set_point.common.io import read_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--held-out", required=True, help="held_out_records.jsonl")
    ap.add_argument("--synthetic", required=True, help="the synthetic_non_latin train split")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out)
    held = held_out_records(a.held_out)
    named = {x for r in read_jsonl(a.synthetic) for x in r["ror_ids"]} & held
    for name, ids in (("held_out", held), ("held_out_in_synthetic", named), ("held_out_not_in_synthetic", held - named)):
        with open(os.path.join(a.out, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{x}\n" for x in sorted(ids))


if __name__ == "__main__":
    main()
