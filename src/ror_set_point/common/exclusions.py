"""Training strings that the models must not see.

A teacher string is left out when the teacher gives p >= 0.5 to a held-out record, or when it is equal to an
evaluation string. Any other training string is left out when it is equal to an evaluation string. "Equal" means
equal after ror_set_point.common.text.normalize.
"""
from .io import read_jsonl
from .text import normalize


def held_out_records(path):
    return {r["ror"] for r in read_jsonl(path)}


def evaluation_texts(paths):
    return {normalize(r["text"]) for p in paths for r in read_jsonl(p)}


def teacher_exclusion(row, held, eval_norm):
    """The reason to leave a teacher_judgments row out, or None."""
    if any(j["p"] >= 0.5 and j["ror_id"] in held for j in row["judgments"]):
        return "names a held-out record"
    if normalize(row["text"]) in eval_norm:
        return "equal to an evaluation string"
    return None
