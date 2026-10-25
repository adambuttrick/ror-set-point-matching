"""A records folder holds these files:

    records.jsonl.gz      one row per active or inactive ROR record (the format of the ror_records config)
    withdrawn.json        {withdrawn ROR id: successor ROR id or null}
    openalex_to_ror.json  {OpenAlex institution id: ROR id or null}

The training step "records" writes such a folder. The folder data/ror_records of the dataset is also one: its
train.jsonl takes the place of records.jsonl.gz. Records resolves identifiers to the records of the folder's ROR
release.
"""
import json
import os

from .io import read_jsonl


def records_file(path):
    for name in ("records.jsonl.gz", "records.jsonl", "train.jsonl"):
        if os.path.exists(os.path.join(path, name)):
            return os.path.join(path, name)
    raise SystemExit(f"{path}: no records.jsonl.gz or train.jsonl")


def read_records(path):
    return read_jsonl(records_file(path))


class Records:
    def __init__(self, path):
        self.path = path
        self.status = {r["ror_id"]: r["status"] for r in read_records(path)}
        with open(os.path.join(path, "withdrawn.json"), encoding="utf-8") as f:
            self.withdrawn = json.load(f)
        with open(os.path.join(path, "openalex_to_ror.json"), encoding="utf-8") as f:
            self.openalex = json.load(f)

    def resolve(self, ror_id, counts=None, what="ids"):
        """The ROR id of an active or inactive record, or None. A withdrawn id goes to its successor, when it has
        one. counts (a Counter) gets one count for the id's status under what."""
        status = self.status.get(ror_id)
        if status:
            out, key = ror_id, status
        elif ror_id in self.withdrawn:
            out = self.withdrawn[ror_id]
            key = "withdrawn_to_successor" if out else "withdrawn_no_successor"
        else:
            out, key = None, "unknown"
        if counts is not None:
            counts[f"{what}_{key}"] += 1
        return out

    def from_openalex(self, openalex_id, counts=None, what="ids"):
        """The ROR id for an OpenAlex institution id, resolved as resolve() does, or None."""
        ror_id = self.openalex.get(str(openalex_id))
        if not ror_id:
            if counts is not None:
                counts[f"{what}_no_ror"] += 1
            return None
        return self.resolve(ror_id, counts, what)
