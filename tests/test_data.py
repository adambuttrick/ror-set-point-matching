import json
import sys

from ror_set_point.common.io import read_jsonl, write_jsonl
from ror_set_point.common.records import Records
from ror_set_point.training.data import records as records_step
from ror_set_point.training.data import split as split_step


def ror_record(i, status="active", rel=(), kind="education", country=("JP", "Japan")):
    return {"id": f"https://ror.org/0{i:07d}", "status": status, "types": [kind],
            "names": [{"value": f"Org {i}", "types": ["ror_display", "label"], "lang": None},
                      {"value": f"O{i}", "types": ["acronym"], "lang": None}],
            "locations": [{"geonames_details": {"name": "City", "country_subdivision_name": None,
                                                "country_name": country[1], "country_code": country[0]}}],
            "relationships": [{"type": t, "id": f"https://ror.org/0{j:07d}", "label": f"Org {j}"} for t, j in rel]}


def run(module, args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"] + args)
    module.main()


def make_records(tmp_path, monkeypatch, n=40):
    dump = [ror_record(i) for i in range(1, n + 1)]
    dump[0]["relationships"] = [{"type": "child", "id": ror_record(2)["id"], "label": "Org 2"}]
    dump += [ror_record(901, "withdrawn", [("successor", 902)]), ror_record(902, "withdrawn", [("successor", 1)]),
             ror_record(903, "withdrawn")]
    (tmp_path / "ror.json").write_text(json.dumps(dump))
    write_jsonl(str(tmp_path / "inst.jsonl.gz"), [{"id": "7", "ror": ror_record(1)["id"], "status": "active"}])
    run(records_step, ["--ror", str(tmp_path / "ror.json"), "--openalex-institutions", str(tmp_path / "inst.jsonl.gz"),
                       "--out", str(tmp_path / "records")], monkeypatch)
    return tmp_path / "records"


def test_records_text_and_withdrawn_chain(tmp_path, monkeypatch):
    path = make_records(tmp_path, monkeypatch)
    rows = list(read_jsonl(str(path / "records.jsonl.gz")))
    assert len(rows) == 40
    assert rows[0]["text"] == "Org 1 | O1 | City, Japan | education | active | child: Org 2"
    ids = Records(str(path))
    assert ids.resolve(ror_record(901)["id"]) == ror_record(1)["id"]
    assert ids.resolve(ror_record(903)["id"]) is None
    assert ids.from_openalex(7) == ror_record(1)["id"]


def test_split_keeps_held_out_records_out_of_train(tmp_path, monkeypatch):
    path = make_records(tmp_path, monkeypatch)
    rid = lambda i: ror_record(i)["id"]
    train = [{"id": str(n), "text": f"Lab {n}, Org {1 + n % 40}", "ror_ids": [rid(1 + n % 40)], "source": "openalex"}
             for n in range(120)]
    train += [{"id": str(120 + n), "text": f"Nothing {n}", "ror_ids": [], "source": "openalex"} for n in range(20)]
    train.append({"id": "140", "text": "Old name", "ror_ids": [ror_record(901)["id"]], "source": "crossref"})
    sample = [{"id": f"s{n}", "text": f"Unit {n}, Org {1 + n % 7} and Org {2 + n % 5}",
               "ror_ids": [rid(1 + n % 7), rid(2 + n % 5)], "sample": "v1"} for n in range(30)]
    files = {"train": train, "sample": sample, "val": [{"id": "0", "text": "Lab 0, Org 1", "ror_ids": []}],
             "teacher": [{"id": "t", "text": "Nothing 3", "judgments": []}]}
    for k, rows in files.items():
        write_jsonl(str(tmp_path / f"{k}.jsonl"), rows)
    run(split_step, ["--records", str(path), "--affiliations-train", str(tmp_path / "train.jsonl"),
                     "--openalex-sample-train", str(tmp_path / "sample.jsonl"),
                     "--affiliations-validation", str(tmp_path / "val.jsonl"), "--teacher", str(tmp_path / "teacher.jsonl"),
                     "--seed", "7", "--share", "0.3", "--out", str(tmp_path / "unseen")], monkeypatch)
    out = {s: list(read_jsonl(str(tmp_path / "unseen" / f"{s}.jsonl"))) for s in ("train", "validation", "test")}
    held = {r["ror"] for r in read_jsonl(str(tmp_path / "unseen" / "held_out_records.jsonl"))}
    excluded = list(read_jsonl(str(tmp_path / "unseen" / "excluded.jsonl")))
    assert held
    assert not {x for r in out["train"] for x in r["ror_ids"]} & held
    ids = [r["id"] for s in out.values() for r in s] + [r["id"] for r in excluded]
    assert len(ids) == len(set(ids)) == len(train) + len(sample)
    # "Nothing 3" is equal to a teacher string, so it is never a validation or test string
    assert "train:123" not in {r["id"] for s in ("validation", "test") for r in out[s]}
    old = next(r for s in out.values() for r in s if r["id"] == "train:140")
    assert old["ror_ids"] == [rid(1)]
    half_of = {}
    for s in ("validation", "test"):
        for r in out[s]:
            for x in set(r["ror_ids"]) & held:
                assert half_of.setdefault(x, s) == s
