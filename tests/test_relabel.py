import http.server
import json
import random
import sys
import threading
from collections import Counter

import pytest

from ror_set_point.common.io import read_jsonl, write_jsonl
from ror_set_point.training.data import relabel
from ror_set_point.training.data.relabel import card, platt, request_body, sigmoid


def rid(i):
    return f"https://ror.org/0{i:07d}"


def record(i, name, labels=(), aliases=(), acronyms=(), types=("education",), places=(("Kyoto", "Japan"),)):
    return {"ror_id": rid(i), "status": "active", "name": name, "acronyms": list(acronyms),
            "labels": [{"v": v, "lang": None} for v in labels], "aliases": list(aliases),
            "places": [{"city": c, "region": None, "country": k, "cc": None} for c, k in places], "types": list(types),
            "text": name}


RECORDS = [record(1, "Org 1", acronyms=["O1"]), record(2, "Org 2", ["Organisation 2"]), record(3, "Org 3")]
TEACHER = {"train": [{"id": f"t{n}", "text": f"Lab {n}, Org {n % 3 + 1}",
                      "judgments": [{"ror_id": rid(n % 3 + 1), "p": [0.98, 1.0, 0.5][n % 3]},
                                    {"ror_id": rid((n + 1) % 3 + 1), "p": [0.04, 0.0, 0.47][n % 3]}]}
                     for n in range(7)],
           "validation": [{"id": "v0", "text": "Org 2", "judgments": [{"ror_id": rid(2), "p": 0.1234}]}]}


def run(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["relabel"] + args)
    relabel.main()


def folders(tmp_path):
    """A records folder, a teacher_judgments folder and a question file."""
    (tmp_path / "records").mkdir()
    write_jsonl(str(tmp_path / "records" / "records.jsonl.gz"), RECORDS)
    (tmp_path / "teacher").mkdir()
    for split, rows in TEACHER.items():
        write_jsonl(str(tmp_path / "teacher" / f"{split}.jsonl"), rows)
    (tmp_path / "q.txt").write_text("Does the string name it?\n")
    return [str(tmp_path / x) for x in ("records", "teacher", "q.txt")]


def test_card_has_the_fields_of_the_teacher_card():
    r = record(1, "Kyoto University", ["京都大学", "Kyoto University"], ["Kyodai", "京都大学", "a", "b", "c", "d", "e"],
               ["KU", "KYU", "KU", "A", "B", "C"], ("education", "funder"), (("Kyoto", "Japan"), ("Tokyo", "Japan")))
    assert card(r) == {"name": "Kyoto University", "other_names": ["京都大学", "Kyodai", "a", "b", "c", "d"],
                       "acronyms": ["A", "B", "C", "KU"], "type": "education", "city": "Kyoto", "country": "Japan"}
    assert card(record(2, "Lab", types=(), places=())) == {"name": "Lab", "other_names": [], "acronyms": [],
                                                           "type": None, "city": None, "country": None}


def test_shards_hold_whole_rows_and_each_pair_once(tmp_path, monkeypatch):
    records, teacher, question = folders(tmp_path)
    rows = TEACHER["train"] + TEACHER["validation"]
    lines = []
    for i in range(3):
        out = str(tmp_path / f"requests{i}.jsonl.gz")
        run(["requests", "--teacher", teacher, "--records", records, "--question", question, "--shard", str(i),
             "--shards", "3", "--out", out], monkeypatch)
        part = list(read_jsonl(out))
        assert {x["id"] for x in part} == {r["id"] for n, r in enumerate(rows) if n % 3 == i}
        lines += part
    want = [(s, r["id"], j["ror_id"], j["p"]) for s in ("train", "validation") for r in TEACHER[s]
            for j in r["judgments"]]
    assert sorted((x["split"], x["id"], x["ror_id"], x["p_jev"]) for x in lines) == sorted(want)
    assert len({x["rid"] for x in lines}) == len(want)
    v0 = next(x for x in lines if x["id"] == "v0")
    assert v0["rid"] == f"validation|v0|{rid(2)}"
    assert v0["body"] == {"model": "jev-1.13.0",
                          "state": {"affiliation_string": "Org 2", "institution": card(RECORDS[1])},
                          "questions": {"names_this_institution": {"type": "noul",
                                                                   "instructions": "Does the string name it?"}}}


def test_stage2_requests_keep_labels_and_folds(tmp_path, monkeypatch):
    records, _, question = folders(tmp_path)
    write_jsonl(str(tmp_path / "stage2.jsonl.gz"),
                [{"k": "train:1", "s": "Org 1", "c": [rid(1), rid(2)], "y": [1.0, 0.0], "fold": 3}])
    args = ["requests", "--stage2", str(tmp_path / "stage2.jsonl.gz"), "--records", records, "--question", question]
    run(args + ["--out", str(tmp_path / "r.jsonl.gz")], monkeypatch)
    assert [(x["rid"], x["y"], x["fold"]) for x in read_jsonl(str(tmp_path / "r.jsonl.gz"))] == [
        (f"stage2|train:1|{rid(1)}", 1.0, 3), (f"stage2|train:1|{rid(2)}", 0.0, 3)]
    write_jsonl(str(tmp_path / "stage2.jsonl.gz"), [{"k": "a", "s": "Org 9", "c": [rid(9)], "y": [0.0], "fold": 0}])
    with pytest.raises(SystemExit, match="not in the records"):
        run(args + ["--out", str(tmp_path / "r2.jsonl.gz")], monkeypatch)


class Teacher(http.server.BaseHTTPRequestHandler):
    """A /v1/systemone server. p is the length of the string / 100. The first request for "busy" gets HTTP 503. The
    strings in refuse get HTTP 400."""
    protocol_version = "HTTP/1.1"
    seen = Counter()
    refuse = set()

    def do_POST(self):
        s = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["state"]["affiliation_string"]
        Teacher.seen[s] += 1
        if s in Teacher.refuse or (s == "busy" and Teacher.seen[s] == 1):
            code, out = (400 if s in Teacher.refuse else 503), {"error": "no"}
        else:
            code, out = 200, {"answers": {"names_this_institution": {"noul": len(s) / 100}},
                              "usage": {"input_tokens": 500}}
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    Teacher.seen, Teacher.refuse = Counter(), {"bad"}
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Teacher)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}/v1/systemone"
    s.shutdown()
    s.server_close()


def test_score_tries_again_continues_and_counts_the_pairs(tmp_path, monkeypatch, server):
    monkeypatch.setattr(relabel, "RETRY_WAIT", 0)
    texts = ["Org 1", "busy", "Lab 22, Org 2", "bad"]
    write_jsonl(str(tmp_path / "requests.jsonl.gz"), [{"rid": f"train|{n}|x", "p_jev": 0.9,
                                                       "body": request_body(s, {"name": "Org"}, "Q?")}
                                                      for n, s in enumerate(texts)])
    out = tmp_path / "scores.jsonl"
    # a pair with a p, then a line that a stop cut
    out.write_text(json.dumps({"rid": "train|0|x", "p_jev": 0.9, "p": 0.42, "tokens": 9}) + '\n{"rid": "train|2|x", "p')
    args = ["score", "--requests", str(tmp_path / "requests.jsonl.gz"), "--out", str(out), "--url", server,
            "--concurrency", "2", "--summary", str(tmp_path / "summary.json")]
    with pytest.raises(SystemExit, match="1 of 4 pairs have no p"):
        run(args, monkeypatch)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    by = {r["rid"]: r for r in rows}
    assert len(rows) == 4 and by["train|0|x"]["p"] == 0.42 and "Org 1" not in Teacher.seen
    assert by["train|1|x"]["p"] == 0.04 and Teacher.seen["busy"] == 2  # HTTP 503, then an answer
    assert by["train|2|x"] == {"rid": "train|2|x", "p_jev": 0.9, "p": 0.13, "tokens": 500}
    assert by["train|3|x"]["err"].startswith("HTTP 400") and Teacher.seen["bad"] == 1
    assert json.loads((tmp_path / "summary.json").read_text())["pairs_without_p"] == 1
    Teacher.refuse = set()
    run(args, monkeypatch)  # only the pair without a p is sent again
    assert Teacher.seen == {"busy": 2, "Lab 22, Org 2": 1, "bad": 2}
    assert {r["rid"] for r in relabel.scored([str(out)])} == {f"train|{n}|x" for n in range(4)}


def test_platt_finds_the_scale_and_shift_of_a_logistic_model():
    rng = random.Random(7)
    xs = [rng.uniform(-6, 6) for _ in range(20000)]
    ys = [int(rng.random() < sigmoid(2.0 * x - 1.0)) for x in xs]
    a, b = platt(xs, ys)
    assert a == pytest.approx(2.0, abs=0.1) and b == pytest.approx(-1.0, abs=0.1)
    loss = lambda a, b: relabel.log_loss([sigmoid(a * x + b) for x in xs], ys)
    assert loss(a, b) <= loss(2.0, -1.0) and loss(a, b) < loss(a + 0.01, b) and loss(a, b) < loss(a, b - 0.01)


def test_calibrate_fits_on_all_folds_and_checks_on_the_folds_left_out(tmp_path, monkeypatch):
    rng = random.Random(3)
    rows = []
    for n in range(5000):
        x = rng.uniform(-6, 6)
        rows.append({"rid": f"stage2|{n}|x", "y": float(rng.random() < sigmoid(2.0 * x - 1.0)), "fold": n % 5,
                     "p": sigmoid(x)})
    error = {"rid": "stage2|e|x", "y": 0.0, "fold": 0, "err": "HTTP 500"}
    write_jsonl(str(tmp_path / "a.jsonl"), rows[:2500] + [error])
    write_jsonl(str(tmp_path / "b.jsonl"), rows[2500:])
    run(["calibrate", "--scores", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), "--out",
         str(tmp_path / "cal.json")], monkeypatch)
    cal = json.loads((tmp_path / "cal.json").read_text())
    assert (cal["pairs"], cal["check"]["pairs_fit"], cal["check"]["pairs_test"]) == (5000, 3000, 2000)
    assert cal["a"] == pytest.approx(2.0, abs=0.2) and cal["b"] == pytest.approx(-1.0, abs=0.2)
    assert cal["check"]["log_loss_calibrated"] < cal["check"]["log_loss_raw"]
    write_jsonl(str(tmp_path / "c.jsonl"), [{"rid": "train|t0|x", "p": 0.5}])
    with pytest.raises(SystemExit, match="no label"):
        run(["calibrate", "--scores", str(tmp_path / "c.jsonl"), "--out", str(tmp_path / "c.json")], monkeypatch)


def pair_scores(p=lambda j: j["p"]):
    return [{"rid": f"{s}|{r['id']}|{j['ror_id']}", "p": p(j)} for s in ("train", "validation")
            for r in TEACHER[s] for j in r["judgments"]]


def test_assemble_with_the_identity_calibration_gives_the_input_config(tmp_path, monkeypatch):
    _, teacher, _ = folders(tmp_path)
    write_jsonl(str(tmp_path / "scores.jsonl"), pair_scores() + [{"rid": "train|t0|x", "err": "HTTP 500"}])
    (tmp_path / "cal.json").write_text(json.dumps({"a": 1.0, "b": 0.0}))
    run(["assemble", "--scores", str(tmp_path / "scores.jsonl"), "--teacher", teacher, "--calibration",
         str(tmp_path / "cal.json"), "--out", str(tmp_path / "new")], monkeypatch)
    for split in ("train", "validation"):
        new, old = (tmp_path / x / f"{split}.jsonl" for x in ("new", "teacher"))
        assert new.read_bytes() == old.read_bytes()
    raw = list(read_jsonl(str(tmp_path / "new" / "raw.jsonl.gz")))
    assert len(raw) == 15
    assert raw[0] == {"split": "train", "id": "t0", "ror_id": rid(1), "p_raw": 0.98, "p_jev": 0.98}
    manifest = json.loads((tmp_path / "new" / "manifest.json").read_text())
    assert manifest["counts"]["pairs_train_p_ge_0.5"] == 7


def test_assemble_calibrates_each_pair_and_needs_a_p_for_each(tmp_path, monkeypatch):
    _, teacher, _ = folders(tmp_path)
    (tmp_path / "cal.json").write_text(json.dumps({"a": 2.0, "b": -1.0}))
    write_jsonl(str(tmp_path / "scores.jsonl"), pair_scores(lambda j: 0.5))
    args = ["assemble", "--scores", str(tmp_path / "scores.jsonl"), "--teacher", teacher, "--calibration",
            str(tmp_path / "cal.json")]
    run(args + ["--out", str(tmp_path / "new")], monkeypatch)
    ps = {j["p"] for r in read_jsonl(str(tmp_path / "new" / "train.jsonl")) for j in r["judgments"]}
    assert ps == {round(sigmoid(-1.0), 4)} == {0.2689}
    write_jsonl(str(tmp_path / "scores.jsonl"), pair_scores()[1:])
    with pytest.raises(SystemExit, match="no p for train"):
        run(args + ["--out", str(tmp_path / "new2")], monkeypatch)
