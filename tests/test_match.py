import contextlib
import json
import sys
import zlib

import pytest

from ror_set_point.inference import match, set_model
from ror_set_point.inference.match import Matcher, chunks, complete_rows, order, unique


def h(*parts):
    return zlib.crc32("|".join(parts).encode())


class FakeLexical:
    def __init__(self):
        self.pools, self.used = 0, []

    def pool(self, workers):
        self.pools += 1
        return contextlib.nullcontext(f"pool {self.pools}")

    def top_many(self, texts, k, workers=1, pool=None):
        self.used.append(pool)
        return [[f"r{h('lex', s, str(i)) % 40}" for i in range(k)] for s in texts]


class FakeRetriever:
    def search(self, texts, k):
        for s in texts:
            yield [f"r{h('ret', s, str(i)) % 40}" for i in range(k)], [0.5] * k, None


class FakeSetModel:
    """Each string-candidate pair gets a fixed p; seen keeps the strings of each call."""

    def __init__(self):
        self.seen = []

    @staticmethod
    def p(text, cand):
        return round((h(text, cand) % 1000) / 1000, 4)

    def predict(self, items, batch_size=32, batch_tokens=65536):
        self.seen.append([t for t, _ in items])
        return [[(c, self.p(t, c)) for c in cands if self.p(t, c) >= 0.5] for t, cands in items]


class FakeMatcher(Matcher):
    def __init__(self):
        self.k, self.lexical_k, self.cap, self.seed = 5, 4, 7, 1363
        self.lexical, self.retriever, self.set_model = FakeLexical(), FakeRetriever(), FakeSetModel()
        self.calls = []

    def match(self, texts, *args):
        self.calls.append(list(texts))
        return super().match(texts, *args)


def expected(text, m):
    """The answer of one string, done alone."""
    ids = next(m.retriever.search([text], m.k))[0]
    lex = m.lexical.top_many([text], m.lexical_k)[0]
    cands = order(list(dict.fromkeys(ids + lex))[:m.cap], text, m.seed)
    return [[c, m.set_model.p(text, c)] for c in cands if m.set_model.p(text, c) >= 0.5]


TEXTS = [f"Dept. {i % 11}, University {i % 4}, Country {i % 3}" for i in range(90)]  # 44 distinct


def write_input(tmp_path, kind="jsonl"):
    if kind == "txt":
        path = tmp_path / "in.txt"
        path.write_text("".join(s + "\n" for s in TEXTS), encoding="utf-8")
        return str(path), [str(n) for n in range(len(TEXTS))]
    path = tmp_path / "in.jsonl"
    keys = [n * 3 if n % 2 else f"k{n}" for n in range(len(TEXTS))]
    path.write_text("".join(json.dumps({"id": k, "text": s}) + "\n" for k, s in zip(keys, TEXTS)), encoding="utf-8")
    return str(path), keys


def main(monkeypatch, m, mode, inp, out, *args):
    monkeypatch.setattr(match, "Matcher", lambda *a, **kw: m)
    monkeypatch.setattr(set_model.SetMatcher, "from_pretrained", lambda *a, **kw: m.set_model)
    monkeypatch.setattr(match, "read_records", lambda path: [])
    monkeypatch.setattr(sys, "argv", ["match", mode, "--input", inp, "--out", str(out), "--records", "-",
                                      "--retriever", "-", "--set-model", "-", *args])
    match.main()
    return out.read_bytes()


def test_unique_and_chunks():
    assert unique(["b", "a", "b", "c", "a"]) == (["b", "a", "c"], [0, 1, 0, 2, 1])
    assert list(chunks(range(10), 4)) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert list(chunks(range(5, 10), 4, done=5)) == [[5, 6, 7], [8, 9]]  # cut where a run from row 0 cuts


def test_chunked_output_is_unchunked_output(tmp_path, monkeypatch):
    inp, keys = write_input(tmp_path)
    one = main(monkeypatch, FakeMatcher(), "run", inp, tmp_path / "one.jsonl", "--chunk-size", "1000")
    m = FakeMatcher()
    seven = main(monkeypatch, m, "run", inp, tmp_path / "seven.jsonl", "--chunk-size", "7")
    assert one == seven
    assert [len(c) for c in m.calls] == [7] * 12 + [6]
    rows = [json.loads(x) for x in one.decode().splitlines()]
    assert [r["id"] for r in rows] == keys
    assert any(r["ror_ids"] for r in rows) and not all(r["ror_ids"] for r in rows)


def test_duplicates_are_done_once_and_map_back_to_their_ids(tmp_path, monkeypatch):
    inp, keys = write_input(tmp_path)
    m = FakeMatcher()
    out = main(monkeypatch, m, "run", inp, tmp_path / "out.jsonl", "--chunk-size", "1000")
    assert m.set_model.seen == [list(dict.fromkeys(TEXTS))]
    for k, s, line in zip(keys, TEXTS, out.decode().splitlines()):
        r = json.loads(line)
        assert r["id"] == k
        assert [list(x) for x in zip(r["ror_ids"], r["p"])] == expected(s, FakeMatcher())


def test_candidates_then_answer_is_run(tmp_path, monkeypatch):
    inp, _ = write_input(tmp_path)
    run = main(monkeypatch, FakeMatcher(), "run", inp, tmp_path / "run.jsonl")
    cand = tmp_path / "cand.jsonl"
    main(monkeypatch, FakeMatcher(), "candidates", inp, cand, "--chunk-size", "9")
    m = FakeMatcher()
    assert main(monkeypatch, m, "answer", str(cand), tmp_path / "ans.jsonl", "--chunk-size", "10") == run
    assert [len(s) for s in m.set_model.seen] == [len(dict.fromkeys(TEXTS[i:i + 10])) for i in range(0, 90, 10)]


def test_lexical_pool_is_made_once(tmp_path, monkeypatch):
    inp, _ = write_input(tmp_path)
    m = FakeMatcher()
    main(monkeypatch, m, "run", inp, tmp_path / "out.jsonl", "--chunk-size", "20", "--workers", "4")
    assert m.lexical.pools == 1 and m.lexical.used == ["pool 1"] * 5


@pytest.mark.parametrize("kind", ["jsonl", "txt"])
def test_resume_after_a_cut_gives_the_same_file(tmp_path, monkeypatch, kind):
    inp, _ = write_input(tmp_path, kind)
    full = main(monkeypatch, FakeMatcher(), "run", inp, tmp_path / "full.jsonl", "--chunk-size", "7")
    lines = full.splitlines(keepends=True)
    out = tmp_path / "out.jsonl"
    for cut in [0, 5, len(lines[0]), sum(map(len, lines[:30])) + 9, len(full) - 1, len(full)]:
        out.write_bytes(full[:cut])
        with pytest.raises(SystemExit):
            main(monkeypatch, FakeMatcher(), "run", inp, out)
        m = FakeMatcher()
        assert main(monkeypatch, m, "run", inp, out, "--chunk-size", "7", "--resume") == full
        done = full[:cut].count(b"\n")
        assert [len(c) for c in m.calls] == [len(c) for c in chunks(range(done, len(TEXTS)), 7, done)]


def test_complete_rows_cuts_a_partial_line(tmp_path):
    p = tmp_path / "out.jsonl"
    assert complete_rows(str(p)) == 0
    p.write_bytes(b'{"a": 1}\n{"a": 2}\n{"a"')
    assert complete_rows(str(p)) == 2 and p.read_bytes() == b'{"a": 1}\n{"a": 2}\n'
