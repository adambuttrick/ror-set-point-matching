import gzip
import json
import random
from collections import defaultdict

import pytest

from ror_set_point.common.text import is_non_latin, normalize, script
from ror_set_point.inference.lexical import LexicalIndex, Tokenizer, segments


def test_normalize_collapses_case_width_and_punctuation():
    assert normalize("Dept. of Physics,  UNIVERSITY of Tokyo") == "dept of physics university of tokyo"
    assert normalize("ＭＩＴ") == "mit"
    assert normalize("Université—Paris") == "université paris"


def test_script_counts_letters_only():
    assert script("北京大学, Beijing 100871") == "Latin"  # 4 Han letters, 7 Latin letters
    assert script("北京大学物理学院, Beijing") == "non-Latin"
    assert is_non_latin("Московский университет, Moscow")
    assert not is_non_latin("12345, ---")


def test_tokens_expand_abbreviations_and_drop_stop_words():
    tok = Tokenizer()
    assert tok.tokens("Univ. of Tokyo") == ["university", "tokyo"]
    assert tok.tokens("Natl. Inst. of Health Sciences") == ["national", "institute", "health", "science"]


def test_tokens_join_initials_and_use_cjk_bigrams():
    tok = Tokenizer()
    assert tok.tokens("M D Anderson Cancer Center") == ["md", "anderson", "cancer", "center"]
    assert tok.tokens("北京大学") == ["北京", "京大", "大学"]


def test_segments_add_neighbouring_pairs():
    assert segments("Dept. of Physics, Univ. of Tokyo; Japan") == [
        "Dept. of Physics", " Univ. of Tokyo", " Japan", "Dept. of Physics  Univ. of Tokyo", " Univ. of Tokyo  Japan"]
    assert segments("no separators") == ["no separators"]


def record(i, name, labels=(), aliases=(), acronyms=(), places=()):
    """A row of a records folder, with the fields that the lexical index reads."""
    return {"ror_id": f"ror-{i}", "status": "active", "name": name, "acronyms": list(acronyms),
            "labels": [{"v": v, "lang": None} for v in labels], "aliases": list(aliases),
            "places": [{"city": c, "region": None, "country": k, "cc": None} for c, k in places]}


def records_folder(path, rows):
    path.mkdir()
    with gzip.open(path / "records.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return str(path)


def rare_words(rng, n):
    return ["".join(rng.choice("bcdfgklmnprstvz") + rng.choice("aeiou") for _ in range(rng.randint(2, 4)))
            for _ in range(n)]


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    """An index of 600 made-up records, whose names mix generic words and rare words, and strings that hold parts of
    their names."""
    rng = random.Random(0)
    generic = ["University", "Institute", "Hospital", "National", "Research", "Center", "College", "Medical",
               "Science", "Technology", "School", "Foundation", "Laboratory", "State", "of", "the", "and"]
    rare = rare_words(rng, 700)
    places = [(w.title(), c) for w, c in zip(rare[:60], ["Japan", "France", "Chile", "Kenya"] * 15)]

    def name():
        words = rng.sample(generic, rng.randint(0, 4)) + rng.sample(rare, rng.randint(1, 3))
        rng.shuffle(words)
        return " ".join(w.title() for w in words)
    rows = []
    for i in range(1, 601):
        city, country = rng.choice(places)
        alts = [name() for _ in range(rng.randint(0, 3))] + [name() + f" ({city})"] * rng.randint(0, 1)
        n = rng.randint(0, len(alts))
        more = [rng.choice(places)] if i % 3 == 0 else []
        rows.append(record(i, name(), alts[:n], alts[n:], ["AB"], [(city, country)] + more))
    texts = []
    for _ in range(400):
        parts = []
        for _ in range(rng.randint(1, 4)):
            words = (rng.choice(rows)["name"] + " " + rng.choice(rare + generic)).split()
            parts.append(" ".join(w for w in words if rng.random() < 0.8))
        texts.append(rng.choice([", ", "; ", " - "]).join(parts + [rng.choice(places)[rng.randint(0, 1)]]))
    return LexicalIndex(records_folder(tmp_path_factory.mktemp("lexical") / "records", rows)), texts


def lexical_without_screen(ix, s, k, gather_df, post):
    """lexical() without the screen: every variant that has a token of the window is scored."""
    scores = defaultdict(float)
    for win in segments(s):
        T = set(ix.tok.tokens(win))
        if not T:
            continue
        cand = set()
        for w in T:
            p = post.get(w)
            if p and len(p) <= gather_df:
                cand |= p
        for vi in cand:
            n, t = ix.variants[vi]
            tot = sum(ix.idf.get(w, 0) for w in t)
            if tot <= 0:
                continue
            got = sum(ix.idf.get(w, 0) for w in t if w in T)
            if got < 5.0:
                continue
            cov = got / tot + 0.25 * (got >= tot - 1e-9)
            if cov > scores[n]:
                scores[n] = cov
    low = ix.tok.fold(s).lower()
    ranked = sorted(((c + 0.15 * ix.here(n, low), n) for n, c in scores.items() if c >= 0.5),
                    key=lambda x: (-x[0], ix.tie(s, x[1])))
    return [(n, round(sc, 3)) for sc, n in ranked[:k]]


@pytest.mark.parametrize("gather_df", [20000, 40])
def test_screened_lexical_is_the_score_of_every_variant(index, gather_df):
    ix, texts = index
    post = defaultdict(set)
    for vi, (_, t) in enumerate(ix.variants):
        for w in t:
            post[w].add(vi)
    assert min(ix.idf.values()) < 3 and max(ix.idf.values()) > 5
    found = 0
    for s in texts:
        want = lexical_without_screen(ix, s, 100, gather_df, post)
        assert ix.lexical(s, 100, gather_df) == want, s
        found += len(want)
    assert found > len(texts)


def test_top_many_in_processes_is_top(index):
    ix, texts = index
    assert ix.top_many(texts, 10, workers=3, chunk=50) == [ix.top(s, 10) for s in texts]


def test_top_many_with_a_kept_pool_is_top(index):
    ix, texts = index
    with ix.pool(3) as pool:
        for part in (texts[:20], texts[20:]):
            assert ix.top_many(part, 10, chunk=50, pool=pool) == [ix.top(s, 10) for s in part]


@pytest.fixture(scope="module")
def tied(tmp_path_factory):
    """40 records with the same name and acronym, each in its own town, among 2,000 other records. Two indexes
    with the records in two orders."""
    rng = random.Random(1)
    words = rare_words(rng, 3000)
    rows = [record(i, " ".join(rng.sample(words, 2)), places=[(rng.choice(words).title(), "Ruritania")])
            for i in range(2000)]
    rows += [record(5000 + i, "Zubrovka Grand Hospital", acronyms=["ZGH"], places=[(f"Town{i}", "Ruritania")])
             for i in range(40)]
    rng.shuffle(rows)
    tmp = tmp_path_factory.mktemp("tied")
    return LexicalIndex(records_folder(tmp / "a", rows)), LexicalIndex(records_folder(tmp / "b", rows[::-1]))


TIED_TEXTS = [f"Department of Unit {i}, Zubrovka Grand Hospital" for i in range(60)]


def test_ties_do_not_depend_on_the_order_of_the_records(tied):
    a, b = tied
    assert [a.top(s, 10) for s in TIED_TEXTS] == [b.top(s, 10) for s in TIED_TEXTS]
    for s in ("ZGH, Department of Unit 3", "ZGH, Town7"):
        assert [a.ror_ids[n] for n in a.acronyms(s, 10)] == [b.ror_ids[n] for n in b.acronyms(s, 10)]


def test_no_record_is_first_in_all_ties(tied):
    a, _ = tied
    top = [a.top(s, 10) for s in TIED_TEXTS]
    tied_ids = {f"ror-{5000 + i}" for i in range(40)}
    assert all(len(t) == 10 and set(t) <= tied_ids for t in top)
    assert len({t[0] for t in top}) >= 20  # 40 tied records, 60 strings


def test_a_named_place_comes_before_a_tie(tied):
    a, _ = tied
    assert a.top("Zubrovka Grand Hospital, Town7", 1) == ["ror-5007"]
    hits = a.acronyms("ZGH, Town7", 10)
    assert len(hits) == 10 and a.ror_ids[hits[0]] == "ror-5007"
