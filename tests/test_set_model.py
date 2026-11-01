import collections
from types import SimpleNamespace

import pytest
import torch
from transformers import BertConfig, BertModel

from ror_set_point.common.io import read_jsonl, write_jsonl
from ror_set_point.inference.set_model import (Packer, SetModel, calibrate, packed_scores, ranges, scores,
                                               token_batches)
from ror_set_point.training.set_model.build import mask_mentions, mention_test
from ror_set_point.training.set_model.train import targets


class CharTokenizer:
    """One token for each character; enough to test the input layout."""
    cls_token_id, sep_token_id, pad_token_id, bos_token_id, eos_token_id = 1, 2, 0, None, None

    def __call__(self, texts, add_special_tokens=False, truncation=True, max_length=None):
        one = isinstance(texts, str)
        ids = [[3 + ord(c) % 90 for c in t][:max_length] for t in ([texts] if one else texts)]
        return {"input_ids": ids[0] if one else ids}


def packer(max_tokens=64):
    return Packer(CharTokenizer(), {"a": "aaaa", "b": "bb", "c": "cccccc"}, 5, 4, max_tokens)


class TokenwiseEncoder(torch.nn.Module):
    """A token's state does not depend on the other tokens, so padded and packed inputs give the same states."""

    def __init__(self, d=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=d)
        self.emb = torch.nn.Embedding(100, d)
        self.lin = torch.nn.Linear(d, d)
        self.norm = torch.nn.LayerNorm(d)

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)
        return SimpleNamespace(last_hidden_state=self.norm(h + torch.tanh(self.lin(h))))


class TokenwiseSetModel(SetModel):
    """SetModel with an encoder that runs on packed inputs without flash-attention."""

    def encode_packed(self, ids, cu_seqlens, max_seqlen):
        return self.enc(ids).last_hidden_state


def test_layout_marks_each_candidate():
    ids, mask, seg, n = packer()([("xyz", ["a", "b"])], torch.device("cpu"))
    assert n == 2
    # [CLS] x y z [SEP] a a a a [SEP] b b [SEP]
    assert ids[0].tolist()[:5] == [1, 3 + ord("x") % 90, 3 + ord("y") % 90, 3 + ord("z") % 90, 2]
    assert seg[0].tolist() == [-1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 1, 1, -1]
    assert mask[0].sum().item() == 13


def test_records_shrink_when_the_input_is_too_long():
    _, _, seg, _ = packer(max_tokens=14)([("xyz", ["a", "b", "c"])], torch.device("cpu"))
    # (14 - 5) // 3 - 1 = 2 tokens for each record
    assert [seg[0].tolist().count(i) for i in range(3)] == [2, 2, 2]


def test_ranges():
    assert ranges([5, 0, 9], [2, 0, 3]).tolist() == [5, 6, 9, 10, 11]
    assert ranges([], []).tolist() == []


def test_token_batches_cut_at_the_budget():
    assert token_batches([3, 4, 9, 2], [0, 1, 2, 3], 7) == [[0, 1], [2], [3]]
    assert token_batches([3, 4], [], 7) == []


@pytest.mark.parametrize("max_tokens", [64, 14, 6])
def test_packed_layout_is_the_padded_layout_without_padding(max_tokens):
    # 64: whole records; 14: records cut to 2 tokens; 6: (6 - 5) // 3 - 1 = -1, so t[:-1] in __call__
    pk = packer(max_tokens)
    items = [("xyz", ["a", "b", "c"]), ("q", ["b"]), ("", ["c", "a"])]
    ids, mask, seg, _ = pk(items, torch.device("cpu"))
    p = pk.plan(items)
    sel = [2, 0, 1]
    fids, cand, cnt, cu, m = pk.pack(p, sel)
    first = 0
    for j, i in enumerate(sel):
        T = mask[i].sum().item()
        assert cu[j + 1] - cu[j] == T == p.length[i]
        assert fids[cu[j]:cu[j + 1]].tolist() == ids[i, :T].tolist()
        assert [x - first if x < len(cnt) else -1 for x in cand[cu[j]:cu[j + 1]]] == seg[i, :T].tolist()
        assert cnt[first:first + len(items[i][1])].tolist() == [max(1, (seg[i] == x).sum().item())
                                                                 for x in range(len(items[i][1]))]
        first += len(items[i][1])
    assert first == len(cnt) and m == max(mask.sum(1).tolist())


def test_candidate_score_is_head_of_mean_token_state():
    torch.manual_seed(0)
    enc = BertModel(BertConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=32))
    model = SetModel(enc).eval()
    ids, mask, seg, n = packer()([("xyz", ["a", "b"]), ("q", ["c"])], torch.device("cpu"))
    with torch.no_grad():
        z = model(ids, mask, seg, n)
        h = enc(input_ids=ids, attention_mask=mask).last_hidden_state
        want = model.head(h[0][seg[0] == 1].mean(0)).item()
    assert abs(z[0, 1].item() - want) < 1e-5
    assert z.shape == (2, 2)


def test_scores_keep_item_order():
    torch.manual_seed(0)
    enc = BertModel(BertConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=32))
    model, p = SetModel(enc), packer()
    items = [("a long string here", ["a", "b", "c"]), ("s", ["a"]), ("nothing", [])]
    out = scores(model, p, items, torch.device("cpu"), batch_size=1)
    assert [len(x) for x in out] == [3, 1, 0]
    alone = scores(model, p, [items[1]], torch.device("cpu"))
    assert out[1] == alone[0]


@pytest.mark.parametrize("batch_tokens", [1, 30, 10_000])
def test_packed_scores_are_the_padded_scores(batch_tokens):
    torch.manual_seed(0)
    model, pk = TokenwiseSetModel(TokenwiseEncoder()), packer(14)
    items = [("a long string here", ["a", "b", "c"]), ("s", ["a"]), ("nothing", []), ("xy", ["c", "b"]),
             ("pq", ["b", "a", "c", "a"])]
    want = scores(model, pk, items, torch.device("cpu"), batch_size=2)
    got = packed_scores(model, pk, items, torch.device("cpu"), batch_tokens)
    assert [len(x) for x in got] == [3, 1, 0, 2, 4]
    for g, w in zip(got, want):
        assert g == pytest.approx(w, abs=2e-4)


def test_calibrate_is_a_rounded_sigmoid():
    assert calibrate(0.0, 1.0, 0.0) == 0.5
    assert calibrate(2.0, 0.8, -0.1) == round(1 / (1 + 2.718281828459045 ** -1.5), 4)


def ror(i, name, acronyms=(), aliases=()):
    return {"ror_id": f"r{i}", "name": name, "acronyms": list(acronyms), "labels": [], "aliases": list(aliases)}


RECORDS = [ror(1, "Centre National de la Recherche Scientifique", ["CNRS"]), ror(2, "Ministry of Health"),
           ror(3, "Ministry of Health"), ror(4, "Universidade Federal do Rio Grande", ["FURG"]),
           ror(5, "Universidade Federal do Rio Grande do Norte", ["UFRN"]), ror(6, "Kyoto University", ["KU"], ["Kyodai"]),
           ror(7, "Universidad Nacional", ["UNA"]), ror(8, "Universidade Nova de Angola", ["UNA"])]


def test_a_mention_is_a_name_or_acronym_of_one_record():
    m = mention_test(RECORDS)
    assert m("UMR 8000, CNRS, Paris", "r1")
    assert not m("UMR 8000, cnrs, Paris", "r1")  # acronyms keep their case
    assert m("Centre national de la recherche scientifique (Paris)", "r1")
    assert m("Dept. of Physics, Kyodai", "r6")
    assert not m("Ministry of Health, Lima", "r2")  # two records have this name
    assert not m("UNA, Lima", "r7")  # two records have this acronym
    assert not m("KU, Japan", "r6")  # fewer than 3 characters
    s = "Universidade Federal do Rio Grande do Norte, Natal"
    assert m(s, "r4")
    assert not m(s, "r4", gold=["r5"])  # part of a name of a gold record
    assert m("Universidade Federal do Rio Grande; UFRN", "r4", gold=["r5"])


def test_mask_mentions_leaves_out_mentioned_negatives_that_have_no_positive(tmp_path):
    stage1 = [{"s": "Lab, CNRS, Paris", "c": ["r1", "r6", "r2"], "y": [0.1, 0.0, 0.0], "src": "teacher"},
              {"s": "Kyoto University", "c": ["r6", "r1"], "y": [0.9, 0.0], "src": "teacher"}]
    stage2 = [{"k": "a", "s": "Kyoto University; Ministry of Health", "c": ["r6", "r2", "r7"], "y": [0.0, 1.0, 0.0],
               "fold": 0},
              {"k": "b", "s": "Universidade Federal do Rio Grande do Norte", "c": ["r5", "r4"], "y": [1.0, 0.0],
               "fold": 1}]
    write_jsonl(str(tmp_path / "stage1.jsonl.gz"), stage1)
    write_jsonl(str(tmp_path / "stage2.jsonl.gz"), stage2)
    counts = collections.Counter()
    mask_mentions(str(tmp_path), RECORDS, counts)
    out1 = list(read_jsonl(str(tmp_path / "stage1.jsonl.gz")))
    out2 = list(read_jsonl(str(tmp_path / "stage2.jsonl.gz")))
    # r1 has no positive and the string has its acronym. r6 has a positive (stage 1), and r4 is a part of r5's name
    assert out1[0]["m"] == [0, 1, 1]
    assert out1[1] == stage1[1] and out2 == stage2
    assert counts == {"stage1_masked_rows": 1, "stage1_masked_candidates": 1}
    assert sorted(x.name for x in tmp_path.iterdir()) == ["stage1.jsonl.gz", "stage2.jsonl.gz"]


def test_masked_candidates_are_not_in_the_loss():
    rows = [{"c": ["a", "b", "c"], "y": [1.0, 0.0, 0.0], "m": [1, 0, 1]}, {"c": ["a"], "y": [0.0]}]
    y, real = targets(rows, 3, torch.device("cpu"))
    assert real.tolist() == [[True, False, True], [True, False, False]]
    assert y.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
