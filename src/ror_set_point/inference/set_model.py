"""The set model: one cross-encoder reads an affiliation string together with all its candidate ROR records, and
gives each candidate the probability that the string names it.

The input is [start] string [sep] record 1 [sep] record 2 [sep] ... A linear layer scores each candidate from the
mean of its token states. Platt scaling, p = 1 / (1 + exp(-(a * score + b))), makes the scores probabilities. The
answer is every candidate with p >= the threshold (0.5).

    m = SetMatcher.from_pretrained("adambuttrick/ror-set-point-set-model-mmbert-base", record_texts)
    m.predict([("Dept. of Physics, University of Tokyo, Japan", ["https://ror.org/057zh3y96", ...])])
    # -> [[("https://ror.org/057zh3y96", 0.99...)]]

record_texts is {ROR id: text}, from the text field of the ror_records config of the dataset
adambuttrick/ror-set-point-matching-train-validation-test. A saved model folder holds the encoder (transformers
format), head.safetensors, the tokenizer and set_model_config.json (input limits, calibration and threshold). This
file needs only torch, transformers, safetensors, huggingface_hub and numpy.

There are two ways to run the model. The padded way (scores) puts a batch of strings in a [batch, longest input]
tensor. Training and CPU or MPS inference use it. The packed way (packed_scores) puts the inputs of a batch one after
the other in one sequence, with no padding, and runs flash-attention 2 on each input. It needs a CUDA GPU, the
flash-attn package and a ModernBERT encoder (such as mmBERT).
"""
import itertools
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import torch


class SetModel(torch.nn.Module):
    """An encoder and a linear head on the mean of each candidate's token states."""

    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder
        self.head = torch.nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, ids, mask, seg, n):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        B, _, D = h.shape
        w = (seg >= 0).float().unsqueeze(-1)
        idx = (seg.clamp_min(0) + torch.arange(B, device=ids.device).unsqueeze(1) * n).view(-1)
        sums = torch.zeros(B * n, D, device=ids.device).index_add_(0, idx, (h * w).view(-1, D))
        cnt = torch.zeros(B * n, device=ids.device).index_add_(0, idx, w.view(-1))
        return self.head(sums / cnt.clamp_min(1.0).unsqueeze(-1)).view(B, n)

    def forward_packed(self, ids, cand, cnt, cu_seqlens, max_seqlen):
        """The score of each candidate of a packed batch (Packer.pack). The head is linear, so the score of a
        candidate is the mean over its tokens of head(token state)."""
        h = self.encode_packed(ids, cu_seqlens, max_seqlen)
        t = h.float() @ self.head.weight[0].float()
        sums = torch.zeros(len(cnt) + 1, device=ids.device).index_add_(0, cand, t)
        return sums[:-1] / cnt + self.head.bias.float()

    def encode_packed(self, ids, cu_seqlens, max_seqlen):
        """The final token states of a packed batch. The encoder must be a ModernBERT model loaded with
        attn_implementation="flash_attention_2"; its layers then take the packed input as it is."""
        enc = self.enc
        h = enc.embeddings(input_ids=ids)
        for layer in enc.layers:
            h = layer(h, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)[0]
        return enc.final_norm(h)


def flash_attn_available():
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def load_encoder(name_or_path, revision=None, dtype=torch.float32, attn_implementation=None, compile=False):
    """The encoder, in fp32 unless dtype is given. ModernBERT models use SDPA attention unless attn_implementation
    says otherwise (transformers would pick flash-attention 2 when flash-attn is installed, but the released models
    trained with SDPA), and torch.compile their embeddings and feed-forward blocks only when compile is set."""
    from transformers import AutoConfig, AutoModel
    conf = AutoConfig.from_pretrained(name_or_path, revision=revision)
    kw = {}
    if conf.model_type == "modernbert":
        kw = {"reference_compile": compile, "attn_implementation": attn_implementation or "sdpa"}
    elif attn_implementation:
        kw = {"attn_implementation": attn_implementation}
    return AutoModel.from_pretrained(name_or_path, revision=revision, dtype=dtype, **kw)


def ranges(start, length):
    """The concatenation of range(s, s + n) for each s, n in start, length."""
    end = np.cumsum(length)
    return np.arange(end[-1] if len(end) else 0) + np.repeat(start - (end - length), length)


class Packer:
    """Makes the model input for strings and their candidates. Record texts are tokenized once."""

    def __init__(self, tok, record_texts, string_max_tokens, card_max_tokens, max_tokens):
        self.tok = tok
        self.slen, self.card, self.max_len = string_max_tokens, card_max_tokens, max_tokens
        self.start = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        self.sep = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else self.sep
        ids = list(record_texts)
        self.ctok = {}
        for s in range(0, len(ids), 8192):
            part = ids[s:s + 8192]
            enc = tok([record_texts[x] for x in part], add_special_tokens=False, truncation=True,
                      max_length=card_max_tokens)["input_ids"]
            self.ctok.update(zip(part, enc))
        # For the packed layout, all record tokens in one array: record j is flat[off[j]:off[j] + rlen[j]].
        self.rid = {x: j for j, x in enumerate(self.ctok)}
        self.rlen = np.array([len(t) for t in self.ctok.values()], dtype=np.int64)
        self.off = np.cumsum(self.rlen) - self.rlen
        self.flat = np.fromiter(itertools.chain.from_iterable(self.ctok.values()), dtype=np.int64,
                                count=int(self.rlen.sum()))

    def __call__(self, items, device):
        """items: [(string, [candidate ROR ids])], each with one candidate or more. Returns ids, mask and seg
        (the candidate of each token, -1 for others) on device, and n, the largest candidate count."""
        n = max(len(c) for _, c in items)
        seqs, segs = [], []
        for text, cands in items:
            s = [self.start] + self.tok(text, add_special_tokens=False, truncation=True,
                                        max_length=self.slen)["input_ids"] + [self.sep]
            per = min(self.card, (self.max_len - len(s)) // len(cands) - 1)
            g = [-1] * len(s)
            for i, x in enumerate(cands):
                t = self.ctok[x][:per]
                s += t + [self.sep]
                g += [i] * len(t) + [-1]
            seqs.append(s)
            segs.append(g)
        T = max(len(s) for s in seqs)
        ids = torch.full((len(items), T), self.pad, dtype=torch.long)
        mask = torch.zeros((len(items), T), dtype=torch.long)
        seg = torch.full((len(items), T), -1, dtype=torch.long)
        for i, (s, g) in enumerate(zip(seqs, segs)):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
            seg[i, :len(g)] = torch.tensor(g)
        return ids.to(device), mask.to(device), seg.to(device), n

    def plan(self, items):
        """The packed layout of items [(string, [candidate ROR ids])], with the same tokens as __call__. The strings
        are tokenized in one call. Returns st (the string tokens), n (the candidate count), first (the place of each
        item's first candidate in c and k), c (the record of each candidate), k (its token count) and length (the
        input length of each item)."""
        st = self.tok([t for t, _ in items], add_special_tokens=False, truncation=True,
                      max_length=self.slen)["input_ids"] if items else []
        n = np.array([len(cs) for _, cs in items], dtype=np.int64)
        c = np.array([self.rid[x] for _, cs in items for x in cs], dtype=np.int64)
        head = np.array([len(s) + 2 for s in st], dtype=np.int64)
        per = np.repeat(np.minimum(self.card, (self.max_len - head) // np.maximum(n, 1) - 1), n)
        r = self.rlen[c]
        k = np.where(per >= 0, np.minimum(r, per), np.maximum(r + per, 0))  # len(t[:per]), as in __call__
        item = np.repeat(np.arange(len(n)), n)
        length = head + np.bincount(item, weights=k + 1, minlength=len(n)).astype(np.int64)
        return SimpleNamespace(st=st, n=n, first=np.cumsum(n) - n, c=c, k=k, length=length)

    def pack(self, p, sel):
        """The inputs of the items sel (indices into plan p, each with one candidate or more), one after the other.
        Returns numpy arrays: ids (the tokens), cand (the candidate of each token, from 0 to C - 1 in the order of
        the items and their candidates, or C for other tokens), cnt (the token count of each candidate, at least
        1), cu_seqlens (where each input starts, then the total), and max_seqlen (an int)."""
        sel = np.asarray(sel, dtype=np.int64)
        n = p.n[sel]
        ci = ranges(p.first[sel], n)
        c, k = p.c[ci], p.k[ci]
        B, C = len(sel), len(c)
        heads = [[self.start] + p.st[i] + [self.sep] for i in sel]
        hl = np.array([len(h) for h in heads], dtype=np.int64)
        # Pieces in input order: each item's head ([start] string [sep]), then each candidate's tokens and [sep].
        is_head = np.zeros(B + C, dtype=bool)
        is_head[np.arange(B) + np.cumsum(n) - n] = True
        size = np.empty(B + C, dtype=np.int64)
        size[is_head], size[~is_head] = hl, k + 1
        at = np.cumsum(size) - size
        ids = np.empty(int(size.sum()), dtype=np.int64)
        ids[ranges(at[is_head], hl)] = np.fromiter(itertools.chain.from_iterable(heads), dtype=np.int64,
                                                   count=int(hl.sum()))
        rec = ranges(at[~is_head], k)
        ids[rec] = self.flat[ranges(self.off[c], k)]
        ids[at[~is_head] + k] = self.sep
        cand = np.full(len(ids), C, dtype=np.int64)
        cand[rec] = np.repeat(np.arange(C), k)
        length = p.length[sel]
        cu = np.zeros(B + 1, dtype=np.int32)
        np.cumsum(length, out=cu[1:])
        return ids, cand, np.maximum(k, 1).astype(np.float32), cu, int(length.max())


@torch.no_grad()
def scores(model, packer, items, device, batch_size=32):
    """The raw score of each candidate of each item, rounded to 4 decimals. Items go through the model in order of
    size; the result is in the order of items."""
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i][1]) * 40 + len(items[i][0]))
    out = [[] for _ in items]
    order = [i for i in order if items[i][1]]
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask, seg, n = packer([items[i] for i in idx], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = model(ids, mask, seg, n).float().cpu()
        for j, i in enumerate(idx):
            out[i] = [round(v, 4) for v in z[j, :len(items[i][1])].tolist()]
    return out


def token_batches(lengths, order, batch_tokens):
    """Cut order (item indices) into runs whose lengths sum to at most batch_tokens (or one item, if it is
    longer)."""
    out, cur, size = [], [], 0
    for i in order:
        if cur and size + lengths[i] > batch_tokens:
            out.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += lengths[i]
    return out + [cur] if cur else out


@torch.no_grad()
def packed_scores(model, packer, items, device, batch_tokens=65536):
    """As scores, with packed batches of up to batch_tokens tokens (model.forward_packed), in the dtype of the
    model's weights. The CPU makes the next batch while the GPU runs the last one, and the scores stay on the GPU to
    the end."""
    model.eval()
    p = packer.plan(items)
    order = [i for i in np.argsort(-p.length, kind="stable").tolist() if p.n[i]]
    batches = token_batches(p.length.tolist(), order, batch_tokens)
    pin = device.type == "cuda"
    z = []
    for sel in batches:
        ids, cand, cnt, cu, m = packer.pack(p, sel)
        ids, cand, cnt, cu = ((torch.from_numpy(x).pin_memory() if pin else torch.from_numpy(x))
                              .to(device, non_blocking=True) for x in (ids, cand, cnt, cu))
        z.append(model.forward_packed(ids, cand, cnt, cu, m))
    z = torch.cat(z).cpu().tolist() if z else []
    out = [[] for _ in items]
    at = 0
    for i in itertools.chain.from_iterable(batches):
        out[i] = [round(v, 4) for v in z[at:at + p.n[i]]]
        at += p.n[i]
    return out


def calibrate(z, a, b):
    return round(1 / (1 + math.exp(-(a * z + b))), 4)


def model_folder(name_or_path):
    """A local folder, or the local copy of a model repository on the Hugging Face Hub."""
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_path, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])


class SetMatcher:
    def __init__(self, path, record_texts, device=None, compile=True):
        """path: a saved set-model folder. record_texts: {ROR id: record text} (the text field of the records).

        On a CUDA GPU the model runs in bf16. A ModernBERT encoder then runs packed with flash-attention 2 when
        flash-attn is installed (self.packed), and padded with bf16 autocast otherwise. compile: torch.compile the
        encoder's embeddings and feed-forward blocks (packed only; the first batch takes a few seconds longer). On
        other devices the model runs padded in fp32."""
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer
        with open(os.path.join(path, "set_model_config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cuda = self.device.type == "cuda"
        self.packed = cuda and AutoConfig.from_pretrained(path).model_type == "modernbert" and flash_attn_available()
        self.model = SetModel(load_encoder(path, dtype=torch.bfloat16 if cuda else torch.float32,
                                           attn_implementation="flash_attention_2" if self.packed else None,
                                           compile=self.packed and compile))
        head = load_file(os.path.join(path, "head.safetensors"))
        self.model.head.load_state_dict({k: v.float() for k, v in head.items()})
        self.model.to(self.device).eval()
        c = self.cfg
        self.packer = Packer(AutoTokenizer.from_pretrained(path), record_texts, c["string_max_tokens"],
                             c["card_max_tokens"], c["max_tokens"])

    @classmethod
    def from_pretrained(cls, name_or_path, record_texts, device=None, **kw):
        """A local folder, or a model repository on the Hugging Face Hub. kw goes to SetMatcher."""
        return cls(model_folder(name_or_path), record_texts, device, **kw)

    def scores(self, items, batch_size=32, batch_tokens=65536):
        """The raw score of each candidate of each item. batch_size: strings in a padded batch. batch_tokens:
        tokens in a packed batch."""
        if self.packed:
            return packed_scores(self.model, self.packer, items, self.device, batch_tokens)
        return scores(self.model, self.packer, items, self.device, batch_size)

    def probabilities(self, items, batch_size=32, batch_tokens=65536):
        """items: [(string, [candidate ROR ids])]. Returns [(ROR id, p)] for every candidate of each item."""
        a, b = self.cfg["calibration"]["a"], self.cfg["calibration"]["b"]
        z = self.scores(items, batch_size, batch_tokens)
        return [[(x, calibrate(v, a, b)) for x, v in zip(c, zz)] for (_, c), zz in zip(items, z)]

    def predict(self, items, batch_size=32, batch_tokens=65536):
        """items: [(string, [candidate ROR ids])]. Returns the named records [(ROR id, p)] of each item."""
        t = self.cfg["threshold"]
        return [[(x, p) for x, p in row if p >= t] for row in self.probabilities(items, batch_size, batch_tokens)]
