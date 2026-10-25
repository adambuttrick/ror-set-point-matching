# This module adapts matcher/retrieve.py and matcher/candidates.py of openalex-affiliation-matcher v3.0.0
# (https://github.com/ourresearch/openalex-affiliation-matcher). The names, acronyms and places come from the ROR
# records of a records folder, and the score has no bonus for the works count. Records with the same score are in a
# random order that depends only on the string and their ROR IDs. A screen leaves out, before they are scored, the
# name variants that cannot reach the cut.
#
# Copyright (c) 2026 OurResearch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The lexical candidate generator: ROR records whose names or acronyms occur in an affiliation string.

The index reads the names, acronyms and places of the records in a records folder. Each name variant of a record
scores by the IDF-weighted share of its tokens that occur in a window of one or two comma or semicolon segments of the
string. Acronyms match as case-sensitive whole tokens.

A variant counts only when its score is at least 0.5 and its matched tokens have an IDF of at least 5. Before the
variants of a string are scored, a screen in numpy arrays keeps only the ones that can pass both tests: those that
have a matched token among their "prefix" tokens (the rarest ones, without which the others cannot reach the cut),
and whose matched IDF, summed in any order, is near the cut or above it. The full rule then scores the kept variants.

Records with the same score (often records with the same name) are in the order of a hash of the string and the ROR
ID. Thus the order does not depend on the order of the records in the folder, and no record is first in all ties.

    from ror_set_point.inference.lexical import LexicalIndex
    ix = LexicalIndex("records")                             # a few seconds
    ix.top("Dept. of Physics, Univ. of Tokyo, Japan", 10)    # -> [ROR ID, ...]
"""
import hashlib
import math
import multiprocessing
import re
import unicodedata
from collections import defaultdict

import numpy as np

from ror_set_point.common.records import read_records


class Tokenizer:
    """Tokens for lexical matching: folded, lowercased words; common abbreviations expanded; stop words dropped;
    CJK as character bigrams."""

    def __init__(self):
        self.abbr = {
            "univ": "university", "universidad": "university", "universidade": "university",
            "universite": "university", "universita": "university", "universitat": "university",
            "universitaet": "university", "universiteit": "university", "uniwersytet": "university",
            "inst": "institute", "institut": "institute", "instituto": "institute", "istituto": "institute",
            "dept": "department", "dep": "department", "depto": "department",
            "hosp": "hospital", "hop": "hospital", "hopital": "hospital",
            "natl": "national", "nat": "national", "nacional": "national", "nationale": "national",
            "nazionale": "national",
            "sci": "science", "sciences": "science", "technol": "technology", "tech": "technology",
            "lab": "laboratory", "labs": "laboratory", "laboratories": "laboratory", "lab.": "laboratory",
            "ctr": "center", "cent": "center", "centre": "center", "centro": "center", "zentrum": "center",
            "res": "research", "med": "medical", "coll": "college", "sch": "school", "acad": "academy",
            "fac": "faculty", "engn": "engineering", "eng": "engineering", "agr": "agricultural",
            "st": "saint", "ste": "saint", "mt": "mount", "&": "and", "und": "and", "et": "and", "y": "and",
        }
        self.stop = {"of", "the", "and", "de", "la", "le", "les", "des", "du", "di", "del", "della", "der", "die",
                     "das", "für", "fur", "for", "in", "at", "da", "do", "dos", "e", "i", "a", "an", "en", "van", "von"}
        self.cjk = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
        self.fold_table = str.maketrans({"ı": "i", "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D",
                                         "ß": "ss", "æ": "ae", "œ": "oe", "’": "", "'": "", "`": ""})

    def fold(self, s):
        s = unicodedata.normalize("NFKD", s.translate(self.fold_table))
        return "".join(c for c in s if not unicodedata.combining(c))

    def tokens(self, s):
        s = self.fold(s).lower().replace("&", " and ")
        merged = []  # "m d anderson" -> "md anderson"
        for w in re.findall(r"[^\W_]+", s):
            if len(w) == 1 and not self.cjk.search(w) and merged and merged[-1][1]:
                merged[-1] = (merged[-1][0] + w, True)
            else:
                merged.append((w, len(w) == 1 and not self.cjk.search(w)))
        out = []
        for w, _ in merged:
            if self.cjk.search(w):
                chars = list(w)
                out += [x + y for x, y in zip(chars, chars[1:])] or chars
                continue
            w = self.abbr.get(w, w)
            if w in self.stop:
                continue
            if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
                w = w[:-1]
            out.append(w)
        return out


def segments(s):
    """The comma, semicolon, newline, bar and " - " segments of s, and each pair of neighbouring segments."""
    parts = [p for p in re.split(r"[;,\n|]|\s-\s", s) if p.strip()]
    return parts + [parts[i] + " " + parts[i + 1] for i in range(len(parts) - 1)] or [s]


def ranges(start, length):
    """The concatenation of range(s, s + n) for each s, n in start, length."""
    end = np.cumsum(length)
    return np.arange(end[-1] if len(end) else 0) + np.repeat(start - (end - length), length)


_index = None  # the LexicalIndex of the processes of top_many()


def _set_index(ix):
    global _index
    _index = ix


def _top_part(args):
    texts, k = args
    return [_index.top(s, k) for s in texts]


class LexicalIndex:
    def __init__(self, records):
        """records: a records folder. The index holds the names, acronyms and places of its records."""
        self.tok = Tokenizer()
        self.ror_ids = []            # record number -> ROR ID
        self.places = []             # record number -> its cities and countries, folded and lowercased
        self.variants = []           # (record number, token tuple)
        self.acr = defaultdict(set)  # acronym -> record numbers
        vocab = {}                   # token -> token number
        for n, r in enumerate(read_records(records)):
            self.ror_ids.append(r["ror_id"])
            self.places.append(tuple(dict.fromkeys(self.tok.fold(p).lower() for x in r["places"]
                                                   for p in (x["city"], x["country"]) if p)))
            names = list(dict.fromkeys([r["name"]] + [x["v"] for x in r["labels"]] + r["aliases"]))
            for v in names + [re.sub(r"\s*\([^)]*\)", "", v) for v in names if "(" in v]:
                t = tuple(dict.fromkeys(self.tok.tokens(v)))
                if t:
                    self.variants.append((n, t))
                    for w in t:
                        vocab.setdefault(w, len(vocab))
            for x in set(r["acronyms"]):
                if len(x) >= 2:
                    self.acr[x].add(n)
        self.screen_arrays(vocab)

    def screen_arrays(self, vocab):
        """The arrays of screen(): the tokens of each variant, the IDF and document frequency of each token, the
        cut of each variant, and the posting lists of the prefix tokens."""
        n = len(self.variants)
        tok = np.fromiter((vocab[w] for _, t in self.variants for w in t), np.int64)
        length = np.fromiter((len(t) for _, t in self.variants), np.int64, n)
        start = np.cumsum(length) - length
        df = np.bincount(tok, minlength=len(vocab))
        self.idf = {w: math.log(n / int(d)) for w, d in zip(vocab, df)}
        idf = np.fromiter(self.idf.values(), float, len(vocab))
        var = np.repeat(np.arange(n), length)
        w = idf[tok]
        tot = np.bincount(var, weights=w, minlength=n)
        # A variant counts in lexical() only when its matched IDF is at least 5 and at least half its total IDF
        # (or all of it). The margin covers sums taken in another order.
        cut = np.maximum(5.0, tot / 2) - 1e-6
        # The tokens of each variant from the rarest. rest: the IDF of a token and of the tokens after it. The
        # prefix tokens are those with rest >= cut: a window with none of them cannot reach the cut.
        o = np.lexsort((-w, var))
        rank = np.arange(len(o)) - start[var]
        rest, done = np.empty(len(o)), np.zeros(n)
        for r in range(int(length.max(initial=0))):
            at = np.flatnonzero(rank == r)
            v = var[at]
            rest[at] = tot[v] - done[v]
            done[v] += w[o[at]]
        pre = rest >= cut[var]
        pt, pv = tok[o][pre], var[pre]
        self.vocab = vocab
        self.pre_var = pv[np.argsort(pt, kind="stable")].astype(np.int32)
        self.pre_start = np.concatenate([[0], np.cumsum(np.bincount(pt, minlength=len(vocab)))])
        self.var_tok, self.var_start, self.var_len = tok.astype(np.int32), start, length
        self.df, self.idf_of, self.cut = df, idf, cut

    def screen(self, wins, gather_df=20000):
        """The (window number, variant number) pairs that can pass the tests of lexical(): the window holds one of
        the variant's prefix tokens and one of its tokens with at most gather_df variants, and the matched IDF is
        at least the variant's cut. wins: the token set of each window."""
        ids = [[self.vocab[w] for w in T if w in self.vocab] for T in wins]
        u = np.fromiter((x for i in ids for x in i), np.int64)
        j = np.repeat(np.arange(len(ids)), [len(i) for i in ids])
        num = self.pre_start[u + 1] - self.pre_start[u]
        v = self.pre_var[ranges(self.pre_start[u], num)]
        if not len(v):
            return {}
        win = np.repeat(j, num)
        length = self.var_len[v]
        tok = self.var_tok[ranges(self.var_start[v], length)]
        pair = np.repeat(np.arange(len(v)), length)
        keys = np.sort(j * len(self.vocab) + u)
        key = win[pair] * len(self.vocab) + tok
        hit = keys[np.minimum(np.searchsorted(keys, key), len(keys) - 1)] == key
        p, t = pair[hit], tok[hit]
        got = np.bincount(p, weights=self.idf_of[t], minlength=len(v))
        gathered = np.bincount(p, weights=self.df[t] <= gather_df, minlength=len(v)) > 0
        keep = gathered & (got >= self.cut[v])
        return dict.fromkeys(zip(win[keep].tolist(), v[keep].tolist()))

    def here(self, n, low):
        return any(p in low for p in self.places[n])

    def tie(self, s, n):
        """The sort key of record n among the records with the same score for the string s: a hash of s and the
        ROR ID."""
        return hashlib.blake2b(f"{s}\0{self.ror_ids[n]}".encode(), digest_size=8).digest()

    def lexical(self, s, k=25, gather_df=20000):
        """[(record number, score)]: the best name-coverage score of each record, plus a bonus when the string
        names one of its cities or countries. Records with the same score are in the order of tie()."""
        scores = defaultdict(float)
        wins = [set(self.tok.tokens(win)) for win in segments(s)]
        for j, vi in self.screen(wins, gather_df):
            T = wins[j]
            n, t = self.variants[vi]
            tot = sum(self.idf.get(w, 0) for w in t)
            if tot <= 0:
                continue
            got = sum(self.idf.get(w, 0) for w in t if w in T)
            if got < 5.0:  # generic-only overlap ("hospital", "university of")
                continue
            cov = got / tot + 0.25 * (got >= tot - 1e-9)
            if cov > scores[n]:
                scores[n] = cov
        low = self.tok.fold(s).lower()
        ranked = sorted((-(c + 0.15 * self.here(n, low)), self.tie(s, n), n) for n, c in scores.items() if c >= 0.5)
        return [(n, round(-sc, 3)) for sc, _, n in ranked[:k]]

    def acronyms(self, s, per=6):
        """Record numbers whose acronym is a token of s: per acronym, up to per records, those in the string's
        city or country first, then in the order of tie(). The result is in the same order."""
        low = self.tok.fold(s).lower()
        toks = set(re.findall(r"[A-Za-z][A-Za-z0-9&\-]{1,11}", s))
        toks |= {p for t in list(toks) if "-" in t for p in t.split("-") if len(p) >= 2}  # "IPCF-CNR"
        toks |= {d.replace(".", "") for d in re.findall(r"\b(?:[A-Z]\.){2,8}", s)}  # "U.L.B." -> "ULB"
        hits = [self.acr[w] for w in toks if w in self.acr and any(c.isupper() for c in w)]
        key = {n: (-self.here(n, low), self.tie(s, n)) for ids in hits for n in ids}
        return sorted({n for ids in hits for n in sorted(ids, key=key.get)[:per]}, key=key.get)

    def ranked(self, s):
        """Record numbers in rank order: the lexical top 100, then acronym matches (up to 10 for each acronym) at
        score 0.4."""
        seen = dict(self.lexical(s, k=100))
        for n in self.acronyms(s, per=10):
            seen.setdefault(n, 0.4)
        return [n for n, _ in sorted(seen.items(), key=lambda x: -x[1])][:100]

    def top(self, s, k=10):
        """The ROR IDs of the top k records."""
        return [self.ror_ids[n] for n in self.ranked(s)[:k]]

    def pool(self, workers):
        """workers forked processes for top_many, to keep between calls (not on Windows)."""
        return multiprocessing.get_context("fork").Pool(workers, _set_index, (self,))

    def top_many(self, texts, k=10, workers=1, chunk=256, pool=None):
        """top() for each text. workers > 1 gives parts of chunk texts to forked processes (not on Windows). pool:
        processes from pool(), in place of new ones."""
        if pool is None:
            if workers <= 1 or len(texts) <= chunk:
                return [self.top(s, k) for s in texts]
            with self.pool(workers) as pool:
                return self.top_many(texts, k, chunk=chunk, pool=pool)
        parts = pool.imap(_top_part, [(texts[i:i + chunk], k) for i in range(0, len(texts), chunk)])
        return [x for part in parts for x in part]
