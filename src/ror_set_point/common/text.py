"""Text rules that every step uses: string normalization for equality checks, and the script of a string."""
import re
import unicodedata


def normalize(s):
    """NFKC, casefold, and every run of non-word characters as one space. Two strings are "equal" when their
    normalized forms are equal."""
    return " ".join(re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", s).casefold()).split())


def is_non_latin(s):
    """True when fewer than half of the letters of s are Latin letters."""
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum("LATIN" in unicodedata.name(c, "") for c in letters) / len(letters) < 0.5


def script(s):
    return "non-Latin" if is_non_latin(s) else "Latin"
