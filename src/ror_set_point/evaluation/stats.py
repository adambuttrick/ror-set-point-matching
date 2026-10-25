"""Statistics for paired comparisons of exact match, and bootstrap intervals for precision and recall."""
import functools
import math
import random


def log_binom_pmf(k, n, p):
    if p <= 0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1 - p))


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    return min(1.0, sum(math.exp(log_binom_pmf(i, n, p)) for i in range(0, k + 1)))


def mcnemar_exact(a_only, b_only):
    """The two-sided p-value of the exact McNemar test: a binomial test of the discordant strings at p = 0.5."""
    d = a_only + b_only
    if not d:
        return 1.0
    return min(1.0, 2 * binom_cdf(min(a_only, b_only), d, 0.5))


@functools.lru_cache(maxsize=None)
def upper_critical(d, alpha):
    """The smallest x for which the exact two-sided binomial(d, 0.5) test rejects when X >= x."""
    tail, x = 0.0, d + 1
    for k in range(d, -1, -1):
        t = tail + math.exp(log_binom_pmf(k, d, 0.5))
        if 2 * t <= alpha:
            tail, x = t, k
        else:
            break
    return x


def power_exact(n, psi, delta, alpha=0.05):
    """The power of the exact conditional McNemar test, for n strings, discordance psi and true gain delta.
    Only rejections in the right direction count."""
    q = (psi + delta) / (2 * psi)
    mu, sd = n * psi, math.sqrt(n * psi * (1 - psi))
    power = 0.0
    for d in range(max(0, int(mu - 10 * sd)), min(n, int(mu + 10 * sd) + 1) + 1):
        pd = math.exp(log_binom_pmf(d, n, psi))
        if pd < 1e-14:
            continue
        x = upper_critical(d, alpha)
        if x <= d:
            power += pd * (1 - binom_cdf(x - 1, d, q))
    return power


def mde_exact(n, psi, alpha=0.05, power=0.8):
    """The minimum detectable effect: the smallest exact-match gain that the exact McNemar test detects with the
    given power, for n strings and discordance psi (the share of strings where exactly one system is right)."""
    if psi <= 0:
        return float("nan")
    lo, hi = 0.0, psi
    for _ in range(40):
        mid = (lo + hi) / 2
        if power_exact(n, psi, mid, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi


def bootstrap(per_string, n_resamples, seed, level=0.95):
    """Percentile intervals for exact match, precision and recall. per_string: [(exact 0/1, true positives, named,
    gold)] for each string; strings are resampled with replacement."""
    rng = random.Random(seed)
    n = len(per_string)
    stats = {"exact": [], "precision": [], "recall": []}
    for _ in range(n_resamples):
        ex = tp = named = gold = 0
        for _ in range(n):
            e, t, m, g = per_string[rng.randrange(n)]
            ex, tp, named, gold = ex + e, tp + t, named + m, gold + g
        stats["exact"].append(ex / n)
        stats["precision"].append(tp / named if named else float("nan"))
        stats["recall"].append(tp / gold if gold else float("nan"))
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for k, xs in stats.items():
        xs = sorted(x for x in xs if x == x)
        out[k] = (xs[int(lo * (len(xs) - 1))], xs[int(hi * (len(xs) - 1))]) if xs else (float("nan"), float("nan"))
    return out
