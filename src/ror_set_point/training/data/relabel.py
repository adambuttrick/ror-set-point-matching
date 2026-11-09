"""Ask a teacher model one question about each (string, ROR record) pair through a /v1/systemone server, and make a
teacher_judgments config from the answers. Four modes:

  requests   one request line for each pair, with a card made from the ROR record and the question of --question. The
             pairs are the judgments of --teacher (a teacher_judgments folder), or the candidates of --stage2 (the stage
             2 rows of ror_set_point.training.set_model.build, which have labels and folds). The input rows are
             numbered in order (for --teacher: train, then validation). Shard i of --shards holds the rows whose
             number modulo --shards is i, with all their pairs.
  score      sends the requests to --url, --concurrency at a time, and adds one line for each pair to --out: the
             request line without its body, with p (or err). A pair that has a p in --out is not sent again, so a run
             that stopped continues. At the end, the step reads --out again. It stops with an error when a pair has no
             p.
  calibrate  fits sigmoid(a * logit(p) + b) to the labels of scored --stage2 pairs (Platt scaling), on all folds. As a
             check, it also fits on --fit-folds and gives the log loss on the other folds.
  assemble   the config of --teacher with the calibrated p, rounded to 4 decimals. Each pair of --teacher must have a
             p in --scores. Writes into --out (which must not exist): train.jsonl and validation.jsonl (the format and
             order of --teacher), raw.jsonl.gz {split, id, ror_id, p_raw (the p of the server), p_jev (the p of
             --teacher)} and manifest.json.

    python -m ror_set_point.training.data.relabel requests --teacher teacher_judgments --records records \
        --question src/ror_set_point/training/data/questions/v1.2.txt --shard 0 --shards 24 --out requests.jsonl.gz
    python -m ror_set_point.training.data.relabel score --requests requests.jsonl.gz --out scores.jsonl
    python -m ror_set_point.training.data.relabel calibrate --scores stage2/*/scores.jsonl --out calibration.json
    python -m ror_set_point.training.data.relabel assemble --scores teacher/*/scores.jsonl --teacher teacher_judgments \
        --calibration calibration.json --out teacher_judgments_new

The card has the fields that openalex-affiliation-matcher v3.0.0 gave its teacher (jev_teacher.jev_card): the ROR
display name, up to 6 other names (labels, then aliases), up to 4 acronyms (sorted), the first type, and the city and
country of the first place. The answer to a question of type "noul" is a probability.

The folder questions/ holds two questions. v1.2.txt made the teacher_judgments config of the dataset. jev.txt is the
question of openalex-affiliation-matcher v3.0.0 (jev_teacher.NAMES_IT), for reference.

The steps use only the standard library.
"""
import argparse
import collections
import http.client
import json
import math
import os
import sys
import threading
import time
import urllib.parse

from ror_set_point.common.io import complete_rows, open_text, read_jsonl, sha256, write_jsonl, write_manifest
from ror_set_point.common.records import read_records

SPLITS = ("train", "validation")
KEY = "names_this_institution"  # the name of the question in the request and in the answer
RETRY_WAIT = 1.0  # seconds; attempt n (from 0) waits RETRY_WAIT * (1 + 2n) before the next attempt
EPS = 1e-6


def card(r):
    """The ROR record r as the teacher sees it."""
    other = [x for x in dict.fromkeys([x["v"] for x in r["labels"]] + r["aliases"]) if x != r["name"]]
    place = r["places"][0] if r["places"] else {}
    return {"name": r["name"], "other_names": other[:6], "acronyms": sorted(set(r["acronyms"]))[:4],
            "type": r["types"][0] if r["types"] else None, "city": place.get("city"), "country": place.get("country")}


def request_body(text, institution, question):
    # The format of TypeSafe's /v1/systemone API, as openalex-affiliation-matcher sent it. The StartLux server takes
    # this format and does not use "model".
    return {"model": "jev-1.13.0", "state": {"affiliation_string": text, "institution": institution},
            "questions": {KEY: {"type": "noul", "instructions": question}}}


def input_pairs(a):
    """(split, string, pairs) for each input row. A pair is (rid, ROR id, the other fields of its request line)."""
    if a.teacher:
        for split in SPLITS:
            for r in read_jsonl(os.path.join(a.teacher, f"{split}.jsonl")):
                yield split, r["text"], [(f"{split}|{r['id']}|{j['ror_id']}", j["ror_id"],
                                          {"split": split, "id": r["id"], "ror_id": j["ror_id"], "p_jev": j["p"]})
                                         for j in r["judgments"]]
    else:
        for r in read_jsonl(a.stage2):
            yield "stage2", r["s"], [(f"stage2|{r['k']}|{x}", x, {"id": r["k"], "ror_id": x, "y": y, "fold": r["fold"]})
                                     for x, y in zip(r["c"], r["y"])]


def make_requests(a):
    with open(a.question, encoding="utf-8") as f:
        question = f.read().strip()
    cards = {r["ror_id"]: card(r) for r in read_records(a.records)}
    counts = collections.Counter()

    def lines():
        for n, (split, text, pairs) in enumerate(input_pairs(a)):
            counts[f"rows_{split}"] += 1
            if n % a.shards != a.shard:
                continue
            counts[f"rows_in_shard_{split}"] += 1
            for rid, x, fields in pairs:
                if x not in cards:
                    raise SystemExit(f"{rid}: {x} is not in the records")
                counts["requests"] += 1
                yield {"rid": rid, **fields, "body": request_body(text, cards[x], question)}
    write_jsonl(a.out, lines())
    if a.verbose:
        print(json.dumps(dict(sorted(counts.items()), question_sha256=sha256(a.question)), indent=1))


def scored(paths):
    """The lines of score files that have a p. A file that does not exist has none. A line that a stop cut is
    skipped."""
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "p" in r:
                    yield r


def ask(conn, path, body):
    """{p, tokens} or {err} for one request. A connection error, HTTP 429, 502 or 503 or an answer without p is tried
    again, 6 times in total."""
    last = None
    for attempt in range(6):
        try:
            conn.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            status, data = resp.status, resp.read()
        except (OSError, http.client.HTTPException) as e:
            conn.close()  # the next request connects again
            last = f"{type(e).__name__}: {e}"
        else:
            if status == 200:
                try:
                    d = json.loads(data)
                    return {"p": d["answers"][KEY]["noul"], "tokens": d["usage"]["input_tokens"]}
                except (ValueError, KeyError, TypeError):
                    last = f"no answer: {data[:300].decode(errors='replace')}"
            elif status in (429, 502, 503):
                last = f"HTTP {status}"
            else:
                return {"err": f"HTTP {status}: {data[:300].decode(errors='replace')}"}
        time.sleep(RETRY_WAIT * (1 + 2 * attempt))
    return {"err": last}


def score(a):
    complete_rows(a.out)
    done = {r["rid"] for r in scored([a.out])}
    todo = [r for r in read_jsonl(a.requests) if r["rid"] not in done]
    print(f"{len(done)} pairs already have a p, {len(todo)} to send", file=sys.stderr, flush=True)
    url = urllib.parse.urlsplit(a.url)
    lock, stop = threading.Lock(), threading.Event()
    stats = collections.Counter()
    t0 = time.time()
    it = iter(todo)
    out = open(a.out, "a", encoding="utf-8")

    def worker():
        conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=600)
        while True:
            with lock:
                r = next(it, None)
            if r is None:
                break
            res = ask(conn, url.path, r["body"])
            line = json.dumps({**{k: v for k, v in r.items() if k != "body"}, **res}, ensure_ascii=False) + "\n"
            with lock:
                out.write(line)
                if "p" in res:
                    stats["scored"] += 1
                    stats["tokens"] += res["tokens"]
                    ref = r.get("p_jev", r.get("y"))
                    stats["agree"] += ref is not None and (res["p"] >= 0.5) == (ref >= 0.5)
                else:
                    stats["errors"] += 1
        conn.close()

    def progress():
        dt, n = time.time() - t0, stats["scored"]
        rate = n / dt if dt else 0
        left = len(todo) - n - stats["errors"]
        return {"scored": n, "errors": stats["errors"], "seconds": round(dt), "pairs_per_s": round(rate, 1),
                "prompt_tokens_per_s": round(stats["tokens"] / dt) if dt else 0,
                "tokens_per_pair": round(stats["tokens"] / n, 1) if n else 0,
                "agree_at_0.5": round(stats["agree"] / n, 4) if n else None,
                "eta_min": round(left / rate / 60) if rate else None}

    def sync():
        with lock:
            out.flush()
            os.fsync(out.fileno())

    def reporter():
        while not stop.wait(60):
            sync()
            print(json.dumps(progress()), file=sys.stderr, flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.concurrency)]
    rep = threading.Thread(target=reporter, daemon=True)
    for t in threads + [rep]:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    rep.join()
    sync()
    try:
        out.close()
    except OSError as e:  # a bucket mount can fail on close after a good fsync; the count below reads the file again
        print(f"warning: closing {a.out}: {e}", file=sys.stderr, flush=True)
    want = {r["rid"] for r in read_jsonl(a.requests)}
    missing = len(want - {r["rid"] for r in scored([a.out])})
    summary = dict(progress(), already_scored_before_this_run=len(done), pairs=len(want), pairs_without_p=missing)
    print(json.dumps(summary), flush=True)
    if a.summary:
        try:
            with open(a.summary, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=1)
        except OSError as e:
            print(f"warning: writing {a.summary}: {e}", file=sys.stderr, flush=True)
    if missing:
        raise SystemExit(f"{missing} of {len(want)} pairs have no p in {a.out}")


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def calibrated(p, a, b):
    return round(sigmoid(a * logit(p) + b), 4)


def platt(xs, ys, iters=30):
    """a, b that minimize the log loss of sigmoid(a * x + b) against the labels ys (Newton's method)."""
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, ys):
            s = sigmoid(a * x + b)
            w = s * (1 - s)
            ga += (s - y) * x
            gb += s - y
            haa += w * x * x
            hab += w * x
            hbb += w
        det = haa * hbb - hab * hab
        a -= (hbb * ga - hab * gb) / det
        b -= (haa * gb - hab * ga) / det
    return a, b


def log_loss(ps, ys):
    ps = [min(max(p, EPS), 1 - EPS) for p in ps]
    return -sum(math.log(p) if y else math.log(1 - p) for p, y in zip(ps, ys)) / len(ps)


def calibrate(a):
    pairs = {}
    for r in scored(a.scores):
        if "y" not in r:
            raise SystemExit(f"{r['rid']}: no label. Calibrate with the scores of --stage2 requests")
        pairs[r["rid"]] = (logit(r["p"]), int(r["y"] >= 0.5), r["fold"])
    rows = [pairs[k] for k in sorted(pairs)]
    fit = [(x, y) for x, y, fold in rows if fold in a.fit_folds]
    test = [(x, y) for x, y, fold in rows if fold not in a.fit_folds]
    if not fit or not test:
        raise SystemExit(f"--fit-folds {a.fit_folds} must leave pairs on both sides")
    ca, cb = platt(*zip(*fit))
    tx, ty = zip(*test)
    fa, fb = platt([x for x, _, _ in rows], [y for _, y, _ in rows])
    out = {"a": fa, "b": fb, "pairs": len(rows), "positives": sum(y for _, y, _ in rows),
           "check": {"fit_folds": sorted(a.fit_folds), "a": ca, "b": cb, "pairs_fit": len(fit), "pairs_test": len(test),
                     "log_loss_raw": log_loss([sigmoid(x) for x in tx], ty),
                     "log_loss_calibrated": log_loss([sigmoid(ca * x + cb) for x in tx], ty)}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    if a.verbose:
        print(json.dumps(out, indent=1))


def assemble(a):
    with open(a.calibration, encoding="utf-8") as f:
        cal = json.load(f)
    raw = {r["rid"]: r["p"] for r in scored(a.scores)}
    counts = collections.Counter(scored_pairs=len(raw))
    os.makedirs(a.out)

    def rows(split, side):
        for r in read_jsonl(os.path.join(a.teacher, f"{split}.jsonl")):
            js = []
            for j in r["judgments"]:
                key = f"{split}|{r['id']}|{j['ror_id']}"
                if key not in raw:
                    raise SystemExit(f"no p for {key} in --scores")
                p = calibrated(raw[key], cal["a"], cal["b"])
                js.append({"ror_id": j["ror_id"], "p": p})
                side.write(json.dumps({"split": split, "id": r["id"], "ror_id": j["ror_id"], "p_raw": raw[key],
                                       "p_jev": j["p"]}, ensure_ascii=False) + "\n")
                counts[f"pairs_{split}"] += 1
                counts[f"pairs_{split}_p_ge_0.5"] += p >= 0.5
                counts[f"pairs_{split}_p_jev_ge_0.5"] += j["p"] >= 0.5
            yield {"id": r["id"], "text": r["text"], "judgments": js}

    with open_text(os.path.join(a.out, "raw.jsonl.gz"), "w") as side:
        for split in SPLITS:
            counts[f"strings_{split}"] = write_jsonl(os.path.join(a.out, f"{split}.jsonl"), rows(split, side))
    write_manifest(a.out, calibration={"a": cal["a"], "b": cal["b"]},
                   inputs={f"teacher_{x}": sha256(os.path.join(a.teacher, f"{x}.jsonl")) for x in SPLITS},
                   counts=dict(sorted(counts.items())))
    if a.verbose:
        print(json.dumps(dict(sorted(counts.items())), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["requests", "score", "calibrate", "assemble"])
    ap.add_argument("--teacher", help="requests, assemble: a teacher_judgments folder (train.jsonl, validation.jsonl)")
    ap.add_argument("--stage2", help="requests: stage 2 rows {k, s, c, y, fold}")
    ap.add_argument("--records", help="requests: a records folder")
    ap.add_argument("--question", help="requests: a text file with the question")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--requests", help="score: the file of the requests mode")
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1/systemone")
    ap.add_argument("--concurrency", type=int, default=256, help="score: requests at a time")
    ap.add_argument("--summary", help="score: a JSON file for the counts and speeds")
    ap.add_argument("--scores", nargs="+", help="calibrate, assemble: the files of the score mode")
    ap.add_argument("--fit-folds", type=int, nargs="+", default=[0, 1, 2], help="calibrate: the folds of the check fit")
    ap.add_argument("--calibration", help="assemble: the file of the calibrate mode")
    ap.add_argument("--out", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    need = {"requests": ["records", "question"], "score": ["requests"], "calibrate": ["scores"],
            "assemble": ["scores", "teacher", "calibration"]}[a.mode]
    for x in need:
        if not getattr(a, x):
            ap.error(f"{a.mode} needs --{x}")
    if a.mode == "requests":
        if bool(a.teacher) == bool(a.stage2):
            ap.error("requests needs one of --teacher and --stage2")
        if not 0 <= a.shard < a.shards:
            ap.error("--shard must be from 0 to --shards - 1")
        make_requests(a)
    elif a.mode == "score":
        score(a)
    elif a.mode == "calibrate":
        calibrate(a)
    else:
        assemble(a)


if __name__ == "__main__":
    main()
