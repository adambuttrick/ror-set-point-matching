"""Run match on several GPUs: one process for each GPU, each on a part of the input.

    python jobs/shards.py --gpus 8 --input strings.jsonl --out answers.jsonl --work /tmp/work -- run --records ...

The input is cut into --gpus parts of consecutive rows (a .txt file becomes JSON Lines rows with the line number as
id, as match gives it). Process i runs `python -m ror_set_point.inference.match MODE --input part-i --out
OUT.parts/part-i.jsonl --resume ...` with CUDA_VISIBLE_DEVICES=i. When all parts are complete, they are joined into
--out in the order of the input, and OUT.parts is removed. When a process fails, the others go on; the next run with
the same --gpus continues each part. OUT.parts/plan.json holds the input, its rows and the number of parts: a run
with another plan stops, and parts with no plan (which a bucket can keep after their removal) are removed.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import threading


def lines(path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as f:
        yield from f


def cut(path, n, n_parts, work, text_field, id_field):
    """Write the parts of the input (n rows) into work and return their paths."""
    bounds = [n * i // n_parts for i in range(n_parts + 1)]
    paths = [os.path.join(work, f"part-{i:02d}.jsonl") for i in range(n_parts)]
    files = [open(p, "w", encoding="utf-8") for p in paths]
    part = 0
    for k, line in enumerate(lines(path)):
        while k >= bounds[part + 1]:
            part += 1
        if path.endswith(".txt"):
            line = json.dumps({id_field: str(k), text_field: line.rstrip("\n")}, ensure_ascii=False) + "\n"
        files[part].write(line)
    for f in files:
        f.close()
    print(f"{n} rows in {n_parts} parts: {[b - a for a, b in zip(bounds, bounds[1:])]}", file=sys.stderr)
    return paths


def relay(proc, tag):
    for line in proc.stdout:
        sys.stderr.write(f"[{tag}] {line}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True, help="a local folder for the parts of the input")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("match_args", nargs=argparse.REMAINDER, help="-- MODE and the other options of match")
    a = ap.parse_args()
    args = a.match_args[1:] if a.match_args[:1] == ["--"] else a.match_args
    fields = argparse.ArgumentParser(add_help=False)
    fields.add_argument("--text-field", default="text")
    fields.add_argument("--id-field", default="id")
    f, _ = fields.parse_known_args(args)
    if a.out.endswith(".gz"):
        raise SystemExit("--out must not be gzip (the parts use --resume)")
    n = sum(1 for _ in lines(a.input))
    parts_dir = a.out + ".parts"
    if os.path.exists(a.out):
        done = sum(1 for _ in lines(a.out))
        if done != n:
            raise SystemExit(f"{a.out} has {done} of {n} rows (from a run with one process?): continue it with one "
                             f"GPU, or remove it")
        print(f"{a.out} is complete: nothing to do", file=sys.stderr)
        shutil.rmtree(parts_dir, ignore_errors=True)
        return

    # A bucket can keep parts after they were removed, when the job ends at once. Parts with no plan are such parts,
    # or parts of a run that stopped before its plan: they are removed.
    os.makedirs(parts_dir, exist_ok=True)
    plan = os.path.join(parts_dir, "plan.json")
    want = {"input": os.path.abspath(a.input), "rows": n, "parts": a.gpus}
    if os.path.exists(plan):
        with open(plan, encoding="utf-8") as fp:
            if json.load(fp) != want:
                raise SystemExit(f"{parts_dir} was made for another input or number of parts: remove it, or use "
                                 f"the same --gpus")
    else:
        for x in os.listdir(parts_dir):
            os.remove(os.path.join(parts_dir, x))
        with open(plan, "w", encoding="utf-8") as fp:
            json.dump(want, fp)

    os.makedirs(a.work, exist_ok=True)
    inputs = cut(a.input, n, a.gpus, a.work, f.text_field, f.id_field)
    outs = [os.path.join(parts_dir, f"part-{i:02d}.jsonl") for i in range(a.gpus)]
    procs, threads = [], []
    for i, (inp, out) in enumerate(zip(inputs, outs)):
        cmd = [a.python, "-m", "ror_set_point.inference.match", *args, "--input", inp, "--out", out, "--resume"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t = threading.Thread(target=relay, args=(p, f"gpu {i}"), daemon=True)
        t.start()
        procs.append(p)
        threads.append(t)
    codes = [p.wait() for p in procs]
    for t in threads:
        t.join()
    if any(codes):
        raise SystemExit(f"match failed on GPUs {[i for i, c in enumerate(codes) if c]} (exit codes {codes})")

    with open(a.out + ".tmp", "wb") as w:
        for out in outs:
            with open(out, "rb") as r:
                shutil.copyfileobj(r, w, 1 << 20)
    os.replace(a.out + ".tmp", a.out)
    shutil.rmtree(parts_dir)
    print(f"{a.out}: {sum(1 for _ in lines(a.out))} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
