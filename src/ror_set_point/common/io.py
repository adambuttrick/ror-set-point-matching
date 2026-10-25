"""Read and write JSON Lines files, and record what a step wrote."""
import gzip
import hashlib
import io
import json
import os


def open_text(path, mode="r"):
    """Gzip output has no time stamp, so the same rows give the same bytes."""
    if not path.endswith(".gz"):
        return open(path, mode, encoding="utf-8")
    if mode == "r":
        return gzip.open(path, "rt", encoding="utf-8")
    return io.TextIOWrapper(gzip.GzipFile(path, mode + "b", mtime=0), encoding="utf-8")


def read_jsonl(path, limit=0):
    with open_text(path) as f:
        for n, line in enumerate(f):
            if limit and n >= limit:
                return
            yield json.loads(line)


def write_jsonl(path, rows):
    n = 0
    with open_text(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_manifest(out_dir, **fields):
    """Write out_dir/manifest.json: the given fields, and the sha256 of every other file in out_dir."""
    files = sorted(os.path.relpath(os.path.join(r, x), out_dir) for r, _, xs in os.walk(out_dir) for x in xs)
    fields["sha256"] = {x: sha256(os.path.join(out_dir, x)) for x in files if x != "manifest.json"}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=1, ensure_ascii=False)
    return fields


def check_manifest(data_dir, names):
    with open(os.path.join(data_dir, "manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    for x in names:
        if sha256(os.path.join(data_dir, x)) != m["sha256"][x]:
            raise SystemExit(f"{x}: the sha256 differs from the manifest")
    return m
