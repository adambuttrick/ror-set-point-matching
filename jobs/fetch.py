"""Download the inputs of the records folder, and check them.

    python jobs/fetch.py ror v2.14 zip_dir [json_dir]       # or "latest"
    python jobs/fetch.py version latest                     # prints the version, for example v2.15
    python jobs/fetch.py url URL SHA256 out_file

ror finds the ROR data dump on Zenodo (the "ROR Data" record, all versions), downloads its zip into zip_dir, checks
the md5 that Zenodo gives, and extracts the JSON file (schema v2) into json_dir. It prints the path of the JSON file.
A zip that is already in zip_dir with the right md5 is not downloaded again. url downloads a file and checks its
sha256. Downloads that fail are tried again. Only the standard library is used.
"""
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid
import zipfile

ZENODO = "https://zenodo.org/api/records"
ROR_CONCEPT = "6347574"  # the concept record of the ROR data dumps; each release is a version of it


def get(url, tries=6):
    for n in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except OSError as e:
            if n == tries - 1:
                raise
            print(f"{url}: {e}; again in {2 ** n} s", file=sys.stderr)
            time.sleep(2 ** n)


def download(url, out, digest, tries=6):
    """Download url to out and return its hexdigest of the hashlib name digest. The data goes to a part file with
    a random name first, so that two jobs can download the same file at the same time."""
    part = f"{out}.{uuid.uuid4().hex[:8]}.part"
    for n in range(tries):
        try:
            h = hashlib.new(digest)
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                while b := r.read(1 << 20):
                    f.write(b)
                    h.update(b)
            os.replace(part, out)
            return h.hexdigest()
        except OSError as e:
            if n == tries - 1:
                raise
            print(f"{url}: {e}; again in {2 ** n} s", file=sys.stderr)
            time.sleep(2 ** n)


def file_digest(path, digest):
    h = hashlib.new(digest)
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def ror_release(version):
    """The Zenodo record of a ROR release: "latest" or a version such as "v2.14"."""
    if version == "latest":
        return json.loads(get(f"{ZENODO}/{ROR_CONCEPT}/versions/latest"))
    q = urllib.parse.quote(f'conceptrecid:{ROR_CONCEPT} AND metadata.version:"{version}"')
    hits = json.loads(get(f"{ZENODO}?q={q}&all_versions=true"))["hits"]["hits"]
    hits = [h for h in hits if h["metadata"]["version"] == version]
    if len(hits) != 1:
        raise SystemExit(f"ROR release {version}: {len(hits)} Zenodo records")
    return hits[0]


def ror(version, out_dir, json_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    rec = ror_release(version)
    zips = [f for f in rec["files"] if f["key"].endswith("-ror-data.zip")]
    if len(zips) != 1:
        raise SystemExit(f"Zenodo record {rec['id']}: no single *-ror-data.zip")
    z = zips[0]
    algo, want = z["checksum"].split(":")
    path = os.path.join(out_dir, z["key"])
    print(f"ROR {rec['metadata']['version']} ({rec['metadata']['publication_date']}): Zenodo record {rec['id']}, "
          f"{z['key']}, {z['size']} bytes", file=sys.stderr)
    if not (os.path.exists(path) and file_digest(path, algo) == want):
        got = download(z["links"]["self"], path, algo)
        if got != want:
            raise SystemExit(f"{z['key']}: {algo} {got}, Zenodo gives {want}")
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith("-ror-data.json")]
        if len(names) != 1:
            raise SystemExit(f"{z['key']}: no single *-ror-data.json in {zf.namelist()}")
        json_path = zf.extract(names[0], json_dir or out_dir)
    print(json_path)


def url(link, want, out):
    if os.path.exists(out) and file_digest(out, "sha256") == want:
        return
    got = download(link, out, "sha256")
    if got != want:
        raise SystemExit(f"{link}: sha256 {got}, expected {want}")


def version(name):
    print(ror_release(name)["metadata"]["version"])


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"ror": ror, "version": version, "url": url}[cmd](*args)
