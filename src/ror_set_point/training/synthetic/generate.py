"""Make synthetic affiliation strings in non-Latin scripts from ROR records, with a language model through the codex
CLI. Three modes:

  batches   the records whose first place is in a country of the fixed list, in sha256 order of the ROR id, 40 to a
            batch: --dir/batches/b0000.json, ...
  run       one isolated `codex exec` call for each batch, --workers at a time. The call has no tools (no shell, no
            web search, no skills) and an output schema. A batch with a valid result is not run again, so a stopped
            run continues where it stopped.
  collect   every valid result as rows in the synthetic_non_latin format: --dir/synthetic.jsonl {id, text, ror_ids,
            ror_id, form, confidence, language}. ror_ids is the source record, plus each related record that the
            model says the string names.

    python -m ror_set_point.training.synthetic.generate batches --records records --dir synthetic
    python -m ror_set_point.training.synthetic.generate run --records records --dir synthetic --workers 8
    python -m ror_set_point.training.synthetic.generate collect --records records --dir synthetic
"""
import argparse
import collections
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ror_set_point.common.records import read_records


def languages():
    """{country code: the language and script to write in}."""
    return {
        "CN": "Chinese (Simplified)", "TW": "Chinese (Traditional)", "HK": "Chinese (Traditional)",
        "MO": "Chinese (Traditional)", "JP": "Japanese", "KR": "Korean (Hangul)", "KP": "Korean (Hangul)",
        "RU": "Russian", "UA": "Ukrainian (Russian also usual)", "BY": "Russian or Belarusian",
        "KZ": "Russian or Kazakh (Cyrillic)", "KG": "Russian or Kyrgyz (Cyrillic)", "TJ": "Tajik or Russian (Cyrillic)",
        "MN": "Mongolian (Cyrillic)", "BG": "Bulgarian", "RS": "Serbian (Cyrillic)", "MK": "Macedonian",
        "IR": "Persian", "AF": "Persian (Dari) or Pashto", "EG": "Arabic", "SA": "Arabic", "AE": "Arabic",
        "IQ": "Arabic", "JO": "Arabic", "LB": "Arabic", "SY": "Arabic", "PS": "Arabic", "KW": "Arabic", "QA": "Arabic",
        "BH": "Arabic", "OM": "Arabic", "YE": "Arabic", "LY": "Arabic", "SD": "Arabic", "DZ": "Arabic",
        "MA": "Arabic", "TN": "Arabic", "MR": "Arabic", "TH": "Thai", "LA": "Lao", "KH": "Khmer", "MM": "Burmese",
        "GR": "Greek", "CY": "Greek", "IL": "Hebrew", "GE": "Georgian", "AM": "Armenian", "ET": "Amharic",
        "BD": "Bengali", "NP": "Nepali", "LK": "Sinhala or Tamil"}


def prompt(cards):
    return f"""You write realistic research-paper affiliation strings for organisations, in the language and script that
researchers in the organisation's country use. The strings train a search model that must find these organisations
from affiliations written in local scripts, so they must look like real affiliations from local-language papers.

Below are {len(cards)} organisation cards (JSON). Each has a key, its registered names (English display name, acronyms, names
in other languages, aliases), its place, its types, the language to write in ("write_in"), and its related
organisations, each with a key ("r1", "r2", ...), a relation (parent, child, related, predecessor, successor) and a name.

For each card, write 4 strings, one of each form:
  "full"   the organisation's full local name, usually with its city or address, as it would appear in an affiliation;
  "unit"   a department, faculty, school, institute section or laboratory inside it, then the organisation, as
           researchers write it (an ordinary internal unit, never another registered organisation);
  "short"  a short form: a usual local abbreviation or shortened name, with whatever usually goes with it;
  "mixed"  a mixed or transliterated form, for example a local-script name with an English unit, or a romanised name
           with local-script parts, as such affiliations really occur.
Add a fifth, form "other_script", only when a second script or language is really usual for affiliations in that
country (a Russian form for a Kazakh organisation, for example).

Rules:
- Use the established local name where one exists; the cards' names in other languages are reliable. If you do not
  know an established local name and must translate, translate faithfully and set confidence "low"; otherwise "high".
- Vary the wording and structure across cards, as real affiliations vary (word order, punctuation, postal codes,
  city or province, with or without country).
- A string may name a related organisation from the card's list only where affiliations really do so (for example an
  institute of an academy of sciences written with the academy's name). List the keys of every related organisation a
  string names in "names_related". Never name any other organisation, and never invent a person's name or an email.
- Do not copy the English display name unchanged as a whole string.

Return JSON matching the schema: {{"cards": [{{"key": ..., "strings": [{{"text": ..., "form": ..., "names_related": [...],
"confidence": "high" | "low"}}]}}]}}, with every card key exactly once.

Cards:
{json.dumps(cards, ensure_ascii=False, indent=1)}
"""


def schema():
    forms = ["full", "unit", "short", "mixed", "other_script"]
    s = {"type": "object", "additionalProperties": False, "required": ["text", "form", "names_related", "confidence"],
         "properties": {"text": {"type": "string"}, "form": {"type": "string", "enum": forms},
                        "names_related": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string", "enum": ["high", "low"]}}}
    c = {"type": "object", "additionalProperties": False, "required": ["key", "strings"],
         "properties": {"key": {"type": "string"}, "strings": {"type": "array", "items": s}}}
    return {"type": "object", "additionalProperties": False, "required": ["cards"],
            "properties": {"cards": {"type": "array", "items": c}}}


def valid(out, batch):
    """Every card once, 4 or 5 strings with the 4 required forms, no empty text, only related keys of the card."""
    keys = {c["key"]: c for c in batch["cards"]}
    got = {c["key"]: c for c in out["cards"]}
    if set(got) != set(keys):
        return False
    for k, c in got.items():
        rel = {r["key"] for r in keys[k]["related"]}
        forms = [s["form"] for s in c["strings"]]
        if not 4 <= len(forms) <= 5 or not {"full", "unit", "short", "mixed"} <= set(forms):
            return False
        if any(not s["text"].strip() or set(s["names_related"]) - rel for s in c["strings"]):
            return False
    return True


def make_batches(a, bdir):
    os.makedirs(bdir)
    lang = languages()
    rows = []
    for r in read_records(a.records):
        cc = (r["places"] or [{}])[0].get("cc")
        if cc in lang:
            rows.append((hashlib.sha256(r["ror_id"].encode()).hexdigest(), cc, r))
    rows.sort(key=lambda x: x[0])
    for b in range(0, len(rows), a.per_batch):
        out = []
        for n, (_, cc, r) in enumerate(rows[b:b + a.per_batch], 1):
            rel = [{"key": f"r{j}", "relation": t, "name": x["name"], "id": x["id"]}
                   for j, (t, x) in enumerate(((t, x) for t in ("parent", "child", "related", "predecessor", "successor")
                                               for x in r["relations"][t][:8]), 1)]
            place = (r["places"] or [{}])[0]
            out.append({"key": f"c{n}", "ror": r["ror_id"], "name": r["name"], "acronyms": r["acronyms"],
                        "other_names": [f"{x['v']} ({x['lang']})" if x["lang"] else x["v"] for x in r["labels"]]
                        + r["aliases"],
                        "place": ", ".join(v for v in (place.get("city"), place.get("region"), place.get("country")) if v),
                        "types": r["types"], "status": r["status"], "write_in": lang[cc], "related": rel})
        with open(os.path.join(bdir, f"b{b // a.per_batch:04d}.json"), "x", encoding="utf-8") as f:
            json.dump({"cards": out}, f, ensure_ascii=False)
    if a.verbose:
        print(f"{len(rows):,} records in {-(-len(rows) // a.per_batch):,} batches; by country: "
              f"{dict(collections.Counter(cc for _, cc, _ in rows).most_common())}")


def run_batches(a, bdir, names):
    for sub in ("prompts", "results", "logs", "work"):
        os.makedirs(os.path.join(a.dir, sub), exist_ok=True)
    schema_path = os.path.join(a.dir, "schema.json")
    if not os.path.exists(schema_path):
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema(), f)
    lock = threading.Lock()

    def one(name):
        with open(os.path.join(bdir, f"{name}.json"), encoding="utf-8") as f:
            batch = json.load(f)
        result = os.path.join(a.dir, "results", f"{name}.json")
        if os.path.exists(result):
            return name, "cached"
        view = [{k: v for k, v in c.items() if k != "ror"} | {"related": [{k: v for k, v in r.items() if k != "id"}
                                                                          for r in c["related"]]}
                for c in batch["cards"]]
        p = os.path.join(a.dir, "prompts", f"{name}.txt")
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write(prompt(view))
        for attempt in range(1, a.retries + 1):
            tmp = os.path.join(a.dir, "logs", f"{name}.attempt{attempt}.result.json")
            # No tools: no shell (login shells and shell snapshots read the user's profile), no skills, and no web
            # search (on by default even without --search). The prompt holds everything the model needs.
            cmd = [a.codex, "--no-daemon", "-a", "never", "exec", "--ignore-user-config",
                   "--disable", "plugins", "--disable", "apps", "--disable", "multi_agent",
                   "--disable", "shell_tool", "--disable", "shell_snapshot", "--disable", "skill_search",
                   "-c", 'web_search="disabled"', "-m", a.model, "-c", f'model_reasoning_effort="{a.effort}"',
                   "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "--json",
                   "--output-schema", schema_path, "-o", tmp, "-"]
            started = time.time()
            stem = os.path.join(a.dir, "logs", f"{name}.attempt{attempt}")
            with open(p, "rb") as src, open(f"{stem}.jsonl", "wb") as ev, open(f"{stem}.stderr.log", "wb") as err:
                proc = subprocess.Popen(cmd, cwd=os.path.join(a.dir, "work"), stdin=src, stdout=ev, stderr=err,
                                        start_new_session=True)
                try:
                    proc.wait(timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait()
            ok = proc.returncode == 0 and os.path.exists(tmp)
            if ok:
                try:
                    with open(tmp, encoding="utf-8") as f:
                        out = json.load(f)
                    ok = valid(out, batch)
                except (json.JSONDecodeError, KeyError, TypeError):
                    ok = False
            if a.verbose:
                with lock:
                    print(f"{name} attempt {attempt}: {'ok' if ok else 'failed'} ({time.time() - started:.0f}s)",
                          flush=True)
            if ok:
                with open(result, "x", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False)
                return name, "ok"
            time.sleep(a.rest)
        return name, "failed"

    with ThreadPoolExecutor(a.workers) as pool:
        done = list(pool.map(one, names))
    failed = [n for n, s in done if s == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} of {len(names)} batches have no valid result: {failed}")


def collect(a, bdir, names):
    counts = collections.Counter()
    with open(os.path.join(a.dir, "synthetic.jsonl"), "x", encoding="utf-8") as f:
        for name in names:
            rp = os.path.join(a.dir, "results", f"{name}.json")
            if not os.path.exists(rp):
                counts["batches_missing"] += 1
                continue
            with open(os.path.join(bdir, f"{name}.json"), encoding="utf-8") as fb:
                batch = {c["key"]: c for c in json.load(fb)["cards"]}
            with open(rp, encoding="utf-8") as fr:
                result = json.load(fr)
            for c in result["cards"]:
                card = batch[c["key"]]
                rel = {r["key"]: r["id"] for r in card["related"]}
                for i, s in enumerate(c["strings"]):
                    ids = [card["ror"]] + [rel[k] for k in dict.fromkeys(s["names_related"])]
                    counts["strings"] += 1
                    counts[f"form_{s['form']}"] += 1
                    counts[f"confidence_{s['confidence']}"] += 1
                    f.write(json.dumps({"id": f"{name}:{c['key']}:{i}", "text": s["text"].strip(), "ror_ids": ids,
                                        "ror_id": card["ror"], "form": s["form"], "confidence": s["confidence"],
                                        "language": card["write_in"]}, ensure_ascii=False) + "\n")
    if counts["batches_missing"]:
        print(f"{counts['batches_missing']} batches have no result", flush=True)
    if a.verbose:
        print(json.dumps(dict(sorted(counts.items())), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["batches", "run", "collect"])
    ap.add_argument("--records", required=True, help="a records folder")
    ap.add_argument("--dir", required=True, help="the working folder: batches/, prompts/, results/, logs/")
    ap.add_argument("--per-batch", type=int, default=40)
    ap.add_argument("--codex", default="codex", help="the codex executable")
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900, help="seconds for one call")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--rest", type=float, default=20, help="seconds between attempts")
    ap.add_argument("--limit", type=int, default=0, help="run: only the first N batches")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    a.dir = os.path.abspath(a.dir)  # codex runs in --dir/work
    bdir = os.path.join(a.dir, "batches")
    if a.mode == "batches":
        make_batches(a, bdir)
        return
    names = sorted(x[:-5] for x in os.listdir(bdir) if x.endswith(".json"))
    if a.mode == "run":
        run_batches(a, bdir, names[:a.limit] if a.limit else names)
    else:
        collect(a, bdir, names)


if __name__ == "__main__":
    main()
