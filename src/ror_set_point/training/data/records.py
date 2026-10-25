"""Make a records folder from a ROR data dump: one row per active or inactive record, with a text form for the
models, and the maps that resolve withdrawn records and OpenAlex institution ids.

    python -m ror_set_point.training.data.records --ror v2.14-2026-10-06-ror-data.json \
        --openalex-institutions institutions.jsonl.gz --out records

Writes into --out (which must not exist):
  records.jsonl.gz      {ror_id, status, name, acronyms, labels [{v, lang}], aliases, places [{city, region, country,
                        cc}], types, relations {parent, child, related, predecessor, successor: [{id, name}]}, text}
  withdrawn.json        {withdrawn ROR id: successor ROR id or null}
  openalex_to_ror.json  {OpenAlex institution id: ROR id as the OpenAlex institution gives it}
  manifest.json         inputs, counts and sha256 of each file
"""
import argparse
import collections
import json
import os

from ror_set_point.common.io import read_jsonl, sha256, write_jsonl, write_manifest


def record_row(r, name_of, cap):
    """The row of one ROR v2 record. text = name | acronyms | other names | places | types | status | relations,
    with at most cap names in each relation list."""
    kinds = ("parent", "child", "related", "predecessor", "successor")
    names = r["names"]
    display = next(n["value"] for n in names if "ror_display" in n["types"])
    acronyms = [n["value"] for n in names if "acronym" in n["types"]]
    labels = [{"v": n["value"], "lang": n["lang"]} for n in names if "label" in n["types"] and n["value"] != display]
    aliases = [n["value"] for n in names if "alias" in n["types"]]
    places = []
    for loc in r["locations"]:
        g = loc.get("geonames_details") or {}
        places.append({"city": g.get("name"), "region": g.get("country_subdivision_name"),
                       "country": g.get("country_name"), "cc": g.get("country_code")})
    rel = {k: [] for k in kinds}
    for x in r["relationships"]:
        rel[x["type"]].append({"id": x["id"], "name": name_of.get(x["id"], x["label"])})
    other = [f"{x['v']} ({x['lang']})" if x["lang"] else x["v"] for x in labels] + aliases
    parts = [display, "; ".join(acronyms), "; ".join(other),
             "; ".join(", ".join(v for v in (p["city"], p["region"], p["country"]) if v) for p in places),
             ", ".join(r["types"]), r["status"]]
    for k in kinds:
        if rel[k]:
            more = f" (+{len(rel[k]) - cap} more)" if len(rel[k]) > cap else ""
            parts.append(f"{k}: {'; '.join(x['name'] for x in rel[k][:cap])}{more}")
    return {"ror_id": r["id"], "status": r["status"], "name": display, "acronyms": acronyms, "labels": labels,
            "aliases": aliases, "places": places, "types": r["types"], "relations": rel,
            "text": " | ".join(p for p in parts if p)}


def successors(recs, counts):
    """{withdrawn id: successor}. A withdrawn record goes to its successor when it has exactly one successor link.
    The chain continues through withdrawn successors and must end at an active or inactive record."""
    by = {r["id"]: r for r in recs}
    out = {}
    for r in recs:
        if r["status"] != "withdrawn":
            continue
        x, hops = r, 0
        while x["status"] == "withdrawn" and hops < 10:
            nxt = [y["id"] for y in x["relationships"] if y["type"] == "successor" and y["id"] in by]
            if len(nxt) != 1:
                x = None
                break
            x, hops = by[nxt[0]], hops + 1
        end = x if x is not None and x["status"] != "withdrawn" else None
        out[r["id"]] = end["id"] if end else None
        counts["withdrawn_to_" + (end["status"] if end else "none")] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ror", required=True, help="a ROR data dump (JSON, schema v2)")
    ap.add_argument("--openalex-institutions", required=True,
                    help="institutions.jsonl.gz of openalex-affiliation-matcher (fields id, ror, status)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--relation-cap", type=int, default=8)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out)
    with open(a.ror, encoding="utf-8") as f:
        recs = json.load(f)
    name_of = {r["id"]: next(n["value"] for n in r["names"] if "ror_display" in n["types"]) for r in recs}
    counts = collections.Counter(f"ror_{r['status']}" for r in recs)
    counts["records"] = write_jsonl(os.path.join(a.out, "records.jsonl.gz"),
                                    (record_row(r, name_of, a.relation_cap) for r in recs
                                     if r["status"] in ("active", "inactive")))
    with open(os.path.join(a.out, "withdrawn.json"), "w", encoding="utf-8") as f:
        json.dump(successors(recs, counts), f, indent=0, sort_keys=True)
    openalex = {}
    for c in read_jsonl(a.openalex_institutions):
        openalex[c["id"]] = c["ror"]
    counts["openalex_institutions"] = len(openalex)
    counts["openalex_institutions_without_ror"] = sum(1 for x in openalex.values() if not x)
    with open(os.path.join(a.out, "openalex_to_ror.json"), "w", encoding="utf-8") as f:
        json.dump(openalex, f)
    write_manifest(a.out, inputs={"ror": [os.path.basename(a.ror), sha256(a.ror)],
                                  "openalex_institutions": sha256(a.openalex_institutions)},
                   counts=dict(sorted(counts.items())))
    if a.verbose:
        print(json.dumps(dict(sorted(counts.items())), indent=1))


if __name__ == "__main__":
    main()
