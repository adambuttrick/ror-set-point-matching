#!/usr/bin/env bash
# Step 4: score answers against a labeled split. CPU only.
#
#   GOLD        a labeled split, for example /dataset/data/affiliations/test.jsonl
#   ANSWERS     the answers of match, for example /bucket/runs/<name>/affiliations_test.jsonl
#   REFERENCE   optional: other answers of the same strings (for example the published ones). The scores then
#               compare the two (--paired)
#   SCORE_ARGS  more options of ror_set_point.evaluation.score, for example "--bootstrap 2000"
#   SCORES      the Markdown output (default: ANSWERS without .jsonl, then .scores.md)
#   RECORDS     the records folder that resolves the labels (default /bucket/records/$ROR_VERSION)
source "$(dirname "$0")/lib.sh"

: "${GOLD:?set GOLD}" "${ANSWERS:?set ANSWERS}"
SCORES=${SCORES:-${ANSWERS%.jsonl}.scores.md}
need "$GOLD" "$ANSWERS" "$RECORDS/manifest.json"

args=(--gold "$GOLD" --records "$RECORDS" --system "this run" "$ANSWERS")
if [ -n "${REFERENCE:-}" ]; then
    need "$REFERENCE"
    args+=(--system reference "$REFERENCE" --paired)
fi
read -r -a extra <<< "${SCORE_ARGS:-}"
run py -m ror_set_point.evaluation.score "${args[@]}" "${extra[@]}" > "$SCORES.tmp"
mv "$SCORES.tmp" "$SCORES"
cat "$SCORES"
log "wrote $SCORES"
