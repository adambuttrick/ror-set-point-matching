#!/usr/bin/env bash
# Step 1: get a ROR release from Zenodo and make the records folder. CPU only.
#
#   ROR_VERSION   v2.14 (the release of the released models), another version such as v2.15, or latest
#   RECORDS       the records folder to write (default /bucket/records/$ROR_VERSION)
#   DOWNLOADS     where to keep the ROR zip and institutions.jsonl.gz (default /bucket/downloads)
#   CHECK         1: compare the folder with data/ror_records of the dataset (default 1 for v2.14, else 0)
#   FORCE         1: make the folder again when it exists
#
# The step is skipped when $RECORDS/manifest.json exists. That file is written last.
source "$(dirname "$0")/lib.sh"

INSTITUTIONS_URL=https://raw.githubusercontent.com/ourresearch/openalex-affiliation-matcher/v3.0.0/models/institutions.jsonl.gz
INSTITUTIONS_SHA256=7d2c5dc0da4ba13607f47d35c025d48cbe510d1c33c635e4ba69343a8ad9a57e
V214_SHA256=3b620ada133f533b7eddd01ded9d1a9dd647c7676d7c48b1c619b4bc52b7cf3e
CHECK=${CHECK:-$([ "$ROR_VERSION" = v2.14 ] && echo 1 || echo 0)}

describe_job
if [ -e "$RECORDS/manifest.json" ] && [ "${FORCE:-0}" != 1 ]; then
    log "$RECORDS exists: skip (FORCE=1 makes it again)"
    exit 0
fi

mkdir -p "$DOWNLOADS" "$(dirname "$INSTITUTIONS")"
run py "$CODE/jobs/fetch.py" url "$INSTITUTIONS_URL" "$INSTITUTIONS_SHA256" "$INSTITUTIONS"
dump=$(py "$CODE/jobs/fetch.py" ror "$ROR_VERSION" "$DOWNLOADS" "$WORK/ror")
log "ROR dump: $dump"
if [[ $(basename "$dump") == v2.14-* ]]; then
    [ "$(sha256sum "$dump" | cut -d' ' -f1)" = "$V214_SHA256" ] || die "$dump: the sha256 differs from the docs"
fi

rm -rf "$WORK/records"
run py -m ror_set_point.training.data.records --ror "$dump" --openalex-institutions "$INSTITUTIONS" \
    --out "$WORK/records" -v
if [ "$CHECK" = 1 ]; then
    run py "$CODE/jobs/check.py" records "$WORK/records" /dataset/data/ror_records
fi

# Copy to the bucket. manifest.json goes last, so that a folder with a manifest is complete.
mkdir -p "$RECORDS"
rm -f "$RECORDS/manifest.json"
cp "$WORK"/records/{records.jsonl.gz,withdrawn.json,openalex_to_ror.json} "$RECORDS"/
cp "$WORK/records/manifest.json" "$RECORDS"/
log "wrote $RECORDS: $(ls "$RECORDS" | tr '\n' ' ')"
