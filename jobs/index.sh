#!/usr/bin/env bash
# Step 2: encode the record texts with the retriever (the matcher's index). GPU.
#
#   RECORDS    the records folder (default /bucket/records/$ROR_VERSION)
#   RETRIEVER  a retriever folder or Hub repository (default adambuttrick/ror-set-point-retriever-e5-base)
#   VECTORS    the file to write (default /bucket/index/<records folder name>/<retriever name>.pt)
#
# The file keeps the retriever name and the sha256 of the records file. When they are the same, the file is read and
# not written again. match reads it with --record-vectors.
source "$(dirname "$0")/lib.sh"

describe_job
need "$RECORDS/manifest.json"
setup_env
mkdir -p "$(dirname "$VECTORS")"
run "$PY" -m ror_set_point.inference.retriever --model "$RETRIEVER" --records "$RECORDS" --record-vectors "$VECTORS"
log "index: $VECTORS ($(du -h "$VECTORS" | cut -f1))"
