#!/usr/bin/env bash
# Step 3: match a file of strings. GPU. With several GPUs, one match process runs on each GPU (jobs/shards.py).
#
#   INPUT        JSON Lines (.jsonl or .jsonl.gz, fields id and text) or .txt (one string on each line), for
#                example /dataset/data/affiliations/test.jsonl or a file in the bucket
#   OUT          the answers (JSON Lines, not gzip), for example /bucket/runs/<name>/answers.jsonl
#   MODE         run (default), candidates or answer (INPUT then holds candidate rows)
#   WORKERS      processes for the lexical index for each GPU (default: the vCPUs for each GPU - 1)
#   MATCH_ARGS   more options of match, for example "--text-field affiliation --chunk-size 50000"
#   RECORDS, RETRIEVER, SET_MODEL, VECTORS as in lib.sh
#
# The run always uses --resume: when a job stops, the same job again (or hf jobs run --attempts) continues OUT.
source "$(dirname "$0")/lib.sh"

: "${INPUT:?set INPUT}" "${OUT:?set OUT}"
MODE=${MODE:-run}
[[ $OUT != *.gz ]] || die "OUT must not be gzip (match --resume)"
n_gpus=$(gpus)
[ "$n_gpus" -gt 0 ] || log "warning: no GPU; the set model runs on the CPU, about 1 second for each string"
per_gpu=$(( $(cpus) / (n_gpus > 0 ? n_gpus : 1) ))
WORKERS=${WORKERS:-$(( per_gpu > 1 ? per_gpu - 1 : 1 ))}

describe_job
need "$INPUT" "$RECORDS/manifest.json"
args=("$MODE" --records "$RECORDS" --workers "$WORKERS")
if [ "$MODE" != answer ]; then
    args+=(--retriever "$RETRIEVER" --record-vectors "$VECTORS")
fi
if [ "$MODE" != candidates ]; then args+=(--set-model "$SET_MODEL"); fi
read -r -a extra <<< "${MATCH_ARGS:-}"
args+=("${extra[@]}")

setup_env
mkdir -p "$(dirname "$OUT")"
start=$SECONDS
if [ "$n_gpus" -gt 1 ]; then
    if [ "$MODE" != answer ]; then
        # Without a valid index, each process would encode the records and write the same file. Check it once first.
        mkdir -p "$(dirname "$VECTORS")"
        run "$PY" -m ror_set_point.inference.retriever --model "$RETRIEVER" --records "$RECORDS" \
            --record-vectors "$VECTORS"
    fi
    run "$PY" "$CODE/jobs/shards.py" --gpus "$n_gpus" --input "$INPUT" --out "$OUT" --work "$WORK/parts" \
        --python "$PY" -- "${args[@]}"
else
    mkdir -p "$(dirname "$VECTORS")"
    run "$PY" -m ror_set_point.inference.match "${args[@]}" --input "$INPUT" --out "$OUT" --resume
fi
log "$OUT: $(wc -l < "$OUT") rows, $(( SECONDS - start )) s with loading"
