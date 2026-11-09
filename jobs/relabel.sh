#!/usr/bin/env bash
# Ask the teacher model the question of QUESTION about each pair of one shard (ror_set_point.training.data.relabel).
# GPU: one H200 for each shard. The teacher is StartLux-Decision v1.1 27B Instant, served by vLLM in a venv of its own.
# The relabel steps need only the standard library. One job for each shard:
#
#   SHARDS=24 jobs/submit.sh relabel --flavor h200 --timeout 4h --detach
#   SHARDS=8 jobs/submit.sh relabel --flavor h200 --timeout 2h --detach \
#       -e STAGE2=/set_model/training_data/stage2.jsonl.gz \
#       -v hf://adambuttrick/ror-set-point-set-model-mmbert-base-openalex-jev-distill:/set_model
#
#   SHARD, SHARDS  the shard of the job, from 0 to SHARDS - 1 (submit.sh sets them when SHARDS is set)
#   TEACHER        a teacher_judgments folder: the pairs are its judgments (default /dataset/data/teacher_judgments)
#   STAGE2         stage 2 rows: the pairs are their candidates, with labels and folds, in place of TEACHER. These
#                  pairs are for the calibrate step
#   OUT            the folder of the run (default /bucket/runs/relabel/teacher, or /bucket/runs/relabel/stage2 with
#                  STAGE2). The job writes OUT/shard-II-of-NN/: requests.json (counts), scores.jsonl, summary.json and
#                  logs/. A job of the same shard continues there
#   QUESTION       the question file (default: questions/v1.2.txt of the code)
#   CONCURRENCY    requests at a time (default 256)
#
# The model revision, vLLM and startlux-decide are pinned. The job stops when the server does not use the prompt
# attention of the startlux-decide plugin. When pairs have no p after the score step, the step runs again (3 attempts).
source "$(dirname "$0")/lib.sh"
here=$(dirname "$0")

MODEL_REPO=startlux-models/StartLux-Decision-v1.1-27B-Instant
MODEL_REVISION=32e7d35d74db615c4a059181efbf21c5421f3561
VLLM_VERSION=0.31.0
STARTLUX_DECIDE=a99977d9332decc9fb27a2488b4a83b533b6cd5f

SHARD=${SHARD:?set SHARD}
SHARDS=${SHARDS:?set SHARDS}
TEACHER=${TEACHER:-/dataset/data/teacher_judgments}
STAGE2=${STAGE2:-}
OUT=${OUT:-/bucket/runs/relabel/$([ -n "$STAGE2" ] && echo stage2 || echo teacher)}
QUESTION=${QUESTION:-$CODE/src/ror_set_point/training/data/questions/v1.2.txt}
CONCURRENCY=${CONCURRENCY:-256}
PART=$OUT/shard-$(printf %02d "$SHARD")-of-$(printf %02d "$SHARDS")
if [ -n "$STAGE2" ]; then input=(--stage2 "$STAGE2"); else input=(--teacher "$TEACHER"); fi
need "${input[1]}" "$QUESTION"

describe_job
log "shard $SHARD of $SHARDS -> $PART"
mkdir -p "$PART/logs"
save_logs() {
    cp "$WORK/serve.log" "$PART/logs/" 2>/dev/null || true
    cp -r "$WORK/vllm_logs" "$PART/logs/" 2>/dev/null || true
}
trap save_logs EXIT
bash "$here/ror_data.sh"

log "requests"
run py -m ror_set_point.training.data.relabel requests "${input[@]}" --records "$RECORDS" --question "$QUESTION" \
    --shard "$SHARD" --shards "$SHARDS" --out "$WORK/requests.jsonl.gz" -v > "$WORK/requests.json"
cp "$WORK/requests.json" "$PART/requests.json"
cat "$WORK/requests.json" >&2

log "vLLM $VLLM_VERSION, startlux-decide ${STARTLUX_DECIDE:0:7}"
uv venv -q -p 3.12 "$WORK/vllm"
VIRTUAL_ENV=$WORK/vllm uv pip install -q "vllm==$VLLM_VERSION" \
    "startlux-decide @ https://github.com/StartLuxLabs/StartLux-Decision/archive/$STARTLUX_DECIDE.tar.gz"
"$WORK/vllm/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" >&2

log "model $MODEL_REPO at ${MODEL_REVISION:0:7}"
"$WORK/vllm/bin/python" -c "from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='$WORK/model')"

log "serve"
PATH=$WORK/vllm/bin:$PATH "$WORK/vllm/bin/startlux-decide" serve --model "$WORK/model" --launch-vllm --gpus 0 \
    --port 8090 --no-images --max-model-len 16384 --vllm-log-dir "$WORK/vllm_logs" > "$WORK/serve.log" 2>&1 &
serve_pid=$!
up() {
    python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=5)" 2>/dev/null
}
for _ in $(seq 240); do
    up && break
    if ! kill -0 "$serve_pid" 2>/dev/null; then
        tail -n 80 "$WORK/serve.log" >&2
        tail -n 80 "$WORK"/vllm_logs/* >&2 || true
        die "the server stopped"
    fi
    sleep 10
done
up || die "the server did not start in 40 min"
grep -qh "prefix-bidirectional prompt attention ON" "$WORK"/vllm_logs/* || die "the startlux-decide plugin is not on"

for attempt in 1 2 3; do
    log "score (attempt $attempt)"
    if py -m ror_set_point.training.data.relabel score --requests "$WORK/requests.jsonl.gz" \
        --out "$PART/scores.jsonl" --concurrency "$CONCURRENCY" --summary "$PART/summary.json"; then
        log "done: each pair of the shard has a p"
        exit 0
    fi
done
die "pairs without a p after 3 attempts (see $PART/summary.json)"
