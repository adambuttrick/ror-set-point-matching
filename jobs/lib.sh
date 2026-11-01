# The common part of the job scripts. submit.sh mounts the code at /code (read only), the bucket at /bucket and the
# dataset at /dataset.
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_PROGRESS_BARS=1

CODE=${CODE:-/code}
WORK=${WORK:-/tmp/work}
mkdir -p "$WORK"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
run() { log "+ $*"; "$@"; }
die() { log "error: $*"; exit 1; }

# The vCPUs of the job, from its cgroup quota. nproc gives all the CPUs of the host (224 on an h200 job with 23).
cpus() {
    local q p
    if read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then
        echo $(( (q + p - 1) / p ))
    else
        nproc
    fi
}

gpus() {
    if command -v nvidia-smi > /dev/null; then nvidia-smi -L | grep -c '^GPU' || true; else echo 0; fi
}

# Python with the code alone. The steps for data and evaluation need only the standard library.
py() { PYTHONPATH="$CODE/src" python3 "$@"; }

# The project's environment (torch, and flash-attn on a GPU) in $WORK/code/.venv. After it, $PY is its Python.
setup_env() {
    if [ -n "${PY:-}" ]; then return; fi
    local extra=()
    if [ "$(gpus)" -gt 0 ]; then extra=(--extra gpu); fi
    rm -rf "$WORK/code"
    cp -r "$CODE" "$WORK/code"
    log "uv sync --locked --no-dev ${extra[*]}"
    (cd "$WORK/code" && uv sync --locked --no-dev "${extra[@]}" --quiet)
    PY=$WORK/code/.venv/bin/python
    export PY
    if [ "${#extra[@]}" -gt 0 ]; then
        # Without flash-attn the set model runs in padded batches, 5.6 times slower. Stop instead.
        "$PY" -c "import flash_attn" || die "flash-attn does not import"
    fi
    "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
}

describe_job() {
    log "$(cpus) vCPU, $(gpus) GPU, ROR $ROR_VERSION, records $RECORDS"
    if [ "$(gpus)" -gt 0 ]; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2; fi
}

need() {
    local x
    for x in "$@"; do [ -e "$x" ] || die "$x does not exist"; done
}

# Defaults that the steps share. A job can change each with -e NAME=value.
ROR_VERSION=${ROR_VERSION:-v2.14}
if [ "$ROR_VERSION" = latest ]; then
    ROR_VERSION=$(py "$CODE/jobs/fetch.py" version latest)
fi
RECORDS=${RECORDS:-/bucket/records/$ROR_VERSION}
DOWNLOADS=${DOWNLOADS:-/bucket/downloads}
INSTITUTIONS=${INSTITUTIONS:-$DOWNLOADS/institutions.jsonl.gz}
RETRIEVER=${RETRIEVER:-adambuttrick/ror-set-point-retriever-e5-base}
SET_MODEL=${SET_MODEL:-adambuttrick/ror-set-point-set-model-mmbert-base-mask}
VECTORS=${VECTORS:-/bucket/index/$(basename "$RECORDS")/${RETRIEVER##*/}.pt}
export ROR_VERSION RECORDS DOWNLOADS INSTITUTIONS RETRIEVER SET_MODEL VECTORS
