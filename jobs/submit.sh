#!/usr/bin/env bash
# Run a step of the pipeline as a Hugging Face Job.
#
#   jobs/submit.sh STEP [OPTIONS OF hf jobs run]
#
# STEP is ror_data, index, match, evaluate, pipeline or verify (a script in jobs/). The options go to hf jobs run,
# for example --flavor h200 --timeout 2h --detach -e INPUT=... . This script adds:
#
#   -v <a copy of the code>:/code          src, jobs, pyproject.toml, uv.lock, README.md and .python-version
#   -v hf://buckets/$BUCKET:/bucket        the outputs (read and write)
#   -v hf://datasets/$DATASET:/dataset     the dataset (read only)
#   --secrets HF_TOKEN                     the models and the dataset are private
#
# Environment: BUCKET (default adambuttrick/ror-set-point-matching-jobs), DATASET
# (default adambuttrick/ror-set-point-matching-train-validation-test), IMAGE (default
# ghcr.io/astral-sh/uv:0.12.1-python3.12-trixie).
set -euo pipefail

step=${1:?usage: jobs/submit.sh STEP [OPTIONS OF hf jobs run]}
shift
root=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$root/jobs/$step.sh" ] || { echo "no step $step (jobs/$step.sh)" >&2; exit 2; }
BUCKET=${BUCKET:-adambuttrick/ror-set-point-matching-jobs}
DATASET=${DATASET:-adambuttrick/ror-set-point-matching-train-validation-test}
IMAGE=${IMAGE:-ghcr.io/astral-sh/uv:0.12.1-python3.12-trixie}

# Each job gets its own copy of the code. hf jobs run uploads it to the bucket <your user>/jobs-artifacts.
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
code=$stage/ror-set-point-matching
mkdir "$code"
tar -C "$root" --exclude __pycache__ -cf - src jobs pyproject.toml uv.lock README.md .python-version | tar -C "$code" -xf -

hf jobs run --name "ror-$step" --label "step=$step" --secrets HF_TOKEN \
    -v "$code:/code" -v "hf://buckets/$BUCKET:/bucket" -v "hf://datasets/$DATASET:/dataset" \
    "$@" "$IMAGE" bash "/code/jobs/$step.sh"
