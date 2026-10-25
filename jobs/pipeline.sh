#!/usr/bin/env bash
# The whole pipeline in one job: the records folder (step 1), the index (step 2), the answers (step 3) and, with
# GOLD, the scores (step 4). GPU. A step whose output exists is skipped or continued.
#
#   INPUT, OUT, MODE, WORKERS, MATCH_ARGS   as in match.sh
#   GOLD, REFERENCE, SCORE_ARGS             as in evaluate.sh (GOLD=$INPUT for a labeled split of the dataset)
#   ROR_VERSION, RECORDS, RETRIEVER, ...    as in lib.sh
source "$(dirname "$0")/lib.sh"
here=$(dirname "$0")

: "${INPUT:?set INPUT}" "${OUT:?set OUT}"
describe_job
bash "$here/ror_data.sh"
setup_env
bash "$here/index.sh"
bash "$here/match.sh"
if [ -n "${GOLD:-}" ]; then
    ANSWERS=$OUT bash "$here/evaluate.sh"
fi
