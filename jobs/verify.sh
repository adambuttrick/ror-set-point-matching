#!/usr/bin/env bash
# Check the code and the released models end to end, against the published results. GPU.
#
# The job makes the records folder of ROR v2.14 and compares it with the dataset, makes the index, then does each
# split two times: match candidates, compared with the published candidates, and match run, compared with the
# published answers and scored against the gold links. The published files are in the set model's repository, which
# submit.sh mounts at /set_model with this option:
#
#   -v hf://adambuttrick/ror-set-point-set-model-mmbert-base:/set_model
#
#   RUN     the folder for the outputs (default /bucket/runs/verify). A run that stops continues there
#   SPLITS  the splits (default: the four test splits). The published files have answers for the four test splits
#           and for affiliations/validation and affiliations_unseen_institutions/validation. They have candidates
#           for the test splits only
source "$(dirname "$0")/lib.sh"
here=$(dirname "$0")

RUN=${RUN:-/bucket/runs/verify}
SPLITS=${SPLITS:-affiliations/test affiliations_unseen_institutions/test datacite/test openalex_sample/test}
PUBLISHED=${PUBLISHED:-/set_model/evaluation}
[ "$ROR_VERSION" = v2.14 ] || die "the published results use ROR v2.14"
need "$PUBLISHED"

describe_job
start=$SECONDS
bash "$here/ror_data.sh"
setup_env
bash "$here/index.sh"
mkdir -p "$RUN"
summary=$RUN/summary.md
: > "$summary.tmp"
for split in $SPLITS; do
    # The names of the published files: affiliations_unseen_institutions/test is unseen_test.
    name=${split//\//_}
    name=${name/affiliations_unseen_institutions/unseen}
    gold=/dataset/data/$split.jsonl
    log "== $split"
    INPUT=$gold OUT=$RUN/$name.candidates.jsonl MODE=candidates bash "$here/match.sh"
    INPUT=$gold OUT=$RUN/$name.jsonl MODE=run bash "$here/match.sh"
    {
        echo "## $split"
        echo
        if [ -e "$PUBLISHED/candidates/$name.jsonl.gz" ]; then
            py "$CODE/jobs/check.py" candidates "$RUN/$name.candidates.jsonl" "$PUBLISHED/candidates/$name.jsonl.gz"
        else
            echo "candidates: no published file"
        fi
        echo
        py "$CODE/jobs/check.py" answers "$RUN/$name.jsonl" "$PUBLISHED/predictions/$name.jsonl"
        echo
    } >> "$summary.tmp"
    GOLD=$gold ANSWERS=$RUN/$name.jsonl REFERENCE=$PUBLISHED/predictions/$name.jsonl bash "$here/evaluate.sh" \
        > /dev/null
    { cat "$RUN/$name.scores.md"; echo; } >> "$summary.tmp"
done
mv "$summary.tmp" "$summary"
cat "$summary"
log "verify: $(( SECONDS - start )) s; outputs in $RUN"
