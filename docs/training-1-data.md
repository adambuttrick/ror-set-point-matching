# Training phase 1: data

This phase makes the data that the other phases read:

- the records folder, which holds the ROR records and the maps for withdrawn records and OpenAlex IDs;
- the split of the labeled strings by institution (`affiliations_unseen_institutions`);
- the teacher judgments: for each (string, ROR record) pair, the probability that the string names the record.

The dataset [ror-set-point-matching-train-validation-test](https://huggingface.co/datasets/adambuttrick/ror-set-point-matching-train-validation-test) already contains the results. Its folder `data/ror_records` is a records folder: `train.jsonl` holds the records, and `withdrawn.json` and `openalex_to_ror.json` are the two maps. The configs `teacher_judgments` and `affiliations_unseen_institutions` hold the other results. Do this phase to make them again, or to use a different ROR release.

The teacher is [StartLux-Decision v1.1 27B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant), a decision model with open weights. It judges the (string, institution) pairs that Jev judged for openalex-affiliation-matcher v3.0.0. Jev is a closed commercial model from TypeSafe AI. The dataset [ror-set-point-matching-train-validation-test-openalex-jev-distill](https://huggingface.co/datasets/adambuttrick/ror-set-point-matching-train-validation-test-openalex-jev-distill) holds Jev's judgments of the same pairs, with ROR IDs.

## Inputs

| Input | Source | sha256 or revision |
|---|---|---|
| `v2.14-2026-10-06-ror-data.json` | The ROR data dump v2.14 (6 October 2026), from the [ROR data releases](https://zenodo.org/communities/ror-data) | `3b620ada133f533b7eddd01ded9d1a9dd647c7676d7c48b1c619b4bc52b7cf3e` |
| `institutions.jsonl.gz` | `models/` in [openalex-affiliation-matcher](https://github.com/ourresearch/openalex-affiliation-matcher), tag v3.0.0 | `7d2c5dc0da4ba13607f47d35c025d48cbe510d1c33c635e4ba69343a8ad9a57e` |
| `train_v1_jev_labels.jsonl.gz` | The [v3.0.0 release](https://github.com/ourresearch/openalex-affiliation-matcher/releases/tag/v3.0.0) of openalex-affiliation-matcher | `50e32c7c567a2362c5fc0e8b8255db05280d04d0a80f775c3c9d21f6faf75f73` |
| `affiliations` train and validation, `openalex_sample` train | The dataset | |
| The teacher model | [startlux-models/StartLux-Decision-v1.1-27B-Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant) (CC BY-NC 4.0) | `32e7d35d74db615c4a059181efbf21c5421f3561` |
| The teacher server | `startlux-decide` from [StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision), with vLLM 0.31.0 | `a99977d9332decc9fb27a2488b4a83b533b6cd5f` |
| The question | `src/ror_set_point/training/data/questions/v1.2.txt` | |
| The calibration pairs | `training_data/stage2.jsonl.gz` of [ror-set-point-set-model-mmbert-base-openalex-jev-distill](https://huggingface.co/adambuttrick/ror-set-point-set-model-mmbert-base-openalex-jev-distill) | `6700054b39ffc183f8a6d2bb1b5a9f056d1aae9a732205e400af5f78419b27ce` |

Download the files of openalex-affiliation-matcher and the dataset (into the folder `dataset`):

```
curl -LO https://raw.githubusercontent.com/ourresearch/openalex-affiliation-matcher/v3.0.0/models/institutions.jsonl.gz
curl -LO https://github.com/ourresearch/openalex-affiliation-matcher/releases/download/v3.0.0/train_v1_jev_labels.jsonl.gz
hf download adambuttrick/ror-set-point-matching-train-validation-test --repo-type dataset --local-dir dataset
```

Make sure that the sha256 of each file is the value in the table.

## Step 1: make the records folder

```
uv run python -m ror_set_point.training.data.records --ror v2.14-2026-10-06-ror-data.json \
    --openalex-institutions institutions.jsonl.gz --out records
```

The step writes these files:

| File | Content |
|---|---|
| `records.jsonl.gz` | One row for each active or inactive ROR record. The rows are the same as the `ror_records` config |
| `withdrawn.json` | For each withdrawn record, its successor record, or `null` |
| `openalex_to_ror.json` | For each OpenAlex institution, its ROR ID |
| `manifest.json` | The sha256 of the inputs and outputs, and the counts |

Each record row has a `text` field. The models read this text. It has this form:

```
name | acronyms | other names | places | types | status | parent: ...; child: ...; related: ...
```

Each relation list shows a maximum of 8 names, then "(+N more)".

A withdrawn record goes to its successor when it has exactly one successor link. When that successor is also withdrawn, the chain continues. The chain must end at an active or inactive record.

| Count | Value |
|---|---:|
| Records in the dump | 144,717 |
| Active records | 141,387 |
| Inactive records | 1,897 |
| Withdrawn records that go to an active record | 1,248 |
| Withdrawn records that go to an inactive record | 36 |
| Withdrawn records with no successor | 149 |

## Step 2: change Jev's judgments to ROR IDs

```
uv run python -m ror_set_point.training.data.teacher --judgments train_v1_jev_labels.jsonl.gz --records records \
    --out teacher_jev
```

The input has one row for each (string, OpenAlex institution) pair. The output has one row for each string, with a list of `{ror_id, p}`. `p` is the probability, from Jev, that the string names the record.

- An OpenAlex ID goes to the ROR ID of its institution.
- A withdrawn ROR ID goes to its successor.
- A judgment with no ROR ID is dropped (68 have no ROR ID, and 105 have a withdrawn record with no successor).
- When two judgments of one string get the same ROR ID, the higher `p` stays (7 times).

The output has 196,018 train strings and 3,982 validation strings, with 3,763,451 judgments. It is the `teacher_judgments` config of the `-openalex-jev-distill` dataset. Its pairs are the pairs that the teacher judges in step 4.

## Step 3: divide the labeled strings by institution

```
uv run python -m ror_set_point.training.data.split --records records \
    --affiliations-train dataset/data/affiliations/train.jsonl \
    --openalex-sample-train dataset/data/openalex_sample/train.jsonl \
    --affiliations-validation dataset/data/affiliations/validation.jsonl \
    --teacher teacher_jev/train.jsonl teacher_jev/validation.jsonl \
    --seed 20261022 --out unseen
```

The step reads only the strings of the teacher files. Step 4 does not change the strings, so the `teacher_judgments` config of the dataset gives the same result.

This split measures how well a model finds institutions that it did not see in training. The procedure is:

1. Collect the labeled strings: `affiliations` train and `openalex_sample` train (20,387 strings). A label on a withdrawn record goes to its successor.
2. Put each ROR record of the labels into a stratum. The stratum is the type of the record, the script of its strings and the number of its strings (1, 2–3, 4–9 or 10+).
3. In each stratum, hold out 15% of the records, in sha256 order of the seed and the ROR ID. The count is rounded up or down at random, so that small strata do not always round down.
4. Move every string that names a held-out record out of the train split. Also move 15% of the strings that name no record.
5. Put the moved strings into groups. Strings that name the same held-out record are in one group.
6. Give the groups, in sha256 order, to the validation and test splits. Each group goes to the split that has fewer strings at that time. Do this in each source and script.
7. Remove duplicates. Strings are compared after normalization (NFKC, casefold, and non-word characters as spaces). A train string that is equal to a moved string leaves the train split. A moved string that is equal to an `affiliations` validation string, or to a teacher string, is dropped.

The step writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, `held_out_records.jsonl`, `excluded.jsonl` (the strings in no split, with the reason), `excluded_labels.jsonl` and `manifest.json`.

| Count | Value |
|---|---:|
| ROR records in the labels | 8,340 |
| Held-out records | 1,255 |
| Train strings | 16,452 |
| Records that train strings name | 6,519 |
| Validation strings | 1,914 |
| Test strings | 1,957 |
| Strings in no split | 64 |

The largest group (399 strings) holds every string that names CNRS (`https://ror.org/02feahw73`). It is in the validation split, so the test split has no CNRS strings.

## Step 4: ask the teacher

`ror_set_point.training.data.relabel` asks the teacher one question about each pair: does the string name the organization on the card? The answer is a probability. The steps of the module need only the standard library. The server of the teacher needs a GPU.

The card of a pair has the fields that openalex-affiliation-matcher v3.0.0 gave Jev. These are the ROR display name, up to 6 other names (labels, then aliases) and up to 4 acronyms. They also include the first type, and the city and country of the first place. The question (`questions/v1.2.txt`) follows the annotation guidelines v1.2 of the dataset. For example, a parent organization counts only when the string names it. The comparisons below also use Jev's question: the question that openalex-affiliation-matcher v3.0.0 asked Jev (`NAMES_IT` in [`matcher/training/jev_teacher.py`](https://github.com/ourresearch/openalex-affiliation-matcher/blob/v3.0.0/matcher/training/jev_teacher.py)).

The step has four parts:

1. Score the calibration pairs. These are the candidates of the labeled train strings, with their labels and folds (615,010 pairs). They come from the stage 2 rows of the `-openalex-jev-distill` set model.
2. Score the 3,763,451 pairs of step 2.
3. Fit the calibration on the calibration pairs.
4. Write the config: the pairs of step 2 with the calibrated p.

On Hugging Face Jobs, `jobs/relabel.sh` does parts 1 and 2, with one H200 for each shard ([jobs.md](jobs.md#the-teacher)). Parts 3 and 4 run on a CPU:

```
uv run python -m ror_set_point.training.data.relabel calibrate --scores stage2/*/scores.jsonl --out calibration.json
uv run python -m ror_set_point.training.data.relabel assemble --scores teacher/*/scores.jsonl --teacher teacher_jev \
    --calibration calibration.json --out teacher_judgments
```

To score the pairs without Jobs, start the server, then make the requests and send them:

```
startlux-decide serve --model StartLux-Decision-v1.1-27B-Instant --launch-vllm --gpus 0 --port 8090 \
    --no-images --max-model-len 16384
uv run python -m ror_set_point.training.data.relabel requests --teacher teacher_jev --records records \
    --question src/ror_set_point/training/data/questions/v1.2.txt --out requests.jsonl.gz
uv run python -m ror_set_point.training.data.relabel score --requests requests.jsonl.gz --out teacher/all/scores.jsonl
```

For the calibration pairs, give `--stage2 stage2.jsonl.gz` to `requests` in place of `--teacher`. The vLLM log of the server must show "prefix-bidirectional prompt attention ON". `score` sends 256 requests at a time. A run that stops continues from the pairs that have no p.

On 1 × NVIDIA H200, the teacher answers 24 pairs each second. Each request has about 514 tokens. The 3,763,451 pairs took 24 jobs of about 1.9 hours each.

The calibration is Platt scaling on the logit of the teacher's p: p' = 1 / (1 + exp(-(a × logit(p) + b))). The fit on all 615,010 calibration pairs gives a = 2.3029 and b = -3.2924. As a check, `calibrate` also fits on folds 0 to 2 and measures the log loss on folds 3 and 4. The log loss is 0.103 for the raw p and 0.0144 for the calibrated p. With Jev's question, the calibrated log loss is 0.0180.

The pair-level quality of the raw p on the calibration pairs, without the parents of labels:

| Question | AUC | Precision / recall at 0.5 | Yes for a parent that the string does not name |
|---|---:|---:|---:|
| v1.2 | 0.9986 | 71.3 / 98.3 | 23% |
| Jev's question | 0.9981 | 55.0 / 98.9 | 69% |

`assemble` writes these files:

| File | Content |
|---|---|
| `train.jsonl`, `validation.jsonl` | The rows of step 2, in the same order, with the calibrated p (4 decimals). These are the `teacher_judgments` config |
| `raw.jsonl.gz` | One row for each pair: `split`, `id`, `ror_id`, `p_raw` (the p of the teacher) and `p_jev` (the p of step 2) |
| `manifest.json` | The calibration, the counts and the sha256 of the inputs and outputs |

| Count | Value |
|---|---:|
| Train pairs | 3,688,556 |
| Validation pairs | 74,895 |
| Train pairs with p ≥ 0.5 | 188,547 (Jev: 391,457) |
| Validation pairs with p ≥ 0.5 | 3,855 (Jev: 7,933) |

## Strings that training does not use

`src/ror_set_point/common/exclusions.py` holds the rule. Phases 3 and 4 apply it.

- A teacher string is not used when the teacher gives p ≥ 0.5 to a held-out record (19,455 train strings).
- A teacher string is not used when it is equal to an evaluation string (12 more train strings).
- A labeled or synthetic string is not used when it is equal to an evaluation string.
- With `--strict`, the builders also leave out the synthetic strings that name a held-out record (2,871 strings). The released models use this option. Thus no training row of them has a held-out record as a positive (a label, or a teacher p ≥ 0.5).

The evaluation strings are the validation and test splits of `affiliations_unseen_institutions`, and the `affiliations` validation split.

A teacher string that gives a held-out record a p below 0.5 stays. Stage 1 of the set model uses the teacher's p as its target. Thus 1,600 of its strings have a target from 0.1 to 0.49 for a held-out record. When the string mentions the record, the mention mask of phase 4 removes the record from the loss of that row ([phase 4](training-4-set-model.md#the-mention-mask)).

## Check the result

With the inputs above, these outputs are equal to the datasets:

- The rows of `records/records.jsonl.gz` are the rows of `ror_records`, and the two maps are equal to the maps in `data/ror_records`.
- `teacher_jev/train.jsonl` and `teacher_jev/validation.jsonl` are equal, byte for byte, to the `teacher_judgments` config of the `-openalex-jev-distill` dataset.
- `unseen/train.jsonl`, `validation.jsonl`, `test.jsonl` and `held_out_records.jsonl` are equal, byte for byte, to the `affiliations_unseen_institutions` config.

The teacher's p is not always the same in two runs. The server puts the requests into batches, and the GPU does not always add numbers in the same order. In a second run of 1,884 pairs, 1,870 pairs got the same p. The largest difference was 0.0057, and no pair moved across 0.5. Thus `teacher_judgments` can differ a little from the dataset. From the same score files, `calibrate` gives the a and b of the dataset to 12 decimal places. With the a and b of the dataset, `assemble` gives the config byte for byte.

The folder `teacher/` of the dataset holds `raw.jsonl.gz` and `manifest.json` of `assemble`. The manifest gives a and b. For each pair, the calibrated `p_raw` (4 decimals) is the `p` of the config.
