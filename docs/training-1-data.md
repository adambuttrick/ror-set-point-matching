# Training phase 1: data

This phase makes the data that the other phases read:

- the records folder, which holds the ROR records and the maps for withdrawn records and OpenAlex IDs;
- the teacher judgments, with ROR IDs;
- the split of the labeled strings by institution (`affiliations_unseen_institutions`).

The dataset [ror-set-point-matching-train-validation-test](https://huggingface.co/datasets/adambuttrick/ror-set-point-matching-train-validation-test) already contains the results. Its folder `data/ror_records` is a records folder: `train.jsonl` holds the records, and `withdrawn.json` and `openalex_to_ror.json` are the two maps. The configs `teacher_judgments` and `affiliations_unseen_institutions` hold the other results. Do this phase to make them again, or to use a different ROR release.

## Inputs

| Input | Source | sha256 |
|---|---|---|
| `v2.14-2026-10-06-ror-data.json` | The ROR data dump v2.14 (6 October 2026), from the [ROR data releases](https://zenodo.org/communities/ror-data) | `3b620ada133f533b7eddd01ded9d1a9dd647c7676d7c48b1c619b4bc52b7cf3e` |
| `institutions.jsonl.gz` | `models/` in [openalex-affiliation-matcher](https://github.com/ourresearch/openalex-affiliation-matcher), tag v3.0.0 | `7d2c5dc0da4ba13607f47d35c025d48cbe510d1c33c635e4ba69343a8ad9a57e` |
| `train_v1_jev_labels.jsonl.gz` | The [v3.0.0 release](https://github.com/ourresearch/openalex-affiliation-matcher/releases/tag/v3.0.0) of openalex-affiliation-matcher | `50e32c7c567a2362c5fc0e8b8255db05280d04d0a80f775c3c9d21f6faf75f73` |
| `affiliations` train and validation, `openalex_sample` train | The dataset | |

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

## Step 2: change the teacher judgments to ROR IDs

```
uv run python -m ror_set_point.training.data.teacher --judgments train_v1_jev_labels.jsonl.gz --records records \
    --out teacher_judgments
```

The input has one row for each (string, OpenAlex institution) pair. The output has one row for each string, with a list of `{ror_id, p}`. `p` is the probability, from the teacher, that the string names the record.

- An OpenAlex ID goes to the ROR ID of its institution.
- A withdrawn ROR ID goes to its successor.
- A judgment with no ROR ID is dropped (68 have no ROR ID, and 105 have a withdrawn record with no successor).
- When two judgments of one string get the same ROR ID, the higher `p` stays (7 times).

The output has 196,018 train strings and 3,982 validation strings, with 3,763,451 judgments.

## Step 3: divide the labeled strings by institution

```
uv run python -m ror_set_point.training.data.split --records records \
    --affiliations-train dataset/data/affiliations/train.jsonl \
    --openalex-sample-train dataset/data/openalex_sample/train.jsonl \
    --affiliations-validation dataset/data/affiliations/validation.jsonl \
    --teacher teacher_judgments/train.jsonl teacher_judgments/validation.jsonl \
    --seed 20261022 --out unseen
```

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

## Strings that training does not use

`src/ror_set_point/common/exclusions.py` holds the rule. Phases 3 and 4 apply it.

- A teacher string is not used when the teacher gives p ≥ 0.5 to a held-out record (28,474 train strings).
- A teacher string is not used when it is equal to an evaluation string (11 more train strings).
- A labeled or synthetic string is not used when it is equal to an evaluation string.
- With `--strict`, the builders also leave out the synthetic strings that name a held-out record (2,871 strings). The released models use this option. Thus no training row of them has a held-out record as a positive (a label, or a teacher p ≥ 0.5).

The evaluation strings are the validation and test splits of `affiliations_unseen_institutions`, and the `affiliations` validation split.

A teacher string that gives a held-out record a p below 0.5 stays. Stage 1 of the set model uses the teacher's p as its target. Thus 9,606 of its strings have a target from 0.1 to 0.49 for a held-out record. On the validation and test splits, the recall of a held-out record does not increase with the sum of its targets.

## Check the result

With the inputs above, the outputs are equal to the dataset:

- The rows of `records/records.jsonl.gz` are the rows of `ror_records`, and the two maps are equal to the maps in `data/ror_records`.
- `teacher_judgments/train.jsonl` and `teacher_judgments/validation.jsonl` are equal, byte for byte, to the `teacher_judgments` config.
- `unseen/train.jsonl`, `validation.jsonl`, `test.jsonl` and `held_out_records.jsonl` are equal, byte for byte, to the `affiliations_unseen_institutions` config.
