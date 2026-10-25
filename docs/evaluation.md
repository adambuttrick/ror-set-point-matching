# Evaluation

This document tells how to evaluate a matcher on the labeled splits of the dataset, and gives the results of the released models. The code is in `src/ror_set_point/evaluation/`.

| Module | Content |
|---|---|
| `score.py` | Exact match, precision and recall of answer files, with paired tests and intervals |
| `recall.py` | The recall of candidate lists |
| `errors.py` | Recall by class of gold link, and wrong links by class |
| `stats.py` | The exact McNemar test, the minimum detectable effect and bootstrap intervals |
| `groups.py` | Record groups of the held-out institutions, for `score.py --record-group` and `recall.py --record-group` |

## Measures

| Measure | Definition |
|---|---|
| Exact match | The share of strings whose answer is exactly the set of gold records |
| Precision | Correct named links / all named links, over all strings (micro average) |
| Recall | Correct named links / all gold links, over all strings (micro average) |
| Known and new | A record is known when a string of the train split names it, and new when no string does. Recall splits by the gold record, and precision by the named record |
| Latin and non-Latin | A string is non-Latin when fewer than half of its letters are Latin |
| NIL answered | The share of strings that name no record but get an answer |

All measures are strict over ROR IDs. The scripts resolve gold labels in the records folder: a withdrawn record goes to its successor, and a label with no successor is dropped.

Exact match is the main measure. The works of an institution come from the answers for all its strings. Thus a wrong or missing record in an answer is an error.

## Evaluation splits

| Split | Strings | What it measures |
|---|---:|---|
| `affiliations` validation | 1,947 | Strings from Crossref and OpenAlex, about one third non-Latin |
| `affiliations` test | 1,947 | The same, for the final result |
| `affiliations_crossref`, `affiliations_openalex`, `affiliations_openalex_non_latin` | 447, 751, 749 (test) | The rows of `affiliations` for one source each, with the same splits and ids |
| `affiliations_unseen_institutions` validation | 1,914 | Institutions that the train split does not name. It holds all 272 links to CNRS |
| `affiliations_unseen_institutions` test | 1,957 | The same, for the final result. It has no CNRS strings |
| `datacite` test | 3,000 | Strings from DataCite metadata |
| `openalex_sample` test | 40,000 | Two random samples of OpenAlex strings (v1 and v2), the test samples of openalex-affiliation-matcher |

## Rules

- Do not train on a validation or test split.
- Make all choices on the validation splits. Score the test splits one time, at the end.
- Before you score, remove each evaluation string that is equal, after normalization, to a training string. Use `--exclude-equal-to` with the training files. `teacher_judgments` and `synthetic_non_latin` can contain such strings.
- The per-source configs hold the same strings as `affiliations`. Do not combine them with `affiliations`.
- `affiliations_unseen_institutions` uses the same strings as `affiliations` train and `openalex_sample` train. A model that trained on those two splits saw the held-out institutions. For such a model, do not use `affiliations_unseen_institutions`.

## Score answer files

Answer files have one row for each string: `{id, ror_ids}`. The `id` is the `id` of the dataset row. Rows with OpenAlex IDs (`{id, ids}`) are also accepted. A string with no row has an empty answer.

```
U=dataset/data/affiliations_unseen_institutions
uv run python -m ror_set_point.evaluation.score --gold $U/test.jsonl --records dataset/data/ror_records \
    --train $U/train.jsonl --system baseline baseline/evaluation/predictions/unseen_test.jsonl \
    --system "set model" set_model/predictions/unseen_test.jsonl --paired --bootstrap 2000
```

| Option | Effect |
|---|---|
| `--system NAME FILE` | One system. Give the option once for each system |
| `--train FILE` | Adds the columns for known and new records |
| `--exclude-equal-to FILE ...` | Adds a block without the strings that are equal to a string of these files (fields `text`, `s` or `q`) |
| `--group-field FIELD` | Adds a block for each value of the field, for example `source` in `affiliations` |
| `--paired` | Compares each pair of systems with the exact McNemar test. It gives the strings that only A or only B gets right, the discordance psi, the minimum detectable effect (MDE; alpha 0.05, power 0.8) and the p-value |
| `--bootstrap N` | 95% percentile intervals for exact match, precision and recall, from N resamples of strings |
| `--watch ROR_ID` | Gives the recall of one record apart |
| `--record-group NAME FILE` | Adds a table with the recall of the links to a group of records, and the exact match of the strings that name one of them |

The output is Markdown.

## Measure candidate recall

```
uv run python -m ror_set_point.evaluation.recall --gold $U/test.jsonl --records dataset/data/ror_records \
    --train $U/train.jsonl \
    --candidates retriever retriever/topk.jsonl.gz --set unseen_test --lexical
```

`--candidates` reads ranked lists (`{id, ids}`) or the candidate rows of `match candidates` (`{id, candidates}`). A file with several sets, for example the `topk.jsonl.gz` of a training run, needs `--set`. For ranked lists, the script gives the recall at each cut in `--ks`. With `--lexical`, it also gives the lexical top 10, and each ranked list's top 32 with the lexical top 10 added.

## Classify the errors

```
uv run python -m ror_set_point.evaluation.errors --gold $U/validation.jsonl --records dataset/data/ror_records \
    --train $U/train.jsonl --system baseline baseline/evaluation/predictions/unseen_validation.jsonl \
    --system "set model" set_model/predictions/unseen_validation.jsonl set_data/eval.jsonl.gz --set unseen_validation
```

Each gold link gets these classes:

- the string names one record, or several;
- the string is Latin or non-Latin;
- the record is known or new;
- the ROR type of the record;
- how the string names the record: a name, an acronym only, or neither.

A third file after the answers of a system gives its candidates. The misses of that system then split into "not a candidate" and "a candidate below the threshold".

Each wrong named link goes into one of these classes: on a string that names no record, on a parent or child of a gold record, on another related record, or on an unrelated record.

Use the error classes on validation splits only.

## Results

The tables give exact match, precision and recall in percent. The answers of ror-set-point on the test splits come from `match run`, on one NVIDIA H200. The model repositories hold these answers (see [Reproduce the results](#reproduce-the-results)).

The systems are:

| System | What it is |
|---|---|
| v3.0.0, local | openalex-affiliation-matcher v3.0.0 with the parts anyone can run: lexical and dense candidates, the pair scorer and the chooser. It trained on some strings and institutions of `affiliations_unseen_institutions` |
| Baseline, unseen split | The design of v3.0.0, retrained on the train split of `affiliations_unseen_institutions` ([ror-set-point-pair-scorer-baseline](https://huggingface.co/adambuttrick/ror-set-point-pair-scorer-baseline)). Its pair scorer starts from the pair scorer of v3.0.0 and also trains on teacher pairs |
| Baseline, all | The same design, retrained on `affiliations` train and `openalex_sample` train (all institutions) |
| ror-set-point | The released retriever and set model. No training row of these models has a held-out institution of `affiliations_unseen_institutions` as a positive ([phase 1](training-1-data.md#strings-that-training-does-not-use)) |
| Set-model variants | The set models on bge-reranker-v2-m3 and on EmbeddingGemma, with the candidate lists in revision `ae056b6` of the released set model's repository. Their training rows contain synthetic strings that name held-out institutions |

The repository of the baseline holds the answers of v3.0.0 and of "Baseline, all" (`comparison/`).

### Test splits

| Split | Strings | System | Exact match | Precision | Recall |
|---|---:|---|---:|---:|---:|
| `affiliations` (baseline, all) | 1,947 | v3.0.0, local | 78.4 | 94.3 | 79.0 |
|  |  | Baseline | 86.1 | 96.1 | 88.1 |
|  |  | ror-set-point | 90.0 | 95.7 | 92.6 |
| `affiliations_crossref` | 447 | v3.0.0, local | 85.9 | 96.7 | 88.6 |
|  |  | Baseline | 91.7 | 97.9 | 93.9 |
|  |  | ror-set-point | 92.8 | 97.3 | 95.5 |
| `affiliations_openalex` | 751 | v3.0.0, local | 79.6 | 94.9 | 81.7 |
|  |  | Baseline | 87.5 | 96.4 | 90.3 |
|  |  | ror-set-point | 90.0 | 96.0 | 93.0 |
| `affiliations_openalex_non_latin` | 749 | v3.0.0, local | 72.6 | 91.0 | 68.3 |
|  |  | Baseline | 81.4 | 94.0 | 80.8 |
|  |  | ror-set-point | 88.4 | 94.1 | 89.6 |
| `affiliations_unseen_institutions` (baseline, unseen split) | 1,957 | v3.0.0, local | 77.0 | 95.9 | 80.0 |
|  |  | Baseline | 79.2 | 93.6 | 85.1 |
|  |  | ror-set-point | 83.6 | 94.6 | 86.9 |
| `datacite` (baseline, all) | 3,000 | v3.0.0, local | 92.4 | 97.7 | 93.2 |
|  |  | Baseline | 91.9 | 95.1 | 95.6 |
|  |  | ror-set-point | 93.5 | 98.0 | 94.5 |
| `openalex_sample`, sample v1 (baseline, all) | 20,000 | v3.0.0, local | 81.6 | 95.3 | 83.2 |
|  |  | Baseline | 86.9 | 95.6 | 90.0 |
|  |  | ror-set-point | 89.9 | 96.1 | 92.1 |
| `openalex_sample`, sample v2 (baseline, all) | 20,000 | v3.0.0, local | 82.6 | 94.9 | 84.2 |
|  |  | Baseline | 88.0 | 95.2 | 91.5 |
|  |  | ror-set-point | 90.4 | 95.7 | 93.3 |
| `openalex_sample`, high-confidence labels (both samples) | 35,642 | v3.0.0, local | 86.8 | 96.3 | 88.4 |
|  |  | Baseline | 92.0 | 96.3 | 95.0 |
|  |  | ror-set-point | 94.3 | 97.2 | 96.4 |

Paired exact match: the strings where only the other system is right, the strings where only ror-set-point is right, and the exact McNemar p-value:

| Split | Other system | Only the other system is right | Only ror-set-point is right | Exact McNemar p |
|---|---|---:|---:|---:|
| `affiliations` | Baseline | 74 | 150 | 4 × 10⁻⁷ |
|  | v3.0.0, local | 52 | 279 | 1 × 10⁻³⁸ |
| `affiliations_unseen_institutions` | Baseline | 82 | 170 | 3 × 10⁻⁸ |
|  | v3.0.0, local | 106 | 237 | 1 × 10⁻¹² |
| `datacite` | Baseline | 62 | 112 | 2 × 10⁻⁴ |
|  | v3.0.0, local | 71 | 106 | 0.01 |
| `openalex_sample` (both samples) | Baseline | 1,252 | 2,333 | 8 × 10⁻⁷⁴ |
|  | v3.0.0, local | 1,010 | 4,233 | < 10⁻³⁰⁰ |

The p-values compare two answer files. They do not show the variation from another order of the candidates (see [Notes](#notes)). The differences on `datacite` are less than 2 points, which is the same size as that variation.

On `affiliations` test, the 95% bootstrap intervals (2,000 resamples of strings) are:

| System | Exact match | Precision | Recall |
|---|---|---|---|
| ror-set-point | 88.7–91.3 | 94.8–96.6 | 91.4–93.7 |
| Baseline, all | 84.6–87.6 | 95.2–96.9 | 86.7–89.6 |
| v3.0.0, local | 76.6–80.2 | 93.1–95.4 | 77.1–80.9 |

### New institutions

The validation and test splits of `affiliations_unseen_institutions` name 1,255 held-out institutions that no string of its train split names. `ror_set_point.evaluation.groups` writes the list, and `score.py --record-group` scores it. The table gives the recall of the links to these institutions, and the exact match of the strings that name one of them:

| Split | Links | Strings | System | Recall | Exact match |
|---|---:|---:|---|---:|---:|
| test | 1,755 | 1,694 | v3.0.0, local | 83.6 | 74.5 |
|  |  |  | Baseline, unseen split | 86.3 | 77.2 |
|  |  |  | ror-set-point | 89.6 | 82.8 |
| validation | 1,844 | 1,654 | v3.0.0, local | 68.0 | 58.2 |
|  |  |  | Baseline, unseen split | 84.9 | 72.8 |
|  |  |  | ror-set-point | 78.7 | 71.2 |
| validation without the CNRS strings | 1,440 | 1,382 | v3.0.0, local | 79.1 | 69.0 |
|  |  |  | Baseline, unseen split | 83.5 | 75.1 |
|  |  |  | ror-set-point | 86.5 | 79.9 |

The validation split holds the 272 strings that name CNRS. The recall of their links to CNRS is 33.8 for ror-set-point and 93.8 for the baseline.

### Set-model variants

The three set models read the same candidate lists, in [revision `ae056b6`](https://huggingface.co/adambuttrick/ror-set-point-set-model-mmbert-base/tree/ae056b667c8d8cb7db723f009f2fda03efb391da/evaluation/candidates) of the released model's repository. Their lexical candidates come from an index that reads the institutions file of openalex-affiliation-matcher v3.0.0 and adds a bonus for the works count. Thus the values of the released model differ a little from those in [Test splits](#test-splits). The variants trained on rows built without `--strict`. These rows contain the synthetic strings that name a held-out institution. Thus the values of the variants on `affiliations_unseen_institutions` are not a measure of new institutions.

| Split | Set model | Exact match | Precision | Recall |
|---|---|---:|---:|---:|
| `affiliations` | mmBERT-base (released) | 90.3 | 95.8 | 92.6 |
|  | bge-reranker-v2-m3 | 90.4 | 95.6 | 92.1 |
|  | EmbeddingGemma | 90.7 | 96.6 | 92.0 |
| `affiliations_unseen_institutions` | mmBERT-base (released) | 83.7 | 94.5 | 86.8 |
|  | bge-reranker-v2-m3 | 82.3 | 93.6 | 86.9 |
|  | EmbeddingGemma | 81.5 | 94.4 | 84.8 |
| `datacite` | mmBERT-base (released) | 93.5 | 97.9 | 94.5 |
|  | bge-reranker-v2-m3 | 94.3 | 97.8 | 95.3 |
|  | EmbeddingGemma | 93.0 | 98.3 | 93.8 |
| `openalex_sample`, sample v1 | mmBERT-base (released) | 89.8 | 96.1 | 92.1 |
|  | bge-reranker-v2-m3 | 88.8 | 95.6 | 91.4 |
|  | EmbeddingGemma | 89.6 | 96.2 | 91.6 |
| `openalex_sample`, sample v2 | mmBERT-base (released) | 90.4 | 95.8 | 93.3 |
|  | bge-reranker-v2-m3 | 89.3 | 95.0 | 92.4 |
|  | EmbeddingGemma | 90.0 | 95.8 | 92.7 |

On `affiliations` test, the exact McNemar test gives p = 0.93 for the released model against bge-reranker-v2-m3 (63 / 65), and p = 0.58 against EmbeddingGemma (55 / 62).

### Use of the test splits

No model, setting or training-data rule comes from the scores on a test split. The `affiliations_unseen_institutions` test split is a pass-or-fail check for the retriever and for the set model. `openalex_sample` test and `datacite` also have scores for information only, in addition to the final scores.

### Notes

- Some test strings are equal to a training string of ror-set-point. Without these strings, the exact match / precision / recall of ror-set-point is 90.0 / 95.7 / 92.5 on `affiliations` (1,937 strings). It is 93.2 / 97.9 / 94.2 on `datacite` (2,824 strings), and 90.1 / 95.9 / 92.7 on `openalex_sample` (39,925 strings). `affiliations_unseen_institutions` has no such strings.
- Of the 3,000 `datacite` strings, 2,140 (71%) are equal, after normalization, to a name, acronym, alias or label of their only gold record. On these strings, exact match is 96.5 for v3.0.0, local, 94.2 for the baseline and 95.5 for ror-set-point. On the other 860 strings, it is 82.0, 86.0 and 88.6.
- v3.0.0 and the baseline used the whole OpenAlex non-Latin file, with its test rows, for some design choices. Their scores on the non-Latin rows of `affiliations` test can be high for that reason.
- The answers depend a little on the candidate lists. A search in another numeric precision, or another random order of the candidates, changes some answers. On the two validation splits, another order of the same candidates changed the answers of 3 to 5% of the strings. Over 8 orders, exact match changed by up to 1.3 points. Differences of this size between two systems are not meaningful.

### Validation splits

The answers on the validation splits come from the training run of each model.

| Split | Strings | System | Exact match | Precision | Recall |
|---|---:|---|---:|---:|---:|
| `affiliations` | 1,947 | v3.0.0, local | 80.1 | 93.5 | 79.6 |
|  |  | Baseline, unseen split | 85.7 | 94.7 | 87.4 |
|  |  | ror-set-point | 91.0 | 96.5 | 91.9 |
| `affiliations_crossref` | 449 | v3.0.0, local | 85.7 | 97.7 | 84.4 |
|  |  | Baseline, unseen split | 92.0 | 97.7 | 92.7 |
|  |  | ror-set-point | 93.3 | 98.3 | 93.1 |
| `affiliations_openalex` | 748 | v3.0.0, local | 84.4 | 94.1 | 86.7 |
|  |  | Baseline, unseen split | 88.5 | 94.3 | 92.1 |
|  |  | ror-set-point | 92.4 | 96.8 | 94.3 |
| `affiliations_openalex_non_latin` | 750 | v3.0.0, local | 72.4 | 88.8 | 66.9 |
|  |  | Baseline, unseen split | 79.1 | 92.4 | 77.3 |
|  |  | ror-set-point | 88.3 | 94.6 | 88.0 |
| `affiliations_unseen_institutions` | 1,914 | v3.0.0, local | 62.7 | 96.5 | 67.7 |
|  |  | Baseline, unseen split | 75.0 | 94.8 | 84.8 |
|  |  | ror-set-point | 73.9 | 95.4 | 81.7 |

The `affiliations_unseen_institutions` validation split holds 272 links to CNRS. Most of these strings name CNRS only with its acronym, inside the name of a laboratory.

### Where ror-set-point errs

On the `affiliations_unseen_institutions` validation split without the CNRS strings (1,642 strings, 2,306 links):

| Gold links | Links | Baseline recall | ror-set-point recall |
|---|---:|---:|---:|
| Named by a name of the record | 1,476 | 96.1 | 93.0 |
| Named by an acronym only | 147 | 76.9 | 78.2 |
| Named in another form (translated or short) | 683 | 57.8 | 70.7 |
| In strings that name one record | 784 | 89.2 | 94.1 |
| In strings that name several records | 1,522 | 80.6 | 81.0 |
| In non-Latin strings | 362 | 72.9 | 82.9 |

Of the 335 links that ror-set-point misses, 39 are not among the candidates. The set model scores the other 296 below the threshold. Thus most errors come from the set model, not from the retriever.

## Reproduce the results

The model repositories contain their answers in `evaluation/predictions/<set>.jsonl`. The set names are:

| Name | Split |
|---|---|
| `unseen_validation` | `affiliations_unseen_institutions` validation |
| `unseen_test` | `affiliations_unseen_institutions` test |
| `affiliations_validation` | `affiliations` validation |
| `affiliations_test` | `affiliations` test |
| `datacite_test` | `datacite` test |
| `openalex_sample_test` | `openalex_sample` test |

The three set models have all six. The baseline has the first three. The repository of the baseline also holds the answers of v3.0.0 on all six (`comparison/v3.0.0_local/<set>.jsonl`), and of "Baseline, all" on `affiliations_test`, `datacite_test` and `openalex_sample_test` (`comparison/all_institutions/<set>.jsonl`). These rows have OpenAlex institution IDs (`{id, ids}`). `score.py` changes them to ROR IDs with the records folder.

The answers of the released set model on the test splits come from this command:

```
uv run python -m ror_set_point.inference.match run --input dataset/data/affiliations/test.jsonl \
    --records dataset/data/ror_records --retriever adambuttrick/ror-set-point-retriever-e5-base \
    --set-model adambuttrick/ror-set-point-set-model-mmbert-base --device cuda --workers 16 --out affiliations_test.jsonl
```

The repository of the released set model also holds these candidates (`evaluation/candidates/<set>.jsonl.gz`). Give them to `match answer` to answer the same lists again. On other hardware, a small number of answers can change. The variants answer the candidate lists in revision `ae056b6` (see [Set-model variants](#set-model-variants)).

Download the answers, then score them:

```
hf download adambuttrick/ror-set-point-set-model-mmbert-base --include "evaluation/predictions/*" \
    --local-dir set_model_released
hf download adambuttrick/ror-set-point-pair-scorer-baseline --include "evaluation/predictions/*" \
    --local-dir baseline
uv run python -m ror_set_point.evaluation.score --gold $U/test.jsonl --records dataset/data/ror_records \
    --train $U/train.jsonl --system baseline baseline/evaluation/predictions/unseen_test.jsonl \
    --system "set model" set_model_released/evaluation/predictions/unseen_test.jsonl --paired
```

The result is the `affiliations_unseen_institutions` row above. The baseline has 79.2 / 93.6 / 85.1, and the set model has 83.6 / 94.6 / 86.9. The paired test gives 82 against 170 (MDE 2.3 points).
