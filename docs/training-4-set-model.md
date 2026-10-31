# Training phase 4: the set model

The set model selects the records that a string names. It reads the string and all its candidate records in one input, and gives each candidate the probability that the string names it. The answer is every candidate with p ≥ 0.5.

The released model is [ror-set-point-set-model-mmbert-base](https://huggingface.co/adambuttrick/ror-set-point-set-model-mmbert-base).

## The model

1. The candidates are the retriever's top 32, then the lexical top 10 that are not already there (about 37.5 for each string). They are in a random order.
2. The input is `[CLS] string [SEP] record 1 [SEP] record 2 [SEP] ...`. Each record is its `text` field.
3. A linear layer gives a score to each candidate, from the mean of the token states of that candidate.
4. Platt scaling changes each score into a probability: p = 1 / (1 + exp(-(a × score + b))).

| Property | Value |
|---|---|
| Base model | [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base), revision `c5955035435e2bf121cde7f3c8863ef52ff35d82` (MIT), 307M parameters |
| Input | The string: up to 96 tokens. Each record: up to 64 tokens, fewer when the input would be longer than 3,072 tokens |
| Format | The encoder in the transformers format (bf16), `head.safetensors`, the tokenizer and `set_model_config.json` |

The model reads the candidates together. Thus it can learn rules that compare candidates. An example is the rule "label the hospital, but not the university in the hospital's name".

## Training data

| Stage | Source | Strings | Candidates | Targets |
|---|---|---:|---|---|
| 1 | `teacher_judgments` train, without the strings that the exclusion rule removes | 167,533 | The retriever's top 32, then the records that the teacher judged and that are not in it (positives first), to a maximum of 42 | The teacher's p. 0 for records with no judgment |
| 1 | `synthetic_non_latin`, without the strings that name a held-out record | 130,882 | The retriever's top 32 | 1 for the labels, 0 for other records |
| 2 | `affiliations_unseen_institutions` train | 16,448 | The retriever's top 32 and the lexical top 10 | 1 for the labels, 0 for other records |

- A label that is not among the candidates replaces a random candidate that is not a label (663 times).
- Labeled and synthetic strings that are equal to an evaluation string are not used (4 of each).
- The retriever trained on these strings. Thus their candidate lists hold the labels more often than the lists of new strings. The replacement of missing labels makes sure that each label is a candidate in training.
- The evaluation rows (`eval.jsonl.gz`) have the evaluation strings and their candidates, with no labels.

## Procedure

The commands use the folders of phases 1 to 3 and `U=dataset/data/affiliations_unseen_institutions`.

1. Make the rows:

   ```
   uv run python -m ror_set_point.training.set_model.build --records dataset/data/ror_records \
       --retrieved retriever/topk.jsonl.gz --teacher dataset/data/teacher_judgments/train.jsonl \
       --labelled $U/train.jsonl --synthetic dataset/data/synthetic_non_latin/train.jsonl \
       --held-out $U/held_out_records.jsonl \
       --eval unseen_validation $U/validation.jsonl --eval unseen_test $U/test.jsonl \
       --eval affiliations_validation dataset/data/affiliations/validation.jsonl --strict --out set_data -v
   ```

   The `--eval` options must be the same as in phase 3, because `--retrieved` holds the search results of those queries. `--strict` leaves out the synthetic strings that name a held-out record, as in phase 3. Then no training row of the two models has a held-out record as a positive. But stage 1 can give a held-out record a target below 0.5 (see [phase 1](training-1-data.md#strings-that-training-does-not-use)).

2. Do a smoke test:

   ```
   uv run python -m ror_set_point.training.set_model.train --data set_data --out set_model_smoke \
       --base jhu-clsp/mmBERT-base --base-revision c5955035435e2bf121cde7f3c8863ef52ff35d82 \
       --limit 3000 --steps1 40 --log-every 10
   ```

3. Train the model:

   ```
   uv run python -m ror_set_point.training.set_model.train --data set_data --out set_model \
       --base jhu-clsp/mmBERT-base --base-revision c5955035435e2bf121cde7f3c8863ef52ff35d82
   ```

   `--stage1 FILE` writes the stage 1 weights to FILE. When FILE exists, the script reads the weights from it and does not train stage 1. The stage 1 rows and the stage 1 settings must be the same. If they are not, the script stops. Stage 1 rows have no lexical candidates. Thus runs that differ only in their lexical candidates, for example with another `--lexical-k`, can use the same stage 1. This saves 1.7 hours on an H200.

The training script does these steps in sequence:

1. Stage 1: one epoch of the stage 1 rows.
2. Stage 2, five times: start from the stage 1 weights, train on four folds of the labeled rows, and score the fifth fold. This gives an out-of-fold score for each candidate of each labeled string.
3. Fit the calibration (a and b) on the out-of-fold scores.
4. Stage 2 on all labeled rows, from the stage 1 weights. This is the final model.
5. Answer the evaluation rows, and save the model.

| Output | Content |
|---|---|
| `model/` | The model, as `ror_set_point.inference.set_model` reads it |
| `calibration.json` | a, b, and the out-of-fold log loss, precision, recall and exact match |
| `oof.jsonl.gz` | The out-of-fold scores (`k`, `c` candidates, `y` targets, `z` scores) |
| `probabilities.jsonl.gz` | The score and probability of each candidate of each evaluation row |
| `predictions/<set>.jsonl` | The answers (`id`, `ror_ids`) for each evaluation split |
| `train_log.txt`, `summary.json` | The log, the settings, the steps and the times |

## Settings

| Setting | Value |
|---|---|
| Loss | Binary cross-entropy against the targets, over the real candidates |
| Stage 1 | 1 epoch (18,651 steps) at learning rate 2e-5 |
| Stage 2 | 2 epochs at learning rate 1e-5, from the stage 1 weights. The final run has 2,056 steps |
| Folds | 5, by the sha256 of the fold seed (20261022) and the string id |
| Batch | 16 strings |
| Optimizer | AdamW, weight decay 0.01 |
| Schedule | Linear warm-up for 5% of the steps, then linear decay |
| Precision | bf16 autocast. The saved encoder is bf16 |
| Gradient clipping | 1.0 |
| Seed | 1363 |
| Threshold | 0.5 |

The calibration of the released model is a = 0.818 and b = -0.071. It was fitted on 615,010 out-of-fold candidates. On the out-of-fold answers, precision is 95.4, recall 94.3 and exact match 90.9.

On the out-of-fold candidates, p agrees with the frequency of correct candidates. For example, the candidates with p from 0.35 to 0.5 have a mean p of 0.42, and 47% of them are correct. On the `affiliations` validation split, a new fit gives a = 0.807 and b = 0.108, near the values of the released model.

On the `affiliations_unseen_institutions` validation split, p from 0.05 to 0.5 is lower than the frequency of correct candidates. For example, 58% of the candidates with p from 0.2 to 0.35 are correct. This split is not a measure of calibration. It contains every labeled string that names a held-out record. Thus a candidate that is a held-out record is correct more frequently in this split than in other strings. In this split, 41% of these candidates are correct, and on the `affiliations` validation split 7.8%. When only this share changes, b moves by the change in its log-odds (2.1). A new fit on these candidates gives b = 3.30 on this split and b = 1.05 on the `affiliations` validation split, a difference of 2.25.

On the `affiliations` validation split, the difference for held-out records is smaller. Of the 23 candidates that are held-out records and have p from 0.05 to 0.5, 13 are correct. For the other records with no positive in training, p is equal to or higher than the frequency of correct candidates. A threshold below 0.5 decreases exact match on this split, for example from 91.0 to 90.4 at 0.3.

The values come from `oof.jsonl.gz` and `probabilities.jsonl.gz` in the repository of the released model.

## Hardware and time

The released model trained on 1 × NVIDIA H200 in 2.8 hours. The training script answers 130 strings each second on that GPU, in padded batches. At inference, with packed batches and flash-attention, it answers 753 strings each second ([inference.md](inference.md#the-set-model-on-a-cuda-gpu)).

## Evaluation

Exact match, precision and recall in percent. The answers on the test split come from `match run` on one NVIDIA H200. The answers on the validation splits come from the training run.

| Split | Strings | System | Exact match | Precision | Recall |
|---|---:|---|---:|---:|---:|
| `affiliations_unseen_institutions` test | 1,957 | v3.0.0, local | 77.0 | 95.9 | 80.0 |
|  |  | Baseline | 79.2 | 93.6 | 85.1 |
|  |  | This model | 83.6 | 94.6 | 86.9 |
| `affiliations_unseen_institutions` validation | 1,914 | v3.0.0, local | 62.7 | 96.5 | 67.7 |
|  |  | Baseline | 75.0 | 94.8 | 84.8 |
|  |  | This model | 73.9 | 95.4 | 81.7 |
| `affiliations` validation | 1,947 | v3.0.0, local | 80.1 | 93.5 | 79.6 |
|  |  | Baseline | 85.7 | 94.7 | 87.4 |
|  |  | This model | 91.0 | 96.5 | 91.9 |

Recall of the links to the 1,255 held-out institutions is 86.3 for the baseline and 89.6 for this model on the test split (1,755 links). On the validation split (1,844 links), it is 84.9 and 78.7.

The baseline ([ror-set-point-pair-scorer-baseline](https://huggingface.co/adambuttrick/ror-set-point-pair-scorer-baseline)) trained on the same labeled strings. The validation split of `affiliations_unseen_institutions` contains all 272 links to CNRS.

[evaluation.md](evaluation.md) gives the results on the other test splits.

Each set-model repository holds the training folder (`training_data/`), which `train` reads as it is. It also holds the outputs of the training run in `evaluation/`: `calibration.json`, `oof.jsonl.gz`, `probabilities.jsonl.gz`, and the answers on the validation sets in `predictions/<set>.jsonl`. The answers on the four test splits, from `match run`, are in `predictions/<set>.jsonl` too. The repository of the released model also holds the candidates of these answers (`evaluation/candidates/<set>.jsonl.gz`).

## Variants

| Variant | Options | Parameters | Training time | Strings each second (padded, in training) |
|---|---|---:|---|---:|
| Released (mmBERT-base) |  | 307M | 2.8 hours | 130 |
| bge-reranker-v2-m3 | `--base BAAI/bge-reranker-v2-m3 --base-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --lr 1e-5`, rows built without `--strict` | 568M | 5.4 hours | 97 |
| EmbeddingGemma-300m | `--base google/embeddinggemma-300m --base-revision 57c266a740f537b4dc058e1b0cda161fd15afa75 --record-length 48 --max-length 2048`, rows built without `--strict` | 303M | 4.2 hours | 140 |

Exact match, precision and recall of the variants, in percent:

| Split | Variant | Exact match | Precision | Recall |
|---|---|---:|---:|---:|
| `affiliations_unseen_institutions` test | Released (mmBERT-base) | 83.7 | 94.5 | 86.8 |
|  | bge-reranker-v2-m3 | 82.3 | 93.6 | 86.9 |
|  | EmbeddingGemma-300m | 81.5 | 94.4 | 84.8 |
| `affiliations` test | Released (mmBERT-base) | 90.3 | 95.8 | 92.6 |
|  | bge-reranker-v2-m3 | 90.4 | 95.6 | 92.1 |
|  | EmbeddingGemma-300m | 90.7 | 96.6 | 92.0 |
| `affiliations` validation | Released (mmBERT-base) | 91.0 | 96.5 | 91.9 |
|  | bge-reranker-v2-m3 | 90.2 | 96.0 | 91.5 |
|  | EmbeddingGemma-300m | 91.2 | 96.4 | 92.8 |

On the test splits, the three models answer the same candidate lists, in [revision `ae056b6`](https://huggingface.co/adambuttrick/ror-set-point-set-model-mmbert-base/tree/ae056b667c8d8cb7db723f009f2fda03efb391da/evaluation/candidates) of the released model's repository. Their lexical candidates come from an index that reads the institutions file of openalex-affiliation-matcher v3.0.0 and adds a bonus for the works count. Thus the values of the released model differ a little from those in [Evaluation](#evaluation).

The rows of the two variants contain the synthetic strings that name a held-out record. Thus their values on `affiliations_unseen_institutions` are not a measure of new institutions.

To select a base model, compare exact match on the `affiliations_unseen_institutions` validation split without the CNRS strings (1,642 strings). When two results are within 1 point, select the faster model.

All times are on 1 × NVIDIA H200. EmbeddingGemma reads a maximum of 2,048 tokens, so its records have a maximum of 48 tokens. bge-reranker-v2-m3 uses the Apache-2.0 license, and EmbeddingGemma uses the Gemma terms of use.

## Limitations

- The model reads only the first 96 tokens of a string. Long strings that name many organizations lose recall.
- Organizations that a string names only with an acronym (for example "CNRS" or "INSERM") have lower recall.
- Strings that name several organizations have lower recall than strings that name one.
- When a string names a parent organization and its child, the model sometimes names only the child.
