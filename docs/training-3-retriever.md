# Training phase 3: the retriever

The retriever finds candidate ROR records for an affiliation string. It is a dual encoder: one shared encoder changes strings and record texts into vectors of length 1. A high dot product means a good candidate. The matcher keeps the top 32 records of all 143,284.

The released model is [ror-set-point-retriever-e5-base](https://huggingface.co/adambuttrick/ror-set-point-retriever-e5-base).

## The model

| Property | Value |
|---|---|
| Base model | [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base), revision `d128750597153bb5987e10b1c3493a34e5a4502a` (MIT) |
| Pooling | Mean of the token states, then length 1 |
| Prompts | `query: ` for strings, `passage: ` for record texts |
| Input | Record texts: 128 tokens. Strings: 96 tokens in training, 512 tokens in search |
| Format | A sentence-transformers model in fp16, with the prompts `query` and `passage` |

Long strings often name several organizations. The longer limit in search lets the model read all of them.

## Training data

| Part | Source | Hard negatives | Rows |
|---|---|---|---:|
| Teacher | `teacher_judgments` train, without the strings that the exclusion rule removes (167,533 strings). One row for each record with p ≥ 0.5 | The records of the same string with p < 0.1. 18,131 strings have no positive and give no row | 304,626 |
| Labeled | `affiliations_unseen_institutions` train (16,448 strings). Each link is used 3 times | Candidates that are not labels: from a file, or the lexical top 10 | 49,878 |
| Synthetic | `synthetic_non_latin` (130,882 strings). The 2,871 strings that name a held-out record are not used (`--strict`) | Mined with a first retriever (see the procedure) | 136,184 |

In each step, each string has its positive record and 3 hard negatives. Training draws the 3 at random from the hard negatives of the row. When the row has fewer than 3, random records fill the list. All other records of the batch are also negatives, except the other labels of the string.

## Procedure

The procedure trains two retrievers. The first one has no synthetic strings. Its search gives the hard negatives for the synthetic strings. The second one is the released model.

The commands use the dataset (in the folder `dataset`). They also use this shell variable:

```
U=dataset/data/affiliations_unseen_institutions
```

The `--eval` options name the evaluation splits. `build` writes their strings as queries, and leaves out training strings that are equal to them.

1. Make the data for the first retriever:

   ```
   uv run python -m ror_set_point.training.retriever.build --records dataset/data/ror_records \
       --teacher dataset/data/teacher_judgments/train.jsonl --labelled $U/train.jsonl \
       --held-out $U/held_out_records.jsonl \
       --eval unseen_validation $U/validation.jsonl --eval unseen_test $U/test.jsonl \
       --eval affiliations_validation dataset/data/affiliations/validation.jsonl \
       --out retriever_data_first
   ```

2. Train the first retriever:

   ```
   uv run python -m ror_set_point.training.retriever.train --data retriever_data_first --out retriever_first \
       --base-revision d128750597153bb5987e10b1c3493a34e5a4502a --k 0
   ```

3. Search the synthetic strings with the first retriever. Keep the top 64 records and the score of each label:

   ```
   uv run python -m ror_set_point.inference.retriever --model retriever_first/model --records dataset/data/ror_records \
       --queries dataset/data/synthetic_non_latin/train.jsonl --key-field line --positives-field ror_ids \
       --k 64 --out synthetic_top64.jsonl.gz
   ```

4. Mine the hard negatives:

   ```
   uv run python -m ror_set_point.training.retriever.mine --synthetic dataset/data/synthetic_non_latin/train.jsonl \
       --records dataset/data/ror_records --search synthetic_top64.jsonl.gz --out mined.jsonl.gz -v
   ```

   The step reads the 64 records of each synthetic string in rank order. It drops these records:

   - the labels of the string;
   - each record that is related to a label (parent, child, related, predecessor or successor);
   - each record with the same name as the source record;
   - each record with a score of 0.95 × the score of the source record, or more. These records are probably named by the string, but have no label.

   The first 10 records that remain are the negatives.

5. Make the data for the second retriever:

   ```
   uv run python -m ror_set_point.training.retriever.build --records dataset/data/ror_records \
       --teacher dataset/data/teacher_judgments/train.jsonl --labelled $U/train.jsonl \
       --held-out $U/held_out_records.jsonl \
       --eval unseen_validation $U/validation.jsonl --eval unseen_test $U/test.jsonl \
       --eval affiliations_validation dataset/data/affiliations/validation.jsonl \
       --synthetic dataset/data/synthetic_non_latin/train.jsonl --synthetic-negatives mined.jsonl.gz --strict \
       --out retriever_data
   ```

6. Train the second retriever. The script then searches every query of the data folder and writes the top 64 records of each to `retriever/topk.jsonl.gz`. Phase 4 reads this file.

   ```
   uv run python -m ror_set_point.training.retriever.train --data retriever_data --out retriever \
       --base-revision d128750597153bb5987e10b1c3493a34e5a4502a --no-gradient-checkpointing
   ```

Before a full run, do a smoke test with `--limit 3000 --steps 20 --log-every 5`.

The training script saves the model, loads it again, and compares its vectors with the vectors of the training encoder. `summary.json` gives the smallest cosine (`reload_check`). It must be near 1.

## Settings

| Setting | Value |
|---|---|
| Loss | InfoNCE at temperature 0.02 over all records of the batch |
| Batch | 128 strings, each with 1 positive and 3 hard negatives |
| Epochs | 1 (3,833 steps for the released model) |
| Optimizer | AdamW, learning rate 2e-5, weight decay 0.01 |
| Schedule | Linear warm-up for 5% of the steps, then linear decay |
| Precision | bf16 autocast |
| Gradient clipping | 1.0 |
| Seed | 1363 |

Gradient checkpointing (on by default) saves memory. The H200 does not need it.

## Hardware and time

| Model | Rows | Steps | Hardware | Training time |
|---|---:|---:|---|---|
| First retriever | 354,516 | 2,769 | 1 × NVIDIA L4, gradient checkpointing on | 1.8 hours |
| Released retriever | 490,704 | 3,833 | 1 × NVIDIA H200 | 17 minutes |

The search of all 143,284 records and about 320,000 strings takes a few minutes on the same GPU.

## Evaluation

Measure the recall of the candidates with `ror_set_point.evaluation.recall` (see [evaluation.md](evaluation.md)):

```
uv run python -m ror_set_point.evaluation.recall --gold $U/test.jsonl --records dataset/data/ror_records --train $U/train.jsonl \
    --candidates retriever retriever/topk.jsonl.gz --set unseen_test --lexical
```

Recall of gold links in percent, on the `affiliations_unseen_institutions` test split (1,957 strings, 2,620 links). "New" links go to records that no string of the train split names. "Known" links go to the other records. "Held out" links go to the 1,255 held-out institutions. The figures come from the search of the test splits on one NVIDIA H200, in the released model's repository (`search/topk_test_splits.jsonl.gz`).

| Candidates | Records for each string | All links | Known | New | Non-Latin strings | Held out |
|---|---:|---:|---:|---:|---:|---:|
| Lexical top 10 | 8.9 | 82.5 | 75.6 | 84.6 | 76.3 | 86.6 |
| Retriever top 1 | 1.0 | 56.3 | 33.4 | 63.5 | 79.7 | 67.6 |
| Retriever top 10 | 10.0 | 93.7 | 90.6 | 94.7 | 97.0 | 96.2 |
| Retriever top 32 | 32.0 | 96.3 | 93.7 | 97.2 | 97.9 | 98.2 |
| Retriever top 64 | 64.0 | 97.6 | 95.2 | 98.3 | 98.3 | 98.9 |
| Retriever top 32 + lexical top 10 (the matcher's candidates) | 37.5 | 97.6 | 96.5 | 98.0 | 97.9 | 98.9 |
| The candidates of the baseline (lexical, dense, acronym and alias generators) | 18.0 | 94.6 | 95.6 | 94.3 | 85.2 | 95.7 |

On the other test splits (all links):

| Split | Strings | Lexical top 10 | Retriever top 32 | Retriever top 32 + lexical top 10 |
|---|---:|---:|---:|---:|
| `affiliations` test | 1,947 | 81.1 | 98.7 | 98.8 |
| `datacite` test | 3,000 | 96.6 | 99.0 | 99.4 |
| `openalex_sample` test | 40,000 | 86.4 | 98.6 | 98.9 |

The validation split contains all 272 links to CNRS. Most of these strings name CNRS only with its acronym, inside the name of a laboratory. The retriever finds these links less often. Its recall at 32 on the validation split is 92.8.

## Variants

Each variant has its own model repository. The table gives the recall of the top 32 records, in percent.

| Variant | Change to the procedure | `affiliations_unseen_institutions` test: all links | Known | New | Non-Latin strings | `affiliations` test | `openalex_sample` test |
|---|---|---:|---:|---:|---:|---:|---:|
| Released (`ror-set-point-retriever-e5-base`) |  | 96.3 | 93.7 | 97.2 | 97.9 | 98.7 | 98.6 |
| No synthetic strings (`-no-synthetic`) | The first retriever, searched at 512 tokens | 96.0 | 93.1 | 96.8 | 96.6 | 98.4 | 98.5 |
| EmbeddingGemma (`ror-set-point-retriever-embeddinggemma-300m`) | Step 6 with the options below, on the rows of step 5 without `--strict` | 97.4 | 95.3 | 98.0 | 98.7 | 99.3 | 99.0 |

The rows of the EmbeddingGemma variant contain the synthetic strings that name a held-out record. Thus its values on `affiliations_unseen_institutions` are not a measure of new institutions.

The EmbeddingGemma variant uses these training options:

```
--base google/embeddinggemma-300m --base-revision 57c266a740f537b4dc058e1b0cda161fd15afa75 \
--loader sentence-transformers --query-prompt "task: search result | query: " \
--record-prompt "title: none | text: " --no-gradient-checkpointing
```

EmbeddingGemma uses the Gemma terms of use. It keeps its own pooling and dense layers, and it saves in bf16. It trained in 50 minutes on 1 × NVIDIA H200, 3 times longer than the e5 model.

## Files of the released models

Each retriever repository holds its training folder (`training_data/`), as `build` writes it. `train` reads this folder as it is. Each repository also holds a search output (`search/topk.jsonl.gz`) in the format that `train` writes:

- For `ror-set-point-retriever-e5-base`, it covers every query, with 64 records for each. Phase 4 reads it with `--retrieved`.
- For the variants, it covers the evaluation sets only (100 records for each string).

The repository of the first retriever (`-no-synthetic`) also holds the search of step 3 (`search/synthetic_top64.jsonl.gz`).

The query sets have the names `teacher`, `labelled`, `synthetic`, `unseen_validation`, `unseen_test` and `affiliations_validation`. `search/topk_test_splits.jsonl.gz` of the released retriever holds the test splits `affiliations_test`, `datacite_test` and `openalex_sample_test`.

Rows that this code makes are different from the released rows in these ways:

- The released rows use, as hard negatives of the labeled strings, the candidates of the baseline matcher (its lexical, dense, acronym and alias generators). Give such candidates to `build` with `--labelled-negatives` (rows `{id, ror_ids}`). Without that option, `build` uses the lexical top 10.
- The released rows contain 8 strings (4 labeled and 4 synthetic) that are equal to `affiliations` validation strings. This code removes them.
- The released negatives of the synthetic strings were mined with the labels before the label rules of phase 2 (step 5).

With the same candidates and negatives, `build` makes the released rows in the same order, without those 8 strings.
