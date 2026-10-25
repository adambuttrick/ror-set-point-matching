# Inference

This document tells how to match affiliation strings to ROR records with the trained models. The code is in `src/ror_set_point/inference/`.

| Module | Content |
|---|---|
| `lexical.py` | The lexical index: records whose names or acronyms occur in the string |
| `retriever.py` | The retriever: the records with the highest dot product. It also has a command line for search, and for the record vectors alone |
| `set_model.py` | The set model: the probability of each candidate |
| `match.py` | The matcher, which joins the three. It also has a command line |
| `bench.py` | Measures the speed of the set model, and compares its scores with the padded way |

## What you need

| Item | Source |
|---|---|
| A records folder | `data/ror_records` of the dataset, or [phase 1, step 1](training-1-data.md#step-1-make-the-records-folder). Use the ROR release of the models (v2.14 for the released models) |
| A retriever | A model folder from phase 3, or `adambuttrick/ror-set-point-retriever-e5-base` |
| A set model | A model folder from phase 4, or `adambuttrick/ror-set-point-set-model-mmbert-base` |

The lexical index and the retriever read the records of the records folder. With a folder of a later release, they also find its new records. The models did not train on these records.

## Match a file

```
uv run python -m ror_set_point.inference.match run --input strings.jsonl --records dataset/data/ror_records \
    --retriever adambuttrick/ror-set-point-retriever-e5-base \
    --set-model adambuttrick/ror-set-point-set-model-mmbert-base --record-vectors record_vectors.pt \
    --out answers.jsonl
```

The input is a JSON Lines file, or a `.txt` file with one string on each line. In a JSON Lines file, the string is in the field `text` and the key is in `id`. Use `--text-field` and `--id-field` to read other fields.

Each output row is `{"id": ..., "ror_ids": [...], "p": [...]}`. `p` gives the probability of each record in `ror_ids`. An empty `ror_ids` means that the string names no organization.

The first run encodes all 143,284 record texts. `--record-vectors` keeps these vectors in a file. The next runs read the file, when the model and the records are the same. To make the file before a run, give the retriever's command line no `--queries`:

```
uv run python -m ror_set_point.inference.retriever --model adambuttrick/ror-set-point-retriever-e5-base \
    --records dataset/data/ror_records --record-vectors record_vectors.pt
```

## Large files

`match` reads the input in chunks of 100,000 rows (`--chunk-size`). It does one chunk at a time: the lexical index, then the retriever, then the set model. Then it adds the rows of the chunk to `--out` and writes them to the disk. The rows keep the order of the input. Thus the memory does not grow with the input, and a run that stops keeps the chunks that it finished. After each chunk, `match` gives the number of rows done and the speed.

In a chunk, each distinct string is done only one time. The candidates and the answer of a string depend only on the string, so all its rows get the same answer. A string that comes again in a later chunk is done again. If a large file has many copies of the same strings, remove the copies before the run.

The `--workers` processes of the lexical index start one time, before the first chunk, and do all the chunks.

`match` does not write over a file that exists. When a run stops, give the same command again with `--resume`. It keeps the complete rows of `--out`, removes a part of a row at the end, and continues from the next row of the input. Use the same `--chunk-size`: the chunks then start at the same rows as in a run that did not stop. `--resume` needs an `--out` that is not gzip. `--chunk-size` and `--resume` work in the three modes.

## Match in two steps

Do the steps apart when you want to keep or examine the candidates.

1. Find the candidates:

   ```
   uv run python -m ror_set_point.inference.match candidates --input strings.jsonl --records dataset/data/ror_records \
       --retriever adambuttrick/ror-set-point-retriever-e5-base --record-vectors record_vectors.pt \
       --out candidates.jsonl
   ```

   Each row is `{"id": ..., "text": ..., "candidates": [...]}`.

2. Select the records:

   ```
   uv run python -m ror_set_point.inference.match answer --input candidates.jsonl --records dataset/data/ror_records \
       --set-model adambuttrick/ror-set-point-set-model-mmbert-base --out answers.jsonl
   ```

The candidates are the retriever's top 32 (`--k`), then the lexical top 10 (`--lexical-k`) that are not already there. The list stops at 42 (`--cap`). The order of the candidates is random, but it is always the same for a given string and seed (`--seed`, 1363). The set model trained on candidates in a random order. Another order can change the answer for a small number of strings.

## Use from Python

```python
from ror_set_point.inference.match import Matcher

m = Matcher("records",
            "adambuttrick/ror-set-point-retriever-e5-base",
            "adambuttrick/ror-set-point-set-model-mmbert-base",
            record_vectors="record_vectors.pt")
m.match(["Dept. of Physics, University of Tokyo, Japan"])
# -> [[("https://ror.org/057zh3y96", 0.99...)]]
```

`m.match` and `m.candidates` do each distinct string one time. The retriever keeps the vectors of all the strings of a call on the GPU, so give a large list in parts, as `match` does.

Use the parts apart for other tasks:

- `Retriever(model, records_dir).search(texts, k)` gives the top k records and their scores.
- `LexicalIndex(records_dir).top(text, k)` gives the lexical top k.
- `SetMatcher.from_pretrained(model, record_texts).probabilities(items)` gives the probability of every candidate. `items` is a list of `(string, [candidate ROR IDs])`. Give all the items in one call: the set model then makes its own batches (`batch_tokens`). `compile=False` turns off torch.compile.

## Speed and memory

| Part | Apple M4 | NVIDIA H200 |
|---|---|---|
| Lexical index (CPU) | 2.4 seconds to load. About 2,200 strings each second in one process. About 23 milliseconds for each string without the screen | 5.3 seconds to load. 1,014 strings each second in one process, 12,195 with `--workers 16` |
| Retriever, all 143,284 record texts | About 25 minutes on the GPU (`--device mps`) | Less than 1 minute |
| Set model | About 1.3 seconds for each string on the CPU | 753 strings each second, with the `gpu` extra |

The Apple M4 figures for the lexical index are for the 3,861 strings of the two validation splits. The H200 figures for the lexical index are for the 46,904 strings of the four test splits.

Use `--record-vectors`, so that the record texts are encoded only one time. On a CUDA GPU, the retriever runs in fp16 and the set model in bf16. On other devices, both run in fp32. `--device` selects the device (for example `cpu`, `cuda` or `mps`).

On 1 × NVIDIA H200, `match run --workers 16` answered the 40,000 strings of `openalex_sample` test in 95 seconds, with loading. With one process for the lexical index (the default, `--workers 1`), it took 130 seconds. For the 46,904 strings of the four test splits, the set model took 64 seconds and the retriever 4.5 seconds. The lexical index took 3.8 seconds with 16 processes. Use `--workers` on a machine with many CPU cores.

### Chunks

These runs used `match run --workers 16` on 1 × NVIDIA H200. The input was the 400,741 distinct strings of the dataset, in this order:

- the 267,105 distinct strings of the labeled splits and the teacher judgments (102 characters on average);
- the 133,636 strings of `synthetic_non_latin` (32 characters on average).

The dataset has no other distinct strings. The speed is for the chunks after the first one. The first chunk also compiles the set model, and takes 2 to 4 seconds more.

| `--chunk-size` | Strings | Seconds for each chunk | Strings each second | GPU memory | Memory of all processes |
|---:|---|---|---|---:|---:|
| 10,000 | The first 100,000 | 15.9 | 630 | 3.4 GB | 9.0 GB |
| 50,000 | 400,741 | 79.5 (labeled), 65.0 (synthetic) | 629 (labeled), 770 (synthetic) | 3.8 GB | 9.3 GB |
| 400,741 | 400,741 | 592 (one chunk) | 677 (all) | 3.8 GB | 12.2 GB |

The GPU memory is the peak of `torch.cuda.max_memory_allocated` (nvidia-smi showed 7.9 GB at most). The memory of all processes is the peak of the total PSS of the main process and the 16 lexical processes, taken every 3 seconds. The main process alone used 3.5 GB with chunks of 50,000 rows, and 6.7 GB with one chunk of 400,741 rows. Thus each 100,000 rows of a chunk add about 0.9 GB. Loading took 24 seconds with `--record-vectors`, and 47 seconds when the record vectors were encoded.

The size of the chunk does not change the speed from 10,000 rows. A large chunk uses more memory, and loses more work when a run stops. A chunk of 100,000 labeled strings (the default) takes about 2.7 minutes.

At 629 strings each second, 100 million distinct strings take about 44 hours on 1 × H200. On a machine with 8 GPUs (h200x8 has 184 vCPU), split the input into 8 files. Run one `match` for each file, with its own GPU (`CUDA_VISIBLE_DEVICES`) and its own `--out`. This should take about 5.5 hours (not measured). On Hugging Face Jobs, `jobs/match.sh` does these steps on all the GPUs of the job ([jobs.md](jobs.md#several-gpus)).

The answers do not depend on the chunks, to within the noise of the GPU. On `openalex_sample` test, chunks of 7,000 rows and one chunk gave the same `ror_ids` for all 40,000 strings. The exact match, precision and recall were the same (90.1 / 95.9 / 92.7), and one or two values of `p` differed by 0.0001. On the 400,741 strings, chunks of 50,000 rows and one chunk gave the same `ror_ids` for each string. Only 13 values of `p` differed, by 0.0001. Two runs with the same chunks can also differ in this way, because the GPU does not always add numbers in the same order.

With chunks of 7,000 rows, `--resume` gave the same file as a run that did not stop. This was true after `kill -9` in the third chunk, and after a cut in the middle of a row. An input with each of the 40,000 strings two times took 66 seconds, against 65 seconds for each string one time. The set model did 40,000 strings, and each copy got the answer of the first row with its string.

### The lexical index

The index reads the records folder. The name variants of a record are its display name, its labels and its aliases. A name with a part in brackets also gives a variant without that part. The acronyms of a record match as whole words with the same case. The cities and countries of all the locations of a record give the place bonus.

The score of a record is the best score of its variants. The place bonus adds 0.15 when the string contains one of the record's cities or countries. The score has no other parts. The lexical generator of openalex-affiliation-matcher v3.0.0 also adds a bonus for the works count of each OpenAlex institution. That count comes from the links that the OpenAlex matchers made. Such a bonus moves the candidates to the institutions that those matchers found most often. It also moves them away from small and new records. This index has no such bonus.

Records with the same score are in the order of a hash of the string and the ROR ID. Ties are frequent, because many records have the same name. On the two validation splits, the cut at 10 falls in a tie for about 21% of the strings. A tie at the cut holds about 13 records on average. The order does not depend on the order of the records in the folder. No record is first in all ties, because the order changes from string to string. For each acronym, the index keeps up to 10 records. The records in the string's city or country come first, then the others in the order of the hash.

A window of a string is one segment, or two neighbouring segments. In a window, a name variant counts only when its tokens in the window pass two tests:

- their total IDF is at least 5;
- their total IDF is at least half the total IDF of all the tokens of the variant.

The scores of the other variants are below the cut of 0.5. Before it scores the variants of a window, the index leaves out the ones that cannot pass:

1. The prefix tokens of a variant are its rarest tokens, without which its other tokens cannot reach the cut. The index takes the variants that have a prefix token in the window.
2. Arrays in numpy give the IDF of the window's tokens in each of these variants. The variants below the cut are left out. About 32 variants are left in each window, of about 4,600.
3. The index scores the variants that are left with the full rule, in Python.

The screen leaves out only variants below the cut. Thus the screen does not change the results. A check used the 3,861 strings of the `affiliations` and `affiliations_unseen_institutions` validation splits. For each string, the lexical top 100 (records and scores) was the same with and without the screen.

The table gives the speed for the 46,904 strings, in strings each second. The CPU is that of an H200 job: 23 vCPU of an Intel Xeon Platinum 8581C.

| Way | One process | `--workers 16` | `--workers 22` |
|---|---:|---:|---:|
| Without the screen: the index scores every variant that has a token of the window | 20 | 221 | 287 |
| With the screen | 1,014 | 12,195 | 16,109 |

`--workers` gives parts of 256 strings to forked processes (not on Windows). The answers do not depend on the number of processes.

### The set model on a CUDA GPU

Install the `gpu` extra (`uv sync --extra gpu`, Linux and Python 3.12). It adds flash-attn. The set model then runs in this way:

1. The encoder runs in bf16.
2. The inputs of a batch are one after the other in one sequence, with no padding. A batch holds up to 65,536 tokens (`--batch-tokens`). The longest inputs go first.
3. flash-attention 2 runs on each input. The global layers (one in three) read the whole input. The other layers read a window of 128 tokens.
4. The CPU makes the next batch while the GPU runs the current batch. One CPU core makes batches for about 120,000 strings each second.
5. torch.compile fuses the norms and the feed-forward blocks. The first batch takes a few seconds more. Use `--no-compile` to turn it off.

Without flash-attn, the set model runs in padded batches of `--batch-size` strings, in bf16 with bf16 autocast. The bge-reranker and EmbeddingGemma variants also run in this way, because their encoders are not ModernBERT.

The speed on 1 × NVIDIA H200, for the 40,000 strings of `openalex_sample` test and their candidates. Each input has 1,524 tokens on average, and 0.40 TFLOP of useful work (the matrix products, and attention in each input). The bf16 peak of the H200 is 989 TFLOP/s. The time is from the strings and the candidate IDs to the scores, after a first batch.

| Way | Strings each second | TFLOP/s | Share of the peak |
|---|---:|---:|---:|
| Padded batches of 32 strings: fp32 weights, bf16 autocast, SDPA attention (`bench --reference`) | 134 | 54 | 5.5% |
| Packed, bf16 | 665 | 268 | 27% |
| Packed, bf16, torch.compile (the default) | 753 | 304 | 31% |

The padded way is slower for these reasons:

- On these strings, padding gives the batches 1.45 times more tokens than the inputs.
- SDPA attention reads the whole input in all 22 layers, with a mask.
- Python makes the batches, one token list at a time.

Batch sizes of 32,768 to 262,144 tokens change the packed speed by less than 3%.

In the packed way, the matrix products take 37% of the GPU time, at about 740 TFLOP/s. Attention takes 34%. The global layers run near the limit of flash-attention 2, and the window layers are less efficient. Norms, the rotary embedding and the other element-wise work take 29%, near the limit of the GPU memory bandwidth.

The packed way gives the same answers as the padded way, to within the noise of bf16. The scores differ by 0.026 on average (the padded way gives the head's output in bf16). On `openalex_sample` test, 58 of the 40,000 answers change. Against the gold links, with the published candidates:

| Test split | Strings | Padded | Packed (published) | Only padded right | Only packed right |
|---|---:|---|---|---:|---:|
| `affiliations` | 1,947 | 90.0 / 95.8 / 92.5 | 90.0 / 95.7 / 92.6 | 1 | 1 |
| `affiliations_unseen_institutions` | 1,957 | 83.6 / 94.6 / 86.8 | 83.6 / 94.6 / 86.9 | 2 | 3 |
| `datacite` | 3,000 | 93.5 / 98.0 / 94.4 | 93.5 / 98.0 / 94.5 | 1 | 2 |
| `openalex_sample` | 40,000 | 90.1 / 95.9 / 92.7 | 90.1 / 95.9 / 92.7 | 20 | 25 |

The values are exact match / precision / recall, in percent. No difference is significant (exact McNemar test, p ≥ 0.5).

To measure the speed, and to compare the scores with the padded way:

```
uv run python -m ror_set_point.inference.bench --set-model adambuttrick/ror-set-point-set-model-mmbert-base \
    --records dataset/data/ror_records --input candidates.jsonl --compile on,off --reference
```

`--input` holds candidate rows (`match candidates`, or `evaluation/candidates/<set>.jsonl.gz` of the set model's repository).

The retriever's scores in fp32 and in fp16 or bf16 differ by about 0.002. Thus records near rank 32 can change places, but recall does not change. On `affiliations` validation, the released search (bf16) holds 98.7% of gold links in the top 32, and an fp32 search holds 98.8%. With the released models, `match run` gave the published answer for each of the first 60 strings of `affiliations` validation.

## Notes

- transformers can show a warning about an "incorrect regex pattern" when it loads the set model's tokenizer. Ignore it. The saved tokenizer gives the same tokens as the tokenizer of mmBERT-base (checked on 7,947 strings and record texts). Do not set `fix_mistral_regex=True`, because that can change the tokens.
- The answers of the released set model on the evaluation splits are in its repository (`evaluation/predictions/<set>.jsonl`). The candidate lists of the answers on the four test splits are in `evaluation/candidates/<set>.jsonl.gz`. The candidate lists of the answers on the two validation splits are in `training_data/eval.jsonl.gz`. To get the same answers again, give these candidates to `match answer` in the order of the file. From `eval.jsonl.gz`, give the rows as `{id: k, text: s, candidates: c}`.
