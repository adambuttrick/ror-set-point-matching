# Run on Hugging Face Jobs

This document tells how to run the pipeline on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs). The pipeline has four steps. The scripts are in `jobs/`.

| Script | Step | Hardware |
|---|---|---|
| `ror_data.sh` | 1. Get a ROR release from Zenodo and make the records folder | CPU |
| `index.sh` | 2. Encode the record texts with the retriever (the index) | GPU |
| `match.sh` | 3. Match a file of strings, on one GPU or on all the GPUs of the job | GPU |
| `evaluate.sh` | 4. Score answers against a labeled split | CPU |
| `pipeline.sh` | Steps 1 to 4 in one job | GPU |
| `verify.sh` | Steps 1 to 4 on the test splits, compared with the published results | GPU |
| `submit.sh` | Starts a step as a job. It runs on your computer | |

`lib.sh` holds the parts that the steps share. `fetch.py` downloads the inputs and checks them, `shards.py` runs match on several GPUs, and `check.py` compares outputs with the published files.

## What you need

- The `hf` command line, logged in (`hf auth login`), with an account that can run Jobs. The scripts were checked with `hf` 2.2.0.
- `HF_TOKEN` in the environment. The models and the dataset are private, so `submit.sh` gives the token to each job as a secret.
- A bucket for the outputs. `submit.sh` uses `adambuttrick/ror-set-point-matching-jobs`, or the bucket in `BUCKET`:

  ```
  hf buckets create <your user>/ror-set-point-matching-jobs --private
  export BUCKET=<your user>/ror-set-point-matching-jobs
  ```

## How a job is made

`jobs/submit.sh STEP [options of hf jobs run]` copies the code into a new folder. Then it starts `bash /code/jobs/STEP.sh` in a job with these volumes:

| Path in the job | Content | Access |
|---|---|---|
| `/code` | `src`, `jobs`, `pyproject.toml`, `uv.lock`, `README.md` and `.python-version`. `hf jobs run` uploads them to the bucket `<your user>/jobs-artifacts` | Read |
| `/bucket` | The bucket for the outputs | Read and write |
| `/dataset` | The dataset `adambuttrick/ror-set-point-matching-train-validation-test` | Read |

The image is `ghcr.io/astral-sh/uv:0.12.1-python3.12-trixie` (`IMAGE` changes it). Each job has its own copy of the code. Thus a change to the code does not change a job that has started.

The steps on a CPU run the code with the Python of the image, because they need only the standard library. The steps on a GPU first make the project's environment with `uv sync --locked --extra gpu`. This takes 30 to 50 seconds. They stop when flash-attn does not import, because the set model would then run 5.6 times slower.

The options of the steps are environment variables (`-e NAME=value`). These options are the same in all steps:

| Variable | Default | Content |
|---|---|---|
| `ROR_VERSION` | `v2.14` | The ROR release, for example `v2.15`. `latest` is the newest release on Zenodo |
| `RECORDS` | `/bucket/records/$ROR_VERSION` | The records folder |
| `DOWNLOADS` | `/bucket/downloads` | The ROR zip and `institutions.jsonl.gz` |
| `INSTITUTIONS` | `$DOWNLOADS/institutions.jsonl.gz` | The institutions file of openalex-affiliation-matcher v3.0.0. Step 1 makes `openalex_to_ror.json` (OpenAlex institution IDs to ROR IDs) from it |
| `RETRIEVER` | `adambuttrick/ror-set-point-retriever-e5-base` | A retriever folder or Hub repository |
| `SET_MODEL` | `adambuttrick/ror-set-point-set-model-mmbert-base-mask` | A set-model folder or Hub repository |
| `VECTORS` | `/bucket/index/<name of RECORDS>/<name of RETRIEVER>.pt` | The index |

After the steps, the bucket holds these files:

```
downloads/v2.14-2026-10-06-ror-data.zip
downloads/institutions.jsonl.gz
records/v2.14/                                  records.jsonl.gz, withdrawn.json, openalex_to_ror.json, manifest.json
index/v2.14/ror-set-point-retriever-e5-base.pt
runs/<name>/                                    answers, candidates and scores
```

## Step 1: the records folder

```
jobs/submit.sh ror_data --flavor cpu-basic --timeout 30m
```

The step finds the release on Zenodo, downloads its zip, and checks the md5 that Zenodo gives. For v2.14, it also checks the sha256 of the JSON file ([phase 1](training-1-data.md#inputs)). It downloads `institutions.jsonl.gz` and checks its sha256. Then it runs `ror_set_point.training.data.records`, and copies the folder to `RECORDS`. `manifest.json` is copied last, so a folder with a manifest is complete.

For v2.14, the step also compares the folder with `data/ror_records` of the dataset (`CHECK=1`). The rows of the 143,284 records and the two maps must be equal. The step does nothing when `RECORDS/manifest.json` exists. `FORCE=1` makes the folder again.

For a newer release, give `-e ROR_VERSION=v2.15` or `-e ROR_VERSION=latest`. `latest` is resolved to a version when the job starts, and the folder gets the name of that version. Use the same `ROR_VERSION` in the next steps. The released models trained on v2.14. With a newer release, the lexical index reads the names of the new records. The retriever and the set model read their texts.

## Step 2: the index

```
jobs/submit.sh index --flavor l4x1 --timeout 30m
```

The step runs `python -m ror_set_point.inference.retriever --records ... --record-vectors ...` with no `--queries`. It encodes the 143,284 record texts and writes `VECTORS` (210 MB). The file holds the name of the retriever and the sha256 of the records file. When they are the same, the step reads the file and does not encode the records again. Thus the step is fast when the index exists.

`match` makes the index too, when `VECTORS` does not exist. Thus step 2 is necessary only when you want the index before you match.

## Step 3: match

Put the file into the bucket, then match it:

```
hf buckets cp strings.jsonl hf://buckets/adambuttrick/ror-set-point-matching-jobs/inputs/strings.jsonl
jobs/submit.sh match --flavor h200 --timeout 3h \
    -e INPUT=/bucket/inputs/strings.jsonl -e OUT=/bucket/runs/my-run/answers.jsonl
```

Then download the answers:

```
hf buckets cp hf://buckets/adambuttrick/ror-set-point-matching-jobs/runs/my-run/answers.jsonl answers.jsonl
```

The input can also be in a dataset repository. Mount it with `-v hf://datasets/<org>/<name>:/input`, and give `-e INPUT=/input/<file>`.

| Variable | Default | Content |
|---|---|---|
| `INPUT` | | JSON Lines (`.jsonl` or `.jsonl.gz`, fields `id` and `text`), or `.txt` with one string on each line |
| `OUT` | | The answers. JSON Lines, not gzip |
| `MODE` | `run` | `run`, `candidates` or `answer` ([inference.md](inference.md#match-in-two-steps)) |
| `WORKERS` | The vCPUs for each GPU, minus 1 | Processes for the lexical index, for each GPU |
| `MATCH_ARGS` | | Other options of match, for example `--text-field affiliation --chunk-size 50000` |

The step uses the vCPUs of the job's cgroup, not `nproc`. On an h200 job, `nproc` gives the 224 CPUs of the host.

### When a job stops

`match.sh` always gives `--resume` to match. match writes the rows of each chunk (100,000 rows) to `OUT`, and syncs the file. The rows that are synced stay in the bucket when the job stops, also at the timeout. Thus the same command again continues `OUT` from the next row of the input.

`--attempts N` of `hf jobs run` starts the job again when it fails, up to N times in all. Each attempt continues `OUT`. A timeout ends the job with no other attempt. Thus give a `--timeout` that is longer than the run, and `--attempts 3` for other stops. When the timeout ends a job, give the same command again.

In a test, `pipeline.sh` matched the 40,000 strings of `openalex_sample` test, as a `.txt` file in the bucket. The job ran on 1 × NVIDIA L4 with `--chunk-size 2000` and `--timeout 5m`. The timeout ended the job after 14,000 rows. The same command with `--timeout 30m` continued from row 14,001 and wrote the other 26,000 rows. `OUT` had 40,000 rows, with the line numbers 0 to 39,999 as ids, in order.

### Several GPUs

On a flavor with several GPUs, for example `h200x8`, `match.sh` runs one match process on each GPU (`jobs/shards.py`):

1. It checks the index once, so that the processes do not each encode the records.
2. It divides the input into parts of consecutive rows, one part for each GPU. A `.txt` input becomes JSON Lines rows with the line number as `id`, as match gives it.
3. Process i runs on GPU i (`CUDA_VISIBLE_DEVICES`), with its part as `--input` and `OUT.parts/part-i.jsonl` as `--out`, with `--resume`.
4. When all parts are complete, it joins them into `OUT` in the order of the input, and removes `OUT.parts`.

`OUT.parts/plan.json` gives the input, its number of rows and the number of parts. When a job ends just after the join, the bucket can keep some files of `OUT.parts`. Parts with no plan are thus removed when a run starts, and `OUT.parts` is removed when `OUT` is complete.

When a process fails, the others go on. The next run with the same number of GPUs continues each part. With another number of GPUs, the step stops: remove `OUT.parts`, or use the first flavor again. When `OUT` exists from a run with one process and is not complete, the step also stops. Continue that run on one GPU.

On 4 × NVIDIA L4 (`l4x4`), `verify.sh` answered the 40,000 strings of `openalex_sample` test in 201 seconds, with loading. One L4 answers about 68 strings each second. The answers of the parts are the answers of one process, to within the noise of the GPU (see [Check the pipeline](#check-the-pipeline)).

## Step 4: score

```
jobs/submit.sh evaluate --flavor cpu-basic --timeout 30m \
    -e GOLD=/dataset/data/affiliations/test.jsonl -e ANSWERS=/bucket/runs/my-run/affiliations_test.jsonl
```

The step runs `ror_set_point.evaluation.score` ([evaluation.md](evaluation.md#score-answer-files)) and writes the Markdown next to the answers (`<answers>.scores.md`). `REFERENCE` adds other answers of the same strings, and compares the two with the exact McNemar test. `SCORE_ARGS` gives more options, for example `--bootstrap 2000`.

## All steps in one job

```
jobs/submit.sh pipeline --flavor h200 --timeout 3h --attempts 3 \
    -e INPUT=/bucket/inputs/strings.jsonl -e OUT=/bucket/runs/my-run/answers.jsonl
```

`pipeline.sh` runs steps 1, 2 and 3, and step 4 when `GOLD` is given. A step whose output exists is skipped or continued. Thus the same command again, or another attempt, continues the job.

## Check the pipeline

```
jobs/submit.sh verify --flavor h200 --timeout 1h -e RUN=/bucket/runs/verify-h200 \
    -v hf://adambuttrick/ror-set-point-set-model-mmbert-base-mask:/set_model
```

`verify.sh` checks the code and the released models against the published results. It makes the records folder of v2.14 and compares it with the dataset, and it makes the index. Then, for each of the four test splits, it does these steps:

1. `match candidates`. The candidate lists are compared with `evaluation/candidates/` of the set model's repository.
2. `match run`. The answers are compared with `evaluation/predictions/` of that repository.
3. The scores of the answers, against the gold links, with the published answers as the reference.

`RUN/summary.md` holds the results. Use the check after a change to the code, the image or the versions of the packages.

On 1 × NVIDIA H200 (1 November 2026):

| Test split | Strings | Same candidate list | Same answer | This run and published |
|---|---:|---:|---:|---|
| `affiliations` | 1,947 | 100% | 1,947 | 90.4 / 96.1 / 92.6 |
| `affiliations_unseen_institutions` | 1,957 | 100% | 1,957 | 84.7 / 94.6 / 87.4 |
| `datacite` | 3,000 | 100% | 3,000 | 93.1 / 97.8 / 94.4 |
| `openalex_sample` | 40,000 | 100% | 40,000 | 90.2 / 96.0 / 92.7 |

The values are exact match / precision / recall, in percent. The candidate lists are equal, in the same order, and the answers are equal.

On 4 × NVIDIA L4, with one part of each split on each GPU, with [ror-set-point-set-model-mmbert-base](https://huggingface.co/adambuttrick/ror-set-point-set-model-mmbert-base) (the set model without the mention mask) and its published answers:

| Test split | Same candidate set | Same candidate list | Same answer | This run | Only this run right | Only published right |
|---|---:|---:|---:|---|---:|---:|
| `affiliations` | 89.4% | 26.0% | 1,934 | 90.0 / 95.6 / 92.7 | 5 | 6 |
| `affiliations_unseen_institutions` | 89.2% | 28.7% | 1,930 | 83.3 / 94.6 / 86.6 | 5 | 12 |
| `datacite` | 93.4% | 36.7% | 2,980 | 93.3 / 97.9 / 94.4 | 5 | 13 |
| `openalex_sample` | 91.3% | 28.7% | 39,736 | 90.1 / 95.9 / 92.7 | 93 | 119 |

No difference is significant (p ≥ 0.08). The candidate lists differ because the retriever's scores change a little with the GPU, and with the other strings of a batch. Two records with almost the same score can then change places in the top 32. The candidates are put in a random order that depends on the string and on the order of the list ([inference.md](inference.md#match-in-two-steps)). Thus a change of two places gives another order.

On the `affiliations` test split, these pairs of runs gave the same candidates for this share of the strings:

| Runs | Same candidate set | Same candidate list |
|---|---:|---:|
| 1 × L4 and the H200 | 89.9% | 26.8% |
| 1 × L4 and 4 × L4 | 96.0% | 74.4% |

On `openalex_sample` test, `verify.sh` on 1 × L4 gave the same answer as the H200 for 99.35% of the strings, with the set model without the mention mask. Use an H200 to get the published candidates again.

## Times and costs

The times of the checks above, on 24 and 25 October 2026. The cost is the running time at the price of the flavor (`hf jobs hardware`). A job also waits for a machine before it runs: about 10 seconds for `cpu-basic` and `h200`, and about 100 seconds for the L4 flavors.

| Job | Flavor | Running time | Cost |
|---|---|---:|---:|
| `ror_data`, v2.14, with the comparison | `cpu-basic` | 42 s | < $0.01 |
| `ror_data`, v2.15 (145,275 records) | `cpu-basic` | 62 s | < $0.01 |
| `index`, v2.15 | `l4x1` | 132 s | $0.03 |
| `verify` | `h200` | 389 s | $0.54 |
| `verify` | `l4x4` | 647 s | $0.68 |
| `verify` | `l4x1` | 1,065 s | $0.24 |
| `pipeline`, 40,000 strings, in two jobs | `l4x1` | 288 s + 506 s | $0.18 |

The parts of a job on a GPU:

| Part | `h200` | `l4x1` |
|---|---:|---:|
| `uv sync --locked --extra gpu` | 32 s | 50 s |
| Index of 143,284 records, with loading | 32 s | 68 s (v2.15) |
| Loading of match run with the index (lexical index, retriever, set model) | about 33 s | about 45 s |
| Set model | 623 strings each second | 65 strings each second |
| Cost of the set model for 1 million strings | $2.23 | $3.42 |

The H200 is the fastest, and the cheapest for each string. A flavor with several H200s costs the same for each GPU, and has the same vCPUs for each GPU. Thus it should divide the time by the number of GPUs at the same cost (not measured). For a few thousand strings, `l4x1` is cheaper, because each job pays for its setup and loading.

## Watch the jobs

| Command | Effect |
|---|---|
| `hf jobs ps` | The running jobs (`-a` also gives the jobs that ended) |
| `hf jobs logs JOB_ID` | The log. `-f` follows it |
| `hf jobs inspect JOB_ID` | The settings, the stage and the times |
| `hf jobs cancel JOB_ID` | Stops the job |
| `hf buckets list <bucket> -R` | The files in the bucket |

`submit.sh` follows the log until the job ends, and `--detach` returns at once. The log that `hf jobs run` follows can miss the last lines of a job, and it shows only the first attempt. On a slow network, it can stop with an error while the job goes on. `hf jobs logs` gives the full log of the last attempt, and `hf jobs inspect` gives the stage.

## Notes

- Zenodo can be slow. `fetch.py` tries each download 6 times. The zip stays in `DOWNLOADS`, so the next job does not download it again.
- Two jobs can write to the same bucket at the same time, but not to the same `OUT`.
- The copies of the code stay in the bucket `<your user>/jobs-artifacts`. Each one is about 0.7 MB. Remove them with `hf buckets remove`.
- `BUCKET`, `DATASET` and `IMAGE` in the environment of `submit.sh` change the bucket, the dataset and the image.
