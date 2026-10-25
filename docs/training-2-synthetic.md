# Training phase 2: synthetic strings in non-Latin scripts

Many affiliation strings are in a local script, for example Chinese, Russian or Arabic. Most ROR records have few or no names in these scripts, and the labeled data has few such strings for most institutions. In this phase, a language model writes affiliation strings in the local script for ROR records in 51 countries. The labels come from the record, so they are correct by construction. A blind check by a second model family measures the error rate.

The dataset contains the result as the config `synthetic_non_latin` (133,757 strings). Do this phase to make the strings again, or to make strings for a different ROR release.

## Inputs

- A records folder: `data/ror_records` of the dataset, or the output of [phase 1](training-1-data.md).
- The [codex CLI](https://github.com/openai/codex), logged in. The released strings came from the model `gpt-6.1-sol` with reasoning effort `low`.

## Countries and scripts

The records are the active and inactive records whose first place is in one of these countries:

| Script group | Countries |
|---|---|
| Han, kana and Hangul | CN, TW, HK, MO, JP, KR, KP |
| Cyrillic | RU, UA, BY, KZ, KG, TJ, MN, BG, RS, MK |
| Arabic script | IR, AF, EG, SA, AE, IQ, JO, LB, SY, PS, KW, QA, BH, OM, YE, LY, SD, DZ, MA, TN, MR |
| Other | TH, LA, KH, MM, GR, CY, IL, GE, AM, ET, BD, NP, LK |

`languages()` in `src/ror_set_point/training/synthetic/generate.py` gives the language for each country.

## Step 1: make the batches

```
uv run python -m ror_set_point.training.synthetic.generate batches --records dataset/data/ror_records --dir synthetic
```

The records go in sha256 order of the ROR ID, 40 to a batch. Thus a part of the batches is a uniform sample. For ROR v2.14, this gives 32,605 records in 816 batches.

Each record in a batch has its names, place, types, language and related records (a maximum of 8 for each relation).

## Step 2: run the language model

Do a test on two batches first:

```
uv run python -m ror_set_point.training.synthetic.generate run --records dataset/data/ror_records --dir synthetic --limit 2 --workers 2 -v
```

Then run all batches:

```
uv run python -m ror_set_point.training.synthetic.generate run --records dataset/data/ror_records --dir synthetic --workers 8 -v
```

Each batch is one `codex exec` call. The call has no tools: no shell, no shell snapshots, no skills and no web search. The sandbox is read-only, and the session is not kept. The prompt holds all the information that the model needs. The output must agree with a JSON schema.

For each record, the model writes 4 strings, one of each form:

| Form | Content |
|---|---|
| `full` | The full local name, usually with the city or address |
| `unit` | A department, faculty or laboratory inside the organization, then the organization |
| `short` | A usual local abbreviation or short name |
| `mixed` | A mixed or transliterated form, for example a local-script name with an English unit |
| `other_script` | Only when a second script is usual in the country, for example Russian for a Kazakh organization |

The model lists the related records that each string names, and gives a confidence (`high` or `low`). It gives `low` when it does not know an established local name and must translate.

The script checks each result. A valid result has these properties:

- each record of the batch occurs one time;
- each record has 4 or 5 strings, with the forms `full`, `unit`, `short` and `mixed`;
- no string is empty;
- each string names only related records of its own record.

A batch with a valid result is not run again. Thus, if a run stops, run the same command again to continue. A failed call is tried again 3 times.

With 8 calls at a time and about 2 minutes for each call, the 816 batches take about 3.5 hours.

## Step 3: collect the strings

```
uv run python -m ror_set_point.training.synthetic.generate collect --records dataset/data/ror_records --dir synthetic -v
```

The step writes `synthetic/synthetic.jsonl` in the format of `synthetic_non_latin`: `id`, `text`, `ror_ids`, `ror_id` (the source record), `form`, `confidence` and `language`. The labels (`ror_ids`) are the source record, then each related record that the model says the string names. A named parent organization is labeled, as the annotation guidelines tell.

| Count | Value |
|---|---:|
| Strings | 133,757 |
| Strings with confidence high | 121,927 |
| Strings with confidence low | 11,830 |
| Strings of each of the forms `full`, `unit`, `short` and `mixed` | 32,605 |
| Strings of the form `other_script` | 3,337 |
| Strings that name a related record | 7,237 |
| Strings in Han, kana and Hangul | 92,072 |
| Strings in Cyrillic | 16,705 |
| Strings in Arabic script | 11,731 |
| Strings in other scripts | 13,249 |

## Step 4: check a sample

```
uv run python -m ror_set_point.training.synthetic.check sample --dir synthetic --synthetic synthetic/synthetic.jsonl \
    --records dataset/data/ror_records
```

The step takes 60 strings from each script group (240 in all). It writes `synthetic/check/items_<group>.json` for the judges. The judges see the string, the record (names, place, types, status and related records) and the names of the labeled records. They do not see the form, the confidence or the batch.

Give each file to a judge of a different model family than the generator, or to a person, with the annotation guidelines v1.2. The judge writes `synthetic/check/verdicts_<group>.jsonl`, one row for each item:

| Field | Values | Meaning |
|---|---|---|
| `id` | the item id | |
| `org` | `right`, `wrong`, `unsure` | `wrong`: the string names another organization, or a reader in that country would not take it for this one |
| `form` | `natural`, `unnatural` | `unnatural`: no researcher there would write it, for example a word-for-word translation of an English name that has an established local name |
| `labels` | `right`, `missing`, `extra`, `both` | `missing`: the string names a ROR organization that has no label. `extra`: a label that the string does not name, or that the guidelines do not allow |
| `note` | text | |

Then score the verdicts:

```
uv run python -m ror_set_point.training.synthetic.check score --dir synthetic --synthetic synthetic/synthetic.jsonl \
    --records dataset/data/ror_records
```

The step prints each rate for each group with a Wilson 95% interval, and weighted by the share of strings in each group. Use the strings only when the weighted rate of wrong organization is 10% or less.

The check of the released strings gave these results:

| Group | Strings | Wrong organization | Unnatural form | Wrong labels |
|---|---:|---:|---:|---:|
| Han, kana and Hangul | 92,072 | 0 of 60 | 2 of 60 | 2 of 60 |
| Cyrillic | 16,705 | 0 of 60 | 14 of 60 | 5 of 60 |
| Arabic script | 11,731 | 2 of 60 | 18 of 60 | 4 of 60 |
| Other | 13,249 | 0 of 60 | 15 of 60 | 0 of 60 |
| Weighted | 133,757 | 0.3% | 10.3% | 3.9% |

Most unnatural forms are the `mixed` form, which mixes an English unit with a local-script name. These strings are kept, because they occur in real affiliations too. The label errors had two patterns. Step 5 corrects them.

## Step 5: apply the label rules

```
uv run python -m ror_set_point.training.synthetic.fix_labels --synthetic synthetic/synthetic.jsonl --records dataset/data/ror_records \
    --out synthetic/synthetic_fixed.jsonl -v
```

| Rule | Strings | What changes | Count for the released strings |
|---|---|---|---|
| 1 | The source record is not healthcare, and the string names the Russian Academy of Sciences (РАН, or "Российская академия наук") | The Academy is added as a label. A named Siberian, Ural or Far Eastern branch is also added | 898 strings, 1,067 labels added |
| 2 | The source record is healthcare | Labels whose record is not healthcare are dropped. An organization that is only a part of a hospital's name is not labeled | 1,931 strings, 1,955 labels dropped |

The output has the same rows in the same order. This file is the `synthetic_non_latin` config, and phases 3 and 4 read it.

Rule 2 is not always correct. A read of 40 healthcare strings with extra labels found 36 correct drops and 3 correct keeps. In 1 string, the rule dropped a parent agency that the string names inside the name of a research institute.

## Ids

`generate collect` gives each string the id `<batch>:<record key>:<n>`, as in the dataset. `check sample` uses this id to find the record of the string. Phases 3 and 4 identify a synthetic string by its line number, so the order of the rows must not change.
