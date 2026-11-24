# Published research data

Fetched and verified on 2026-11-21. Raw files are used **as published**: no
cleaning, re-encoding, score correction, row filtering, or overwriting. Derived
data belongs elsewhere and is outside this task. Raw data and reference clones
are ignored by Git. `data/MANIFEST.json` records every raw file, including archives,
extracted members, dataset cards, and metadata files; reference-only Git clones
have their commit and licence recorded under `sources`.

## Reproduce

Use Python 3.11 or newer and Git. The inspected environment uses Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -I scripts/fetch_data.py
.venv/bin/python -I scripts/fetch_data.py
.venv/bin/python -I scripts/verify_data.py
.venv/bin/python -I scripts/test_data_pipeline.py
```

The five direct dependencies are pinned in `requirements.txt`. To fetch one
dataset, use `--only between_help_and_harm`, `--only counselbench`,
`--only mentalalign`, or `--only mentalhealthbench`.

The committed manifest locks Git/Hugging Face revisions and SHA-256 checksums.
The fetcher does not advance these revisions. Existing matching files are skipped;
missing files are retrieved at the locked revision. A checksum conflict stops
that dataset and preserves the existing bytes. An upstream archive with changed
bytes is rejected against the recorded checksum. Do not delete or regenerate the
manifest to hide a mismatch. The download dates describe the original acquisition
recorded here, and are preserved on repeat runs.

Downloads are staged separately under ignored `data/.staging/`, checked, and
installed without replacing existing raw files. ZIP members retain their paths
and bytes; traversal paths, symlinks, and duplicate names are rejected. Archive
metadata such as `__MACOSX/` is retained too. No downloaded scripts, notebooks,
dependency lists, or setup files are executed. The entry points automatically
restart with `-I` if invoked without it. All raw-file readers are in `scripts/`.

The fetcher scans Git checkouts for LFS pointers. If found, it runs `git lfs pull`
only in that repository, reports the paths, and refuses unresolved pointers.
Install Git LFS if that is required by a future acquisition. This acquisition
contained **zero LFS pointers**. All Hugging Face sources were public and ungated.
For a gated source, obtain access and set `HF_TOKEN` in the environment; fetching
stops if access is unavailable. Tokens are neither written to files nor committed.

The verifier is read-only and exits nonzero on mismatches or missing data. A
`PASS (observed)` line reports a requested measurement that has no specified
expected value; it does not assert that the data is free from quality issues.
The full 2026-11-21 report is in [DATA_VERIFICATION.txt](DATA_VERIFICATION.txt).

## A. Between Help and Harm

**What:** Mental-health crisis classifications and evaluations of chatbot
responses, including four human classification annotators and repeated LLM judges.

**Citation:** Arnaiz-Rodriguez et al. (2026). *Between Help and Harm: An Evaluation
Study of Mental Health Crisis Handling by Large Language Models*. JMIR Mental
Health, 13, e88435. [DOI: 10.2196/88435](https://doi.org/10.2196/88435).
[Official repository](https://github.com/ellisalicante/LLMs-Mental-Health-Crisis).

**Licence:** Not specified in the inspected source repository. Do not infer a
dataset licence from the journal article's licence. This remains an open issue.

**Layout and use:** A full clone is in
`data/external_repos/between_help_and_harm/`. Its complete tracked `data/` tree is
copied into `data/raw/between_help_and_harm/` without pruning or renaming. Use:

- `human_label/H{1,2,3}_labeled_n206_s42.json` and
  `human_label/H4_labelled_n206_s42.json`: four classification annotations.
- `human_label/H{1,2}_scores_n206_s42.json`: human response scores.
- `llm_label/*-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-*.json`:
  nine independent runs (three models × three runs), each with 206 records.
  The additional `*-merged_labels.json` is an aggregate, not an independent run;
  it is preserved and reported separately.
- `llm_evaluator/sampled/sampled_lowest_scores_n200__jury.json`: the 206 sampled
  replies and the three judges' evaluations.
- `llm_evaluator/protocol.csv`: seven category rows, including `no_crisis`, and
  five response-quality levels.

The four human files have identical input ordering. Crisis means any published
category other than `no_crisis`; their 0/1/2/3/4-human vote counts are
36/33/33/26/78. The verifier validates input alignment before counting votes.
Some score JSON uses non-standard `NaN` values; Python reads these without
rewriting them. No source content is repaired.

**Caveat:** Between Help and Harm: the 206 evaluated replies were sampled toward the LOWEST judge scores (not representative).

## B. CounselBench

**What:** Expert ratings of counseling responses, adversarial prompts, and human
annotations of responses to those prompts.

**Citation:** Li et al. (ICLR 2026). *CounselBench: A Large-Scale Expert Evaluation
and Adversarial Benchmark of Large Language Models in Mental-Health Counseling*.
[Official repository](https://github.com/llm-eval-mental-health/CounselBench).

**Licence:** CC BY-NC-ND 4.0, declared in each Hugging Face dataset card. No
redistribution of modified copies. The reference code repository has no explicit
licence file at the inspected revision; its data sources declare their own licence.

**Layout and use:** Original Hugging Face dataset snapshots, not auto-converted
Parquet exports, are in `data/raw/counselbench/`:

| Dataset | Published file used | Observed structure |
| --- | --- | --- |
| [CounselBench-Eval](https://huggingface.co/datasets/izi-ano/CounselBench-Eval) | `CounselBench-Eval/counselbench_eval.csv` | 2,000 × 20; 100 questions × 4 responders × 5 expert rows |
| [CounselBench-Adv](https://huggingface.co/datasets/izi-ano/CounselBench-Adv) | `CounselBench-Adv/counselbench_adv.csv` | 20 rows × 6 failure-mode columns |
| [Adv human annotations](https://huggingface.co/datasets/izi-ano/CounselBench-Adv-human-annotation) | `CounselBench-Adv-human-annotation/counselbench-adv-human-annotation.csv` | 1,080 rows × 10 columns |

Each snapshot also retains its `.gitattributes` and `README.md`.
`data/external_repos/counselbench_code/` is a reference-only clone; no code is run.
Adv's six columns are `apathetic`, `assumptions`, `symptoms`, `judgmental`,
`medication`, and `therapy`: 20 rows should not be mistaken for 20 total prompts.
The human-annotation columns are `mode`, `example_post`, `example_response`,
`comment_1`, `comment_2`, `new_post`, `model_name`, `new_response`, `question`, and
`human_answer`. No per-sentence annotator flags are assumed.

**Caveat:** CounselBench: no annotator IDs; per-rater rows are in long format; flagged sentences are free text ("*_copy" fields) and must later be matched to reply sentences.

The Eval `survey_id` field is preserved as published; it is not treated as a
confirmed cross-item annotator identity.

## C. MentalAlign-70k / MentalBench-100k

**What:** Human and LLM judgments of mental-health dialogue responses, plus the
published context/response benchmark archive.

**Citation:** Badawi, Rahimi, Laskar et al. (EACL 2026). *When Can We Trust LLMs in
Mental Health? Large-Scale Benchmarks for Reliable LLM Evaluation*.
[Official repository](https://github.com/abeerbadawi/MentalBench-Align).
The repository supplies the [2025 preprint citation, arXiv:2510.19032](https://arxiv.org/abs/2510.19032).

**Licence:** MIT, as supplied in the repository's `LICENSE`.

**Layout and use:** The source clone is
`data/external_repos/mentalbench_align/`. Raw copies preserve source paths under
`data/raw/mentalalign/`:

- `MentalAlign-70k/Human_Judge/Human_judge.xlsx`: 1,000 rows × 12 columns.
- `MentalAlign-70k/LLMs_as_a_judge/*.xlsx`: all four published judge workbooks.
- `Codes/Analysis/*_llm_judge_with_human.csv`: four files, each 1,000 × 22.
- `MentalBench-100k/MentalBench-100k.zip`: original archive retained.
- `MentalBench-100k_extracted/`: archive members, including
  `MentalBench-100k.csv` and the original `__MACOSX` metadata member.

Office lock files named `~$*` are excluded from the raw copy, as requested, but
remain in the full reference clone. Published column whitespace is preserved.
Score cells contain JSON objects; the verifier reads these in memory without
modifying cells. Across ten human-response columns, 9,941 numeric Safety scores
are available, 59 are missing or JSON null, and **34 are outside 1–5** (33 zeros
and one six). There are 8,303 scores equal to 5. No imputation is performed.

**Caveat:** MentalAlign: one human score per item; Safety scores are heavily skewed toward 5.

## D. MentalHealthBench

**What:** Synthetic conversation prefixes and expert-consensus rubrics for
evaluating the next assistant response.

**Citation:** Malik et al., OpenAI (September 2026). *MentalHealthBench: An
Expert-Informed Benchmark of AI Capabilities in Realistic Mental Health
Conversations*. [Official announcement](https://openai.com/index/introducing-mentalhealthbench/)
and [paper](https://cdn.openai.com/ctf-cdn/MentalHealthBench_A_Comprehensive_Benchmark_of_AI_Capabilities_in_Realistic_Mental_Health_Conversations.pdf).

**Licence:** MIT, confirmed by the archive's `LICENSE` (OpenAI, 2026).

**Official discovery:** The announcement links to the paper. Section 8, Data
Access (page 23), supplies the
[official CDN archive](https://cdn.openai.com/ctf-cdn/OAI_MentalHealthBench.zip).
No mirror or reconstructed version is used. No Git/HF revision is published:
the manifest records `revision: null`, the discovery chain, and the archive's
SHA-256 as its content pin. Failure of this URL stops this dataset; there is no
fallback to a third-party copy.

**Layout and use:** `data/raw/mentalhealthbench/OAI_MentalHealthBench.zip` is
retained, with its original `mentalhealthbench/` directory extracted next to it.
That directory contains `README.md`, `LICENSE`, and the file used for analysis,
`mentalhealthbench_eval.jsonl`.

Verification finds 1,215 conversations and 5,262 criteria. Acuity counts are
650 `non_acute`, 221 `high_acuity`, and 344 `emergent`. The release contains
conversation context and rubrics, but no generated target model responses,
grader outputs, or individual expert votes. Assistant turns in the context
are not target responses. The report enumerates exact nested field names and
language counts. Retain context/system messages during later evaluation and
exclude the canary marker from model/grader inputs. Do not publish conversation
examples; the release requests this to limit benchmark contamination.

**Caveat:** MentalHealthBench: rubric criteria were kept only when experts agreed, so expert disagreement is not recoverable.

## Not fetched

- **Verily mental-health crisis dataset (Nelson et al.):** available on request
  only. No request or download is made by these scripts.
- **SIM-VAIL (Weilnhammer et al.):** interactive explorer only; no confirmed
  download. No explorer scraping or reconstructed dataset is substituted.

## Verification and open issues

- Full verification: **192 PASS, 0 FAIL**. Observed data-quality caveats are not
  assertions that all values are valid.
- Repeat fetch: **0 download operations, 0 new raw files, 418 matching files**.
- Preservation/security regression tests: **12 passed**, using synthetic files.
- Between Help and Harm's dataset licence and the CounselBench reference code
  licence remain unspecified in their inspected repositories.
- MentalAlign's 34 out-of-range Safety scores and 59 missing/null scores remain
  unchanged. Later analysis must state its policy for those values explicitly.
- No requested dataset is missing. Verily and SIM-VAIL were intentionally not fetched.

## E. CAPphrase (WP1, data only)

**Citation:** Kucharski AJ (2026). *CAPphrase: Comparative and Absolute Probability
phrase dataset*. [DOI: 10.5281/zenodo.18750055](https://doi.org/10.5281/zenodo.18750055).
[Official repository](https://github.com/adamkucharski/CAPphrase).
**Licence:** CC-BY 4.0, as declared in the source README. Attribution is required.

Fetch with `.venv/bin/python -I scripts/fetch_data.py --only capphrase`. The full
reference clone lives in ignored `data/external_repos/capphrase/`. Only these
three published files are copied byte-for-byte into `data/raw/capphrase/`:

- `absolute_judgements.csv`: response_id, term, probability, order.
- `pairwise_comparisons.csv`: response_id, pair_id, term1, term2, selected.
- `respondent_metadata.csv`: response_id, timestamp, age_band,
  english_background, education_level, country_of_residence.

`data/MANIFEST.json` pins the Git commit and records each file's download time,
source URL, size, SHA-256, and licence. Other source data files, including
benchmark yardsticks, are outside this task. No downloaded R code is executed.
These are the publisher's preprocessed release files; this project applies no
additional outlier filtering, sample exclusions, or analysis in WP1.

`src/data/capphrase.py` exposes `load_judgements()`, `load_pairs()`, and
`load_respondents()`, with an optional root path for synthetic fixtures. Each
returns a DataFrame preserving published rows/order and original fields, plus
`english_group` and `age_group`. Groups are attached to judgements and pairs via
a validated many-to-one respondent join; duplicate metadata or missing coverage
raises rather than dropping rows. Demographic source strings, including literal
`NA`, stay intact. The mapping is also documented in the module docstring:

| Source value | Derived group |
|---|---|
| English is my first language | native |
| English is not my first language but I am fluent | non_native |
| English is not my first language and I am not fluent | non_native |
| 18-24, 25-34 | 18-34 |
| 35-44, 45-54 | 35-54 |
| 55-64, 65-74, 75+ | 55+ |
| Under 18 | under_18 |
| Literal NA, empty/whitespace-only, or null | missing |

Whitespace is stripped only when comparing group labels. Unknown nonmissing
categories raise; they are never silently assigned to `missing`. Under-18 and
missing-demographic respondents remain included.

`.venv/bin/python -I scripts/verify_data.py` checks the requested row counts,
respondent coverage of all 19 terms, probability bounds, presentation ordering,
and pair consistency. Its group counts count **respondents**, not repeated
judgement rows. The requested term medians are an unfiltered verification
summary printed to stdout; WP1 creates no experiment or analysis artifacts.
Synthetic tests cover mappings, preservation, join failures, numeric parsing,
and out-of-range verification. No real respondent rows are embedded in tests.
