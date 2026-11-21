# Derived data dictionary

All loaders read published files under `data/raw/` without network access or writes.
Tables are built only by `python -I scripts/build_processed.py`, under ignored
`data/processed/`. No raw or derived rows belong in Git. Loader functions accept
an optional dataset-root `Path` for synthetic fixtures. Audit counts are returned
in `DataFrame.attrs`; the build prints counts and writes the two BHH audit logs.
Null is an absent observation, not a negative verdict. No random sampling occurs.

## BHH crisis labels (`bhh_crisis_labels.parquet`)

Sources: `between_help_and_harm/human_label/H{1,2,3}_labeled_n206_s42.json`,
`H4_labelled_n206_s42.json`, and the nine independent validation files under
`llm_label/`. The `*-merged_labels.json` aggregates are excluded.

| Column | Type | Meaning / allowed values |
|---|---|---|
| item_id | int64 | Published position, 0–205, content-verified across label files |
| annotator | string | H1–H4 or `<published-model-name>_run<k>` |
| annotator_type | string | human or llm |
| model | nullable string | Published model name; null for humans |
| run | nullable Int64 | 1–3, sorted filename order within model; null for humans |
| label | string | Published crisis category or null for an empty LLM verdict |

Seven categories: `suicidal_ideation`, `self-harm`, `anxiety_crisis`,
`violent_thoughts`, `substance_abuse_or_withdrawal`, `risk_taking_behaviours`,
`no_crisis`. Run numbers identify list/file positions, not shared stochastic seeds.

## BHH crisis degrees (`bhh_crisis_degree.parquet`)

Source: crisis-label files above, one row per published item.

| Column | Type | Meaning / allowed values |
|---|---|---|
| item_id | int64 | 0–205 |
| item_text | string | Published input segments joined by the approved ` | ` separator |
| k_humans_crisis | int64 | Number of crisis votes, 0–4 |
| k_llm_crisis | int64 | Number of observed crisis votes, 0–9; missing excluded |
| n_llm_observed | int64 | Nonmissing LLM verdicts, 0–9 |
| n_llm_missing | int64 | Missing LLM verdicts, 0–9; observed + missing = 9 |
| human_count_`category` | int64 | Seven columns, one for each category listed above; 0–4 |
| llm_count_`category` | int64 | Seven corresponding columns; 0–9 |
| human_majority | string | Strict binary majority: crisis for k>2, no_crisis for k<2, tie for k=2 |

`human_majority` is a binary crisis decision, not an invented tie-break among the
seven categories. There are 27 missing LLM verdicts. Use the reported observed denominator
when estimating an observed-vote proportion; a fixed /9 value would conflate
missing and non-crisis verdicts. This loader returns counts, not an imputed degree.

## BHH appropriateness (`bhh_appropriateness.parquet`)

Sources: `human_label/H{1,2}_scores_n206_s42.json`, sampled judge files under
`llm_evaluator/sampled/`, and only their D12-resolved files under
`llm_evaluator/<judge>/n_200/merged/`.

| Column | Type | Meaning / allowed values |
|---|---|---|
| item_id | int64 | Published position, content-verified under D11 |
| rater | string | H1, H2, or `<judge>_run<k>` |
| rater_type | string | human or llm |
| model | nullable string | gpt-4o-mini, gpt-5-nano, meta-llama; null for humans |
| run | nullable Int64 | 1-based position in the stored evals list; null for humans |
| score | float64 | Published numeric score, including fractional human scores; NaN if missing |
| crisis_label | nullable string | Published score-file label, not an inferred crisis verdict |
| answer_text | nullable string | Original source string; null for non-string artefacts |
| answer_artefact_type | nullable string | Python type name for artefacts; null otherwise |
| answer_artefact_value | nullable string | Python repr of the original artefact; null otherwise |

Unscored judge items contribute zero rows, not padded runs. An item belongs to the
RQ2 complete-case set when H1, H2, and each of the three judges have scored it;
this does not imply nine available LLM runs. The expected 190-item set contains
one GPT-5-nano item with only two stored runs. No replication fills that gap.

`bhh_alignment_log.csv` columns: `item_id` (int64), `file` (dataset-relative
string), `field` (`inputs`/`answer`), `status` (`exact`, `normalised`,
`near_identical`, `artefact_accepted`, `excluded`), `similarity` (float64 0–1,
NaN for unverifiable artefacts). Normalization is comparison-only. Artefacts
require the other field to verify. Excluded items are omitted for that file.

`bhh_run_resolution_log.csv` columns: `item_id` (int64), `judge` (one of the
three names), `source_file` (published filename), `hits` (int64; one for resolved,
zero when lookup is skipped), `status` (`resolved`, `unscored`, `excluded`),
`n_runs` (int64; actual list length, zero for skipped items). Ambiguous matches or
evaluation mismatches raise before scores are returned.

## BHH protocol (`bhh_protocol.parquet`)

Source: `between_help_and_harm/llm_evaluator/protocol.csv`, unpivoted into 35 rows.
`category` is one of the seven names; `level` is int64 1–5; `level_name` is the
published heading without its numeric prefix; `description` is the unchanged
published string. Level names are Harmful / Unacceptable, Inappropriate,
Partially Appropriate, Mostly Appropriate, Fully Appropriate.

## CounselBench expert ratings (`counselbench_eval.parquet`)

Source: `counselbench/CounselBench-Eval/counselbench_eval.csv`.

| Column | Type | Meaning / allowed values |
|---|---|---|
| reply_id | string | `questionID|responder` |
| rater_pos | int64 | 1–5 by published row order within reply; NOT a stable rater identity |
| responder | string | gpt4, llama3, gemini, human |
| topic | string | Published topic |
| overall, empathy, specificity, toxicity | int64 | Four scores, each 1–5 |
| medical_advice | string | yes, no, not_sure |
| factual | string | 1, 2, 3, 4, not_sure (strings keep the mixed scale explicit) |
| overall_reason, medical_reason, factual_reason, toxicity_reason | string | Original strings, including blanks and literal placeholders |
| medical_copy, factual_copy, toxicity_copy | string | Original flagged-sentence text; no sentence matching |
| medical_copy_clean, factual_copy_clean, toxicity_copy_clean | nullable string | Placeholder-only replacement; other strings unchanged |

The exact case-insensitive placeholder set, after stripping for comparison, is
`na`, `n/a`, `n/.a`, `nil`, `none`, `nothing wrong with it.`. The original string
columns are never changed. Empty source fields remain empty strings; neither
empty nor whitespace-only clean fields count as copied sentences.

## CounselBench reply degrees (`counselbench_reply_degree.parquet`)

One row per reply, derived from the expert table. `reply_id`, `responder`, and
`topic` are strings as above. All remaining columns are int64 counts 0–5:

- `overall_count_1` through `_5`, `empathy_count_1` through `_5`,
  `specificity_count_1` through `_5`, `toxicity_count_1` through `_5`.
- `medical_advice_count_yes`, `_no`, `_not_sure`.
- `factual_count_1` through `_4`, and `factual_count_not_sure`.
- `k_medical_yes`, `k_medical_not_sure`, `k_factual_not_sure`: explicit aliases
  for those distribution counts.
- `k_medical_copied`, `k_factual_copied`, `k_toxicity_copied`: nonblank,
  non-placeholder clean copy fields. These count raters, not sentences.

## MentalAlign ratings (`mentalalign_ratings.parquet`)

Sources: the four `mentalalign/Codes/Analysis/*_llm_judge_with_human.csv` files.
Columns are stripped in memory only. `__MACOSX` paths are ignored. There is no
ConvID column in these CSVs: `conv_id` is the 1-based published row position;
exact context/response ordering and parsed human scores must agree across all
four files. This identifier is local to this release.

| Column | Type | Meaning / allowed values |
|---|---|---|
| conv_id | int64 | 1-based published row position |
| response_model | string | original, claude, ds_llama, ds_qwen, gemini, gpt4o, gpt4omini, llama_3, qwen_2, qwen_3 |
| judge | string | claude, gpt-4o, gemini, o4-mini |
| attribute | string | Guidance, Informativeness, Relevance, Safety, Empathy, Helpfulness, Understanding |
| human_score | float64 | 1–5 or NaN after D8 exclusions |
| judge_score | float64 | Stored numeric judge score; NaN if missing/unparseable |
| human_score_flag | string | ok, null, out_of_range, unparseable |
| same_provider | bool | Provider equality, defined below |

Provider mapping: claude→Anthropic; gpt-4o/o4-mini/gpt4o/gpt4omini→OpenAI;
gemini→Google; ds_llama/ds_qwen→DeepSeek; llama_3→Meta;
qwen_2/qwen_3→Alibaba; original→human. DeepSeek models are attributed to the
response provider, not their base-model family. Original human responses never
match an LLM provider. JSON is parsed first; fallback recognizes explicitly named
attribute/numeric-or-null pairs using regex, never `eval`. Human exclusion counts
are reported per unique conversation/response/attribute, not quadrupled for the
four judges. Human scores are compared across files before invalid values are masked.

## Optional MentalHealthBench tables

Source: `mentalhealthbench/mentalhealthbench/mentalhealthbench_eval.jsonl`.
The optional tables are skipped if the dataset directory is absent. If present
but incomplete/malformed, loading fails rather than silently skipping it.

`mhb_conversations.parquet`: `id` (string), `acuity` (string: non_acute,
high_acuity, emergent), `language` (published language-code string),
`user_profile` (adult, teen, clinician, caregiver), `has_prior_context` (bool),
`n_messages` (int64; number of stored context turns).

`mhb_rubric.parquet`: `id` (string), `criterion_idx` (int64, zero-based within
conversation), `behavior_axis` (published string), `criterion_text` (unchanged
string), `points` (published signed integer). No model answers, grader output,
or expert votes are generated or inferred.

The approved CSV-based Safety reference is 34 out-of-range and 69 null scores.
The earlier workbook reference of 59 null scores is a separate source observation;
the loader does not substitute workbook values into CSV records.

## Running the build and tests

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -I scripts/build_processed.py
.venv/bin/python -I -m pytest
```

`pytest.ini` confines discovery to trusted project tests and scripts. The one
integration test skips automatically when `data/raw/` is absent. It reads the
local published inputs and checks the build; all unit fixtures are synthetic.
The build validates every reference before writing Parquet outputs. Audit CSVs
are written before assertions so failed alignment remains inspectable locally.
