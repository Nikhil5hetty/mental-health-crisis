| ID | Decision |
|---|---|
| D2 | Datasets used exactly as published; derived data only via scripts in data/processed/. |
| D3 | Core study is zero-API (reuse stored judge outputs); optional small prompt test later. |
| D4 | RQ1: Between Help and Harm crisis labels; fuzzy degree k/4 humans vs k/9 LLM runs vs majority. |
| D5 | RQ2: Between Help and Harm appropriateness, 2 humans vs 3 judges × 3 runs; MentalAlign replication. |
| D6 | CounselBench: descriptive expert-uncertainty analysis only. |
| D7 | Licences: no redistribution of raw/derived data; BHH research use with citation. |
| D8 | MentalAlign null/out-of-range human scores excluded at load time, counts reported. |
| D9 | Ignore __MACOSX/ archive members. |
| D10 | MentalHealthBench optional. |

## D11 – BHH alignment rule
item_id = published position (0..205). Positional alignment is ACCEPTED ONLY when content-verified, per item and per file:
1. Inputs: score files store inputs as one string = label `inputs` list joined with " | ".
   Compare `" | ".join(label_inputs)` with the score-file string (exact).
2. Answers: compare to the H1 score-file answer after normalisation only for comparison
   (NFKC, remove "\r", collapse whitespace, strip). Accept if exact after normalisation OR
   difflib ratio >= 0.99.
3. A text cell is an "artefact" if it is not a string (NaN, int, float) or matches
   r"^[A-Z]\\+[A-Z]+\d+\\+[A-Z]+\d+$". An artefact cell is "unverifiable", not "mismatched".
   Accept the item for that file only if its OTHER text field (inputs or answer) verifies under rules 1–2.
4. Anything else that fails → exclude the item for that file and log it.
- Never modify raw text. Normalised text is used only for the comparison; loaders output the original strings.
- Write `data/processed/bhh_alignment_log.csv`: item_id, file, field, status
  (exact | normalised | near_identical | artefact_accepted | excluded), similarity.
- Expected (assert in build_processed.py): inputs exact for H1 206/206 and H2 205/206 (item 45 artefact);
  H2 answer artefacts at items 21, 22, 42; near-identical answers at 22 and 124; 0 exclusions.

## D12 – BHH per-run judge scores
For each judge in {gpt-4o-mini, gpt-5-nano, meta-llama}:
- Take the source file name from `sampled_from` (gpt-4o-mini file) or `source_file` (other two).
- Open ONLY `llm_evaluator/<judge>/n_200/merged/<that file>`. Never another judge's folder
  or the top-level `llm_evaluator/n_200/`.
- Find the record with exact equality on BOTH `answer` and `inputs`. Require exactly 1 hit.
- Require merged `evaluation` == sampled `evaluation` (abs diff < 1e-9). Then read `evals`
  (string or list) as the per-run scores; run index = list position (1-based).
- Items whose sampled `evaluation` is missing/unparseable are "unscored", so exclude them for that judge.
- Never impute, duplicate or pad runs. If `evals` has fewer than 3 entries, keep what is there.
- Write `data/processed/bhh_run_resolution_log.csv`: item_id, judge, source_file, hits, status, n_runs.
- Expected (assert): per judge 190 resolved + 16 unscored; gpt-4o-mini and meta-llama 3 runs on
  every resolved item; gpt-5-nano exactly 1 item with 2 runs, the rest 3.
- Complete-case set for RQ2 = items with H1, H2 and all three judges scored (expected 190).

D11 clarification (user approved): match literal plus signs using
`r"^[A-Z]\+[A-Z]+\d+\+[A-Z]+\d+$"`. The original decision text above is retained.

### Approved loader clarifications (2026-11-21)

- Preserve the 27 empty LLM crisis verdicts as missing labels. Exclude missing
  verdicts from crisis/category counts and report `n_llm_observed` and
  `n_llm_missing` per item. Never silently treat missing verdicts as non-crisis.
- Preserve every original answer string. For non-string artefacts, output null
  in `answer_text`, retaining the original Python type and `repr` in
  `answer_artefact_type` and `answer_artefact_value`. String artefacts stay intact.
- The specified MentalAlign CSVs have 69 null Safety scores, versus 59 in the
  separately published workbook. Assert 69 for the CSV-derived table and retain
  the workbook finding as a documented source difference. Both have 34
  out-of-range scores.
- Under D11's ordered comparisons, assert sampled answer item 22 as
  `near_identical` and item 124 as `normalised`. The original D11 wording above
  remains verbatim for audit history.

### Numbered rulings (synced 2026-11-21, after the RQ1/demo plan commit)

| ID | Decision |
|---|---|
| D13 | Missing LLM crisis labels (27 empty labels across 21 items in the 9 validation runs) stay missing and are never read as no_crisis. Crisis degree = k_crisis / n_observed; n_observed is stored per item (minimum 6 of 9). |
| D14 | Non-string text artefacts: answer_text is null, with artefact_type and artefact_value columns. Original strings are kept unchanged. |
| D15 | RQ2 rulings: verdicts use paired-difference CIs vs S2; identical systems stay "inconclusive"; the added M3b/M5 paired differences are accepted as deviations; the 10 hypothesis comparisons are reported as exploratory (no multiplicity correction). |
| D16 | PR #22 deviations accepted; the planned post-hoc demo sensitivity is superseded by the word-meaning pivot (D17) and not implemented. |
| D17 | Pivot (2026-11-24): main study = citizen vs LLM word models (docs/ANALYSIS_PLAN_WORDS.md); earlier RQ1/RQ2/CounselBench results become the motivation section; PR #23 kept as the paper template. |
| D18 | PR #26 rulings on the LLM elicitation harness: (1) parsing: the first number must itself lie in [0, 100], otherwise re-ask, accepted as a stricter reading of the plan, with the re-ask rate reported (item 1 superseded by D19); (2) a re-ask happens in the same conversation with the same seed, so the changed context gives a new draw without new wording; (3) `num_predict = 64` is accepted as a technical cap, and replies that hit it are reported from the `done_reason` column (`length`, final attempt), added before the Qwen, Gemma and Mistral runs; the original Llama rows predate that column and are covered by the D19 superseded counts, while the requeued Llama rows carry it; (4) the local Ollama server on localhost is the only network endpoint, recorded in the output, and AGENTS.md is clarified accordingly. |
| D19 | Strict bare-number parsing adopted after inspecting the first real model's raw replies and before any analysis; verbose replies such as 'Step 1 ...' had been parsed as 1. Superseded rows kept in superseded.csv. |
| D20 | Word-study analysis rulings (PR #30, PR #31; recorded in WP7, to be confirmed by the PM): (1) RQ2: a model x prompt x term with fewer than 30 valid answers is excluded (WP4 brief), and the width ratio is undefined (NaN, no CI) when the citizen IQR is 0 (About Even); H2b compares IQRs directly, so such a term is never counted as narrower. (2) RQ3: when all four level-word memberships are 0, the nearest-centre fallback stays the primary rule; the fallbacks at m = 0 and m = 100 are harmless (all codebooks agree) and are described; the LLM gap at about m = 33 is reported as a finding; two post-hoc sensitivities, labelled as such, are added: (a) flips counted only on items decided without a fallback, (b) the more severe of the two neighbouring words as fallback. |
