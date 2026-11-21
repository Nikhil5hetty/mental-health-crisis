# Loader preflight blockers

Inspection on 2026-11-21, using `python -I scripts/inspect_loader_alignment.py`.
No dataset excerpts are included here. No raw files were modified.

The task explicitly requires stopping if required fields are absent or alignment
fails. Loader implementation and processed-data generation have therefore stopped.

- The sampled `sampled_lowest_scores_n200.json` has an `evals` list of three
  numeric scores. The sampled `__gpt-5-nano.json` and `__meta-llama.json` files
  instead expose `evaluation` and `source_file`, with no `evals` field. The
  requested three independent scores per judge cannot be read from those files
  using the specified schema. Following `source_file` into other published
  artifacts would require an explicitly agreed source-selection rule.
- All four human crisis-label files and nine independent LLM crisis-label runs
  have matching `inputs` lists in published order. Both human score files use
  strings instead. Exact cross-file equality fails on all 206 entries by type.
  Joining the H1 label input lists with empty strings, spaces, newlines, or
  double newlines matches only 79 of the 206 H1 score strings in each case.
  A type conversion alone therefore does not establish the requested alignment.
- Answer alignment, complete-score counts, and the remaining dataset loader
  schemas have not been validated. No claim is made that the loaders or build
  are complete, and no integration-test results are available.

Resolution needed: authorize investigation of an explicit text-alignment rule
and alternative published per-run source files, or revise the requested schema.
Do not silently normalize text, align by position alone, or replicate aggregate
scores as independent runs.

## Resolution

The user approved D11 and D12 in `docs/DECISIONS.md`. D11 permits positional
alignment only after content verification using the exact ` | ` input join,
comparison-only answer normalization/similarity, and the explicit artefact rule.
D12 resolves runs only in the corresponding judge's merged folder, with exactly
one content match and matching evaluations. No run padding is permitted.
Implementation will retain both audit logs under ignored `data/processed/` and
assert the approved reference counts. Any remaining discrepancy will be reported.

### Validation of D11/D12 and remaining clarifications

D12 resolves 190 items and marks 16 unscored for each judge. GPT-5-nano has one
resolved item with two runs; all other resolved judge/items have three. The
complete-case set has 190 items. D11 has zero exclusions using the approved
literal-plus regex. H1 inputs verify exactly for 206 items; H2 for 205, with
item 45 accepted as an artefact. H2 answer artefacts occur at 21, 22, and 42.
The sampled judges' item 22 answers are near-identical to H1. Their item 124
answers match after normalization, and therefore receive `normalised` status
under D11's ordered rules, rather than `near_identical`.

Remaining decisions requested from the user:

- There are 27 empty LLM crisis labels. Neither D11 nor D12 specifies how missing
  verdicts enter crisis counts. Proposed: preserve missing verdicts and report
  observed/missing counts; do not count blanks as crisis or non-crisis.
- Accepted human answer artefacts include non-string values. Parquet cannot store
  arbitrary numeric/string values in one string column. Proposed: preserve every
  original string, use null for non-string answer cells, and retain artefact
  type/value metadata separately.
- All four specified MentalAlign CSVs agree on 69 null Safety scores and 34
  out-of-range scores. The earlier 59-null reference came from Human_judge.xlsx.
  Proposed: assert 69 for the requested CSV-based table, document the discrepancy,
  and keep the workbook reference separately rather than changing source selection.
- Proposed: assert `near_identical` at item 22 and `normalised` at item 124,
  retaining the verbatim original D11 expectations above.

The full build remains blocked until these choices are resolved. Synthetic
loader tests pass; a successful integration/build result is not claimed.

### Final resolution (2026-11-21)

The user approved all four remaining choices listed above. They are implemented
and recorded under the approved loader clarifications in `DECISIONS.md`.
The full build now passes: nine Parquet tables and both BHH CSV audit logs are
written only under ignored `data/processed/`. No unresolved alignment exclusions
remain. The source discrepancies and missing scores remain documented findings,
not repaired data. Prior blocked-state notes above are retained as history.

Final validation: the full build passes all approved reference assertions, the
project test suite passes 38 tests (including integration and preservation tests),
and the raw-data verifier reports 192 PASS / 0 FAIL with all 418 checksums intact.
