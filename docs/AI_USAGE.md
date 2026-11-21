# AI usage

| Date | Tool | Task | Scope and validation |
| --- | --- | --- | --- |
| 2026-11-21 | Codex | data fetch | Created the fetcher, read-only verifier, provenance manifest, documentation, and synthetic preservation tests on `feature/data`. Retrieved four published dataset families without modifying raw bytes; verified expected counts, recorded source quality/licence issues, and confirmed a repeat fetch downloads nothing. No downloaded repository code was executed. |
| 2026-11-21 | Codex | Repo conventions and loader preflight | Created `feature/loaders` from updated main and completed Part 0. Read-only structural inspection found missing per-run fields in sampled GPT-5-nano/Meta-Llama files and input alignment differences between BHH labels and human score files. Stopped before loader implementation as explicitly required; raw data unchanged. |
| 2026-11-21 | Codex | Read-only loaders, D11/D12 continuation | Added the approved decisions without removing prior blockers, implemented offline loaders and synthetic tests, and verified D11/D12 against published sources. The user approved literal-plus artefact matching. Further clarification requested for empty crisis labels, mixed-type answer serialization, CSV/workbook null-count differences, and normalized versus near-identical status. Raw files unchanged; full build pending. |
| 2026-11-21 | Codex | Complete read-only loaders | Applied user-approved missing-label counts, non-string artefact metadata, 69-null CSV Safety reference, and normalized item-124 assertion. Built nine Parquet tables and both audit logs offline; raw data unchanged. Full validation and source discrepancies are included in the PR. |

Validation note: the first unrestricted pytest invocation attempted collection
of `test_answer.py` inside a downloaded reference clone and failed while importing
its unavailable `openai` dependency. Added `pytest.ini` to restrict discovery to
trusted `tests/` and `scripts/`, excluding `data/`. No source dependency was
installed to satisfy that test. Raw checksums remained unchanged.
