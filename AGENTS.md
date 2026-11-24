# Repository conventions

- Raw data in `data/raw/` is read-only. Never edit, re-save or "fix" it.
- Derived data goes only to `data/processed/` (gitignored), produced by scripts.
- Never commit raw or derived data, or excerpts of it (licence: CounselBench CC BY-NC-ND; Between Help and Harm has no licence). Tests use small synthetic fixtures only.
- Every number in the paper must come from a script under `experiments/`, writing to `results/`.
- Fixed random seeds; no network calls in loaders or experiments, except a local model server on localhost, recorded in the output.
- One task = one branch = one PR; do not edit folders owned by another open PR.
- Run Python that reads downloaded data with `python -I`.
- Log each agent task in `docs/AI_USAGE.md`.
- Exclusions/cleaning happen only in loaders, are rule-based, and are logged with counts.
