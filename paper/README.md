# Seminar paper (ACM acmart)

Every number in the text is a macro generated from `results/`; nothing is typed by hand.

```bash
make            # export numbers and tables, run the number check, build main.pdf (latexmk)
make numbers    # python -I scripts/export_numbers.py && python -I scripts/export_tables.py
make check      # python -I scripts/check_numbers.py (fails on hand-typed numbers)
```

- `main.tex`, `sections/*.tex`: hand-written sources. `refs.bib`: verified references (see header comments).
- `generated/numbers.tex`: one macro per number; `generated/numbers_index.csv` maps each macro to its source row.
- `generated/tables/*.tex`: tables written from the CSVs; hypothesis verdicts are copied from `results/*/SUMMARY.md`.
- `figures/`: copied from `results/*/figures/`.
- `scripts/number_allowlist.txt`: the only literal numbers allowed in the text, each with its source.

Overleaf: upload this folder (including `generated/` and `figures/`) and compile `main.tex` with pdfLaTeX.
