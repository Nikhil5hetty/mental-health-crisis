# Seminar paper (ACM acmart): Same word, different meaning

Every number in the text is a macro generated from `results/`; nothing is typed by hand.

```bash
make            # export numbers and tables, run the number check, build main.pdf (latexmk, pdfLaTeX)
make tectonic   # the same with Tectonic (XeTeX), if no TeX distribution is installed
make numbers    # python -I scripts/export_numbers.py && python -I scripts/export_tables.py
make check      # python -I scripts/check_numbers.py (fails on hand-typed numbers)
```

- `main.tex`, `sections/*.tex`: hand-written sources. `refs.bib`: verified references (see header comments and `% VERIFY`).
- `generated/numbers.tex`: one macro per number, from `results/words/` (main study) and the earlier study's
  `results/{rq1,rq2,counselbench,demo}/` (motivation). `generated/numbers_index.csv` maps each macro to its source.
- `generated/tables/*.tex`: tables written from the CSVs; hypothesis verdicts are copied from `results/words/*/SUMMARY.md`.
- `figures/`: copied from `results/words/*/figures/` and the app screenshots in `docs/img/`.
- `scripts/number_allowlist.txt`: the only literal numbers allowed in the text, each with its source.

Overleaf: upload this folder (including `generated/` and `figures/`) and compile `main.tex` with pdfLaTeX.
