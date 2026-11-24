# Build the seminar paper. Requires a TeX distribution with acmart (TeX Live 2023+), Tectonic, or Overleaf.
PY ?= ../.venv/bin/python

.PHONY: all pdf tectonic numbers check clean

all: numbers check pdf

numbers:
	$(PY) -I scripts/export_numbers.py
	$(PY) -I scripts/export_tables.py

check:
	$(PY) -I scripts/check_numbers.py

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# Alternative engine without a TeX installation (XeTeX; fetches packages on first use).
tectonic: numbers check
	tectonic -X compile --keep-logs main.tex

clean:
	latexmk -C main.tex
