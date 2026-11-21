# Build the seminar paper. Requires a TeX distribution with acmart (TeX Live 2023+), or use Overleaf.
PY ?= ../.venv/bin/python

.PHONY: all pdf numbers check clean

all: numbers check pdf

numbers:
	$(PY) -I scripts/export_numbers.py
	$(PY) -I scripts/export_tables.py

check:
	$(PY) -I scripts/check_numbers.py

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C main.tex
