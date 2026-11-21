#!/usr/bin/env python3
"""Fail if the paper's hand-written .tex sources contain a number that is not a macro.

Scanned: paper/main.tex and paper/sections/*.tex (generated files are not scanned).
Removed before scanning: comments; arguments of citation/reference/label/file/url commands;
optional arguments of figure/table/graphics commands; tikzpicture environments (diagram coordinates).

A standalone number (digits, optionally with a decimal part) is allowed only if it is
- a year (1900-2099) in running text,
- listed in paper/scripts/number_allowlist.txt (CWW parameters from docs/METHOD.md, plan constants,
  service phone numbers), or
- a single digit inside math mode (indices, formula constants such as 1/2).
Digits that are part of identifiers (S2, RQ1, H2a, A3, IT2, D11, M3b) are not standalone numbers.

Run from anywhere:  python -I paper/scripts/check_numbers.py   (exit code 1 on violations)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
ALLOWLIST = PAPER / 'scripts/number_allowlist.txt'

STRIP_ARG_CMDS = r'(?:cite[a-z]*|Cite[a-z]*|ref|cref|Cref|autoref|eqref|label|input|include|includegraphics|url|href|' \
                 r'bibliography|bibliographystyle|documentclass|usepackage|citestyle|setcopyright|acmConference|' \
                 r'begin|end|hspace|vspace|setlength|addtolength|resizebox|fontsize|todo|email|orcid)'
# Standalone number: not part of an identifier (S2, IT2), a hyphenated term (type-1, 2-tuple) or a macro
# parameter (#1).
NUMBER = re.compile(r'(?<![A-Za-z0-9_.\\#])(?<![A-Za-z]-)\d+(?:\.\d+)?(?![A-Za-z0-9_])(?!-[A-Za-z])')
MATH = re.compile(r'\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|'
                  r'\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}', re.S)


def load_allowlist() -> set[str]:
    allowed = set()
    for line in ALLOWLIST.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            allowed.add(line.split()[0])
    return allowed


def blank(m: re.Match) -> str:
    """Replace a match by spaces/newlines of the same shape, so line numbers stay correct."""
    return re.sub(r'[^\n]', ' ', m.group(0))


def strip_comments(text: str) -> str:
    return re.sub(r'(?<!\\)%.*', blank, text)


def strip_commands(text: str) -> str:
    text = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', blank, text, flags=re.S)
    # \cmd[opt]{arg}{arg2} -> blanked (repeat for nested/multiple arguments).
    pattern = re.compile(r'\\' + STRIP_ARG_CMDS + r'\*?(\[[^\]]*\])*(\{[^{}]*\})+')
    prev = None
    while prev != text:
        prev, text = text, pattern.sub(blank, text)
    text = re.sub(r'\\\\\[[^\]]*\]', blank, text)  # line-break spacing \\[2pt]
    return text


def violations(text: str, allowed: set[str]) -> list[tuple[int, str, str]]:
    found = []
    if '\\begin{document}' in text:  # preamble holds definitions, not running text
        head, body = text.split('\\begin{document}', 1)
        text = re.sub(r'[^\n]', ' ', head) + ' ' * len('\\begin{document}') + body
    no_comments = strip_comments(text)
    # Math spans are found before commands are blanked (blanking keeps every position unchanged).
    math_spans = [(m.start(), m.end()) for m in MATH.finditer(no_comments)]
    clean = strip_commands(no_comments)

    def line_of(pos: int, src: str) -> int:
        return src.count('\n', 0, pos) + 1

    for m in NUMBER.finditer(clean):
        tok = m.group(0)
        in_math = any(a <= m.start() < b for a, b in math_spans)
        if tok in allowed:
            continue
        if not in_math and re.fullmatch(r'(19|20)\d\d', tok):
            continue
        if in_math and re.fullmatch(r'\d', tok):
            continue
        found.append((line_of(m.start(), clean), tok, 'math' if in_math else 'text'))
    return found


def main() -> int:
    allowed = load_allowlist()
    files = [PAPER / 'main.tex', *sorted((PAPER / 'sections').glob('*.tex'))]
    bad = 0
    for f in files:
        for line, tok, where in violations(f.read_text(encoding='utf-8'), allowed):
            print(f'{f.relative_to(PAPER.parent)}:{line}: hand-typed number {tok!r} in {where}; use a macro from '
                  f'paper/generated/numbers.tex or justify it in number_allowlist.txt')
            bad += 1
    print('check_numbers: FAIL' if bad else f'check_numbers: OK ({len(files)} files)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
