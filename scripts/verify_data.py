#!/usr/bin/env python3
"""Read-only checks of published datasets; FAIL never triggers a data repair.

PASS (observed) denotes a requested measurement with no expected value supplied.
All other PASS/FAIL lines compare actual values to explicit expectations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

if __name__ == "__main__" and not sys.flags.isolated:
    os.execv(sys.executable, [sys.executable, "-I", str(Path(__file__).resolve()), *sys.argv[1:]])

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, label, actual, expected):
        ok = actual == expected
        self.passed += int(ok)
        self.failed += int(not ok)
        print(f'{"PASS" if ok else "FAIL"} {label}: actual={actual!r}; expected={expected!r}')
        return ok

    def observe(self, label, actual):
        self.passed += 1
        print(f"PASS (observed) {label}: actual={actual!r}; no fixed expectation")

    def run(self, label, function, *args):
        try:
            return function(*args)
        except Exception as exc:
            # A missing or malformed source must not suppress the remaining report.
            self.check(label, f"{type(exc).__name__}: {exc}", "readable published data")
            return None


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_manifest(r, root):
    manifest = read_json(root / "data/MANIFEST.json")
    paths = [entry["path"] for entry in manifest["files"]]
    r.check("manifest paths unique", len(paths), len(set(paths)))
    recorded = set(paths)
    actual = {p.relative_to(root).as_posix() for p in (root / "data/raw").rglob("*") if p.is_file()}
    r.check("raw files missing from manifest", sorted(actual - recorded), [])
    r.check("manifest files missing on disk", sorted(recorded - actual), [])
    for dataset in ("between_help_and_harm", "counselbench", "mentalalign", "mentalhealthbench", "capphrase"):
        entries = [e for e in manifest["files"] if e["dataset"] == dataset]
        r.check(f"{dataset} fetch status", manifest["datasets"].get(dataset, {}).get("status"), "fetched")
        r.check(f"{dataset} manifest has files", bool(entries), True)
        failures, pointers = [], []
        for entry in entries:
            path = root / entry["path"]
            try:
                if not path.resolve().is_relative_to((root / "data/raw").resolve()) or path.is_symlink():
                    raise ValueError("unsafe path")
                if path.stat().st_size != entry["size"] or hash_file(path) != entry["sha256"]:
                    failures.append(entry["path"])
                with path.open("rb") as stream:
                    if stream.read(64).startswith(b"version https://git-lfs"):
                        pointers.append(entry["path"])
            except (OSError, ValueError):
                failures.append(entry["path"])
        r.check(f"{dataset} size/SHA-256 mismatches ({len(entries)} files)", failures, [])
        r.check(f"{dataset} unresolved LFS pointers", pointers, [])
        r.observe(f"{dataset} published bytes (archives and extracted members both counted)", sum(e["size"] for e in entries))


def verify_counselbench(r, raw):
    def inspect(name, filename, evaluation=False):
        df = pd.read_csv(raw / "counselbench" / name / filename)
        r.observe(f"{name} columns", df.columns.tolist())
        if not evaluation:
            r.observe(f"{name} rows", len(df))
            return
        r.check(f"{name} shape", df.shape, (2000, 20))
        r.check(f"{name} unique questionID", df.questionID.nunique(), 100)
        r.check(f"{name} responders", sorted(df.responder.dropna().unique().tolist()), ["gemini", "gpt4", "human", "llama3"])
        r.check(f"{name} medical_advice_score counts", df.medical_advice_score.value_counts(dropna=False).to_dict(),
                {"No": 1729, "Yes": 220, "I am not sure": 51})
        r.check(f"{name} factual_consistency_score 'I am not sure'", int(df.factual_consistency_score.eq("I am not sure").sum()), 70)
        counts = df.groupby(["questionID", "responder"], dropna=False).size()
        r.check(f"{name} question/responder groups", len(counts), 400)
        r.check(f"{name} rows per group distribution", dict(Counter(counts.tolist())), {5: 400})
        print("Rows per (questionID, responder); each cell must be 5 (one row per expert):")
        for question, group in counts.groupby(level=0, sort=True):
            values = {responder: int(count) for (_, responder), count in group.items()}
            r.check(f"  {question}", values, {"gemini": 5, "gpt4": 5, "human": 5, "llama3": 5})
    for name, filename, evaluation in (
        ("CounselBench-Eval", "counselbench_eval.csv", True),
        ("CounselBench-Adv", "counselbench_adv.csv", False),
        ("CounselBench-Adv-human-annotation", "counselbench-adv-human-annotation.csv", False),
    ):
        r.run(name, inspect, name, filename, evaluation)


def verify_between_help_and_harm(r, raw):
    base = raw / "between_help_and_harm"
    humans = []
    def records(relative):
        data = read_json(base / relative)
        if not isinstance(data, list):
            raise ValueError(f"Expected record list: {relative}")
        r.check(f"{relative} records", len(data), 206)
        return data
    for name in [f"H{i}_labeled_n206_s42.json" for i in (1, 2, 3)] + ["H4_labelled_n206_s42.json"]:
        humans.append(r.run(name, records, "human_label/" + name))
    for i in (1, 2):
        r.run(f"H{i} scores", records, f"human_label/H{i}_scores_n206_s42.json")

    candidates = sorted((base / "llm_label").glob("*-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-*.json"))
    # *_merged_labels.json is an aggregate, not an independent model run.
    aggregates = [p for p in candidates if p.name.endswith("-merged_labels.json")]
    r.observe("validation aggregate label files (not independent runs)", [p.name for p in aggregates])
    labels = [p for p in candidates if p not in aggregates]
    r.check("validation-set LLM label files", len(labels), 9)
    models = Counter()
    for path in labels:
        models[path.name.split("-labeled-")[0]] += 1
        r.run(path.name, records, path.relative_to(base))
    r.check("validation LLM files per model", dict(models), {
        "gpt-4o-mini": 3, "gpt-5-nano": 3, "meta-llama-Llama-4-Scout-17B-16E-Instruct": 3})

    def jury_check():
        jury = records("llm_evaluator/sampled/sampled_lowest_scores_n200__jury.json")
        expected = {"gpt-4o-mini", "gpt-5-nano", "meta-llama"}
        bad = [i for i, row in enumerate(jury) if set(row["llm_evaluations"]) != expected]
        r.observe("jury llm_evaluations keys", sorted({k for row in jury for k in row["llm_evaluations"]}))
        r.check("jury records with unexpected evaluator keys", bad, [])
    r.run("sampled jury", jury_check)

    def protocol_check():
        protocol = pd.read_csv(base / "llm_evaluator/protocol.csv")
        r.observe("protocol crisis categories (rows)", protocol["label"].tolist())
        r.observe("protocol level columns", protocol.columns[1:].tolist())
        r.check("protocol level count", len(protocol.columns) - 1, 5)
        return set(protocol["label"])
    categories = r.run("protocol", protocol_check)

    def votes_check():
        if any(h is None for h in humans):
            raise ValueError("One or more human label files missing/unreadable")
        # There is no item ID in these files: require identical input order before voting.
        aligned = all([row["inputs"] for row in h] == [row["inputs"] for row in humans[0]] for h in humans[1:])
        if not r.check("human input sequences aligned", aligned, True):
            return
        observed = {row["label"] for h in humans for row in h}
        if categories is None or not r.check("unknown human labels", sorted(observed - categories), []):
            return
        votes = Counter(sum(row["label"] != "no_crisis" for row in rows) for rows in zip(*humans))
        r.check("messages labelled crisis by 0/1/2/3/4 humans", {n: votes[n] for n in range(5)},
                {0: 36, 1: 33, 2: 33, 3: 26, 4: 78})
    r.run("human crisis vote histogram", votes_check)


def safety_counts(df):
    columns = [c for c in df.columns if c.strip().startswith("human_response_")]
    if not columns:
        raise ValueError("No human score columns")
    values, unreadable, missing = [], [], []
    for col in columns:
        for i, value in enumerate(df[col]):
            try:
                if pd.isna(value):
                    missing.append((i + 2, col))
                    continue
                # Cells contain JSON score objects; never use eval or rewrite cells.
                score = json.loads(value)["Safety"]
                if score is None:
                    missing.append((i + 2, col))
                    continue
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError("non-numeric Safety")
                values.append(score)
            except (TypeError, ValueError, KeyError):
                unreadable.append((i + 2, col))
    return {"score_columns": columns, "parsed": len(values), "unreadable_cells": unreadable,
            "missing_cells": missing,
            "outside_1_to_5": sum(not 1 <= v <= 5 for v in values),
            "distribution": dict(sorted(Counter(values).items()))}


def verify_mentalalign(r, raw):
    base = raw / "mentalalign"
    csvs = sorted((base / "Codes/Analysis").glob("*_llm_judge_with_human.csv"))
    r.check("MentalAlign analysis CSV files", len(csvs), 4)
    def inspect(path):
        df = pd.read_csv(path)
        r.check(path.name + " shape", df.shape, (1000, 22))
    for path in csvs:
        r.run(path.name, inspect, path)
    def workbook_check():
        df = pd.read_excel(base / "MentalAlign-70k/Human_Judge/Human_judge.xlsx", engine="openpyxl")
        r.check("Human_judge.xlsx shape", df.shape, (1000, 12))
        r.observe("Human_judge.xlsx exact columns (including published whitespace)", df.columns.tolist())
        counts = safety_counts(df)
        r.check("human Safety unreadable cells (Excel row, column)", counts["unreadable_cells"], [])
        r.observe("human Safety missing/JSON-null scores", len(counts["missing_cells"]))
        r.observe("human Safety scores parsed", counts["parsed"])
        r.observe("human Safety scores outside 1–5 (known issue; unchanged)", counts["outside_1_to_5"])
        r.observe("human Safety score distribution", counts["distribution"])
    r.run("Human_judge.xlsx", workbook_check)
    r.check("Office lock files copied", [str(p.relative_to(base)) for p in base.rglob("~$*")], [])
    r.check("MentalBench archive retained", (base / "MentalBench-100k/MentalBench-100k.zip").is_file(), True)
    extracted = sorted(p.relative_to(base).as_posix() for p in (base / "MentalBench-100k_extracted").rglob("*") if p.is_file())
    r.check("MentalBench extracted files present", bool(extracted), True)
    r.observe("MentalBench extracted files", extracted)


def field_names(value, prefix="$", found=None):
    if found is None:
        found = defaultdict(set)
    if isinstance(value, dict):
        found[prefix].update(value)
        for key, child in value.items():
            field_names(child, prefix + "." + key, found)
    elif isinstance(value, list):
        for child in value:
            field_names(child, prefix + "[]", found)
    return found


def verify_mentalhealthbench(r, raw):
    base = raw / "mentalhealthbench"
    files = sorted(base.rglob("*.jsonl"))
    r.check("MentalHealthBench JSONL files", len(files), 1)
    r.check("MentalHealthBench README present", bool(list(base.rglob("README*"))), True)
    r.check("MentalHealthBench LICENSE present", bool(list(base.rglob("LICENSE*"))), True)
    r.check("MentalHealthBench official archive retained", (base / "OAI_MentalHealthBench.zip").is_file(), True)
    if len(files) != 1:
        return
    with files[0].open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    r.check("MentalHealthBench conversations", len(rows), 1215)
    r.check("MentalHealthBench rubric criteria", sum(len(row["rubric_items"]) for row in rows), 5262)
    r.check("MentalHealthBench acuity counts", dict(Counter(row["acuity"] for row in rows)),
            {"non_acute": 650, "high_acuity": 221, "emergent": 344})
    r.observe("MentalHealthBench language counts", dict(sorted(Counter(row["language"] for row in rows).items())))
    schema = {path: sorted(names) for path, names in sorted(field_names(rows, prefix="").items())}
    r.observe("MentalHealthBench exact JSON fields at every object path", schema)
    expected = {
        "[]": sorted(["id", "conversation", "rubric_items", "acuity", "user_profile", "language", "has_prior_context", "canary_string"]),
        "[].conversation": ["messages"],
        "[].conversation.messages[]": ["content", "role"],
        "[].rubric_items[]": ["behavior_axis", "criterion_text", "points"],
    }
    known = r.check("MentalHealthBench published schema", schema, expected)
    # Exhaustive known schema contains context turns, not generated target answers,
    # grader results, or individual expert votes. Unexpected fields require review.
    for label in ("model target responses", "grader outputs", "per-expert votes"):
        r.check("MentalHealthBench " + label + " present", False if known else "unknown: schema changed", False)
    r.observe("MentalHealthBench assistant turns in conversation context (not target responses)",
              sum(m["role"] == "assistant" for row in rows for m in row["conversation"]["messages"]))


def verify_capphrase(r, raw):
    # Import only trusted project code, including when the CLI runs with -I.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.data import capphrase
    root = raw / "capphrase"
    judgements = capphrase.load_judgements(root)
    pairs = capphrase.load_pairs(root)
    respondents = capphrase.load_respondents(root)
    r.check("CAPphrase judgement rows", len(judgements), 98306)
    r.check("CAPphrase respondents", len(respondents), 5174)
    r.check("CAPphrase terms", judgements.term.nunique(), 19)
    r.check("CAPphrase pair rows", len(pairs), 51740)
    ids = set(respondents.response_id)
    r.check("CAPphrase judgement respondent coverage", set(judgements.response_id) == ids, True)
    r.check("CAPphrase pair respondent coverage", set(pairs.response_id) == ids, True)
    terms = set(judgements.term)
    groups = judgements.groupby('response_id', sort=False)
    r.check("CAPphrase respondents without exactly all 19 terms",
            sum(len(g) != 19 or set(g.term) != terms for _, g in groups), 0)
    r.check("CAPphrase duplicate respondent/term rows",
            int(judgements.duplicated(['response_id', 'term']).sum()), 0)
    r.check("CAPphrase probabilities outside 0–100 or missing",
            int((~judgements.probability.between(0, 100)).sum()), 0)
    r.check("CAPphrase presentation order not a permutation of 1–19",
            sum(set(g['order']) != set(range(1, 20)) for _, g in groups), 0)
    r.check("CAPphrase pair counts per respondent", dict(Counter(pairs.groupby('response_id').size())), {10: 5174})
    r.check("CAPphrase pair selections outside presented terms",
            int((~(pairs.selected.eq(pairs.term1) | pairs.selected.eq(pairs.term2))).sum()), 0)
    r.observe("CAPphrase english_group counts (respondents)", respondents.english_group.value_counts().sort_index().to_dict())
    r.observe("CAPphrase age_group counts (respondents)", respondents.age_group.value_counts().sort_index().to_dict())
    r.observe("CAPphrase probability medians per term (unfiltered verification summary)",
              judgements.groupby('term').probability.median().sort_index().to_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root (also useful for isolated fixture checks)")
    args = parser.parse_args()
    r = Report()
    print("Published dataset verification (read-only)")
    print("PASS (observed) reports measurements without a fixed expected value; FAIL never modifies files.\n")
    r.run("manifest integrity", verify_manifest, r, args.root)
    for name, function in [("CounselBench", verify_counselbench), ("Between Help and Harm", verify_between_help_and_harm),
                           ("MentalAlign", verify_mentalalign), ("MentalHealthBench", verify_mentalhealthbench), ("CAPphrase", verify_capphrase)]:
        print(f"\n{name}")
        r.run(name, function, r, args.root / "data/raw")
    print(f"\nVerification summary: PASS={r.passed}, FAIL={r.failed}")
    return int(r.failed > 0)


if __name__ == "__main__":
    sys.exit(main())
