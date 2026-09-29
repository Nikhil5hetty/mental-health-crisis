#!/usr/bin/env python3
"""Reproduce Table 1 classifier-agreement aggregates from Between Help and Harm.

Paper:
https://doi.org/10.2196/88435

Source repository:
https://github.com/ellisalicante/LLMs-Mental-Health-Crisis

This script uses only Python's standard library and downloads the public
annotation JSON files committed by the paper authors. It does NOT perform
fresh model inference.
"""

from __future__ import annotations

import json
import math
import urllib.request
from itertools import combinations

BASE = "https://raw.githubusercontent.com/ellisalicante/LLMs-Mental-Health-Crisis/main/"

HUMANS = [
    "data/human_label/H1_labeled_n206_s42.json",
    "data/human_label/H2_labeled_n206_s42.json",
    "data/human_label/H3_labeled_n206_s42.json",
    "data/human_label/H4_labelled_n206_s42.json",
]

MODELS = {
    "gpt-4o-mini": [
        "data/llm_label/gpt-4o-mini-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-20250818-164149.json",
        "data/llm_label/gpt-4o-mini-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-20250819-120514.json",
        "data/llm_label/gpt-4o-mini-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-20250819-121533.json",
    ],
    "gpt-5-nano": [
        "data/llm_label/gpt-5-nano-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-20250819-122630.json",
        "data/llm_label/gpt-5-nano-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-20250819-123814.json",
        "data/llm_label/gpt-5-nano-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-20250819-124909.json",
    ],
    "llama-4-scout": [
        "data/llm_label/meta-llama-Llama-4-Scout-17B-16E-Instruct-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-20250820-130636.json",
        "data/llm_label/meta-llama-Llama-4-Scout-17B-16E-Instruct-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-20250820-130838.json",
        "data/llm_label/meta-llama-Llama-4-Scout-17B-16E-Instruct-labeled-sampled_dataset_n_200_merged_n50-noSeed_156-s42-20250820-131430.json",
    ],
}


def download_labels(path: str) -> list[str]:
    with urllib.request.urlopen(BASE + path) as response:
        records = json.load(response)
    return [row["label"] for row in records]


def cohen_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")

    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n

    a_counts: dict[str, int] = {}
    b_counts: dict[str, int] = {}
    for x, y in zip(a, b):
        a_counts[x] = a_counts.get(x, 0) + 1
        b_counts[y] = b_counts.get(y, 0) + 1

    labels = set(a_counts) | set(b_counts)
    expected = sum(
        (a_counts.get(label, 0) / n) * (b_counts.get(label, 0) / n)
        for label in labels
    )

    kappa = (observed - expected) / (1 - expected)
    return kappa, observed


def mean(values):
    return sum(values) / len(values)


def population_sd(values):
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def main():
    human_labels = [download_labels(path) for path in HUMANS]

    print("Between Help and Harm validation agreement reproduction")
    print("Paper: https://doi.org/10.2196/88435")
    print()

    for model_name, run_paths in MODELS.items():
        kappas = []
        agreements = []

        for run_path in run_paths:
            run_labels = download_labels(run_path)
            for human in human_labels:
                kappa, agreement = cohen_kappa(run_labels, human)
                kappas.append(kappa)
                agreements.append(agreement)

        print(
            f"{model_name:17s} "
            f"kappa={mean(kappas):.4f} "
            f"sd={population_sd(kappas):.4f} "
            f"agreement={100 * mean(agreements):.2f}%"
        )

    human_kappas = []
    human_agreements = []
    for a, b in combinations(human_labels, 2):
        kappa, agreement = cohen_kappa(a, b)
        human_kappas.append(kappa)
        human_agreements.append(agreement)

    print(
        f"{'human-human':17s} "
        f"kappa={mean(human_kappas):.4f} "
        f"sd={population_sd(human_kappas):.4f} "
        f"agreement={100 * mean(human_agreements):.2f}%"
    )


if __name__ == "__main__":
    main()
