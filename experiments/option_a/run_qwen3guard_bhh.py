#!/usr/bin/env python3
"""Qwen3Guard feasibility smoke test on the BHH-206 expert-consensus dataset.\n\nCI note: this script is intentionally small enough for a CPU smoke test.

Primary purpose:
- Verify that an open guardrail can be run end-to-end on the public BHH data.
- Test the only clean taxonomy overlap for Qwen3Guard:
    no_crisis vs (suicidal_ideation OR self-harm)

This is NOT the final benchmark. By default it uses a deterministic stratified
smoke-test sample (12 per class) and Qwen3Guard-Gen-0.6B so it can run on a
CPU-only CI runner. The final study should repeat the same pipeline with 8B
on suitable GPU compute and then run the full compatible subset.

References:
- BHH: https://doi.org/10.2196/88435
- Qwen3Guard: https://arxiv.org/abs/2510.14276
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BHH_URL = (
    "https://raw.githubusercontent.com/ellisalicante/"
    "LLMs-Mental-Health-Crisis/main/data/human_label/"
    "human-labeled-sampled_dataset_n206_s42-merged_labels.json"
)

POSITIVE_LABELS = {"suicidal_ideation", "self-harm"}
NEGATIVE_LABEL = "no_crisis"

SAFE_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.I)
CATEGORY_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|"
    r"PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)",
    re.I,
)


def load_bhh() -> list[dict]:
    with urllib.request.urlopen(BHH_URL) as response:
        return json.load(response)


def eligible_records(data: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(data):
        label = row.get("label")
        if label in POSITIVE_LABELS or label == NEGATIVE_LABEL:
            out.append(
                {
                    "source_index": i,
                    "inputs": row["inputs"],
                    "source_label": label,
                    "gold_binary": 1 if label in POSITIVE_LABELS else 0,
                }
            )
    return out


def stratified_sample(records: list[dict], n_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {
        "no_crisis": [],
        "suicidal_ideation": [],
        "self-harm": [],
    }
    for row in records:
        groups[row["source_label"]].append(row)

    selected = []
    for label, rows in groups.items():
        rows = list(rows)
        rng.shuffle(rows)
        selected.extend(rows[: min(n_per_class, len(rows))])

    rng.shuffle(selected)
    return selected


def parse_output(text: str) -> tuple[str | None, list[str]]:
    safety = SAFE_RE.search(text)
    cats = CATEGORY_RE.findall(text)
    return (safety.group(1).title() if safety else None, cats)


def confusion(rows: list[dict]) -> dict:
    tp = sum(r["gold_binary"] == 1 and r["pred_binary"] == 1 for r in rows)
    tn = sum(r["gold_binary"] == 0 and r["pred_binary"] == 0 for r in rows)
    fp = sum(r["gold_binary"] == 0 and r["pred_binary"] == 1 for r in rows)
    fn = sum(r["gold_binary"] == 1 and r["pred_binary"] == 0 for r in rows)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (tn + fp) if tn + fp else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3Guard-Gen-0.6B")
    parser.add_argument("--n-per-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--full-compatible-subset",
        action="store_true",
        help="Run all BHH records with labels no_crisis, suicidal_ideation, or self-harm.",
    )
    parser.add_argument("--output", default="results/option_a_qwen_smoke.json")
    args = parser.parse_args()

    raw = load_bhh()
    compatible = eligible_records(raw)
    records = (
        compatible
        if args.full_compatible_subset
        else stratified_sample(compatible, args.n_per_class, args.seed)
    )

    print("BHH total:", len(raw))
    print("Compatible subset:", len(compatible))
    print("Compatible label distribution:", Counter(r["source_label"] for r in compatible))
    print("Evaluation sample:", len(records))
    print("Evaluation distribution:", Counter(r["source_label"] for r in records))
    print("Model:", args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    results = []
    for idx, row in enumerate(records, 1):
        # Preserve BHH's conversation-as-list meaning without injecting new content.
        prompt = "\n".join(row["inputs"])
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated = output[0][encoded["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        safety, categories = parse_output(text)

        # The only defensible direct category mapping for this first experiment:
        # BHH suicidal_ideation/self-harm -> Qwen "Suicide & Self-Harm".
        pred_positive = any(c.lower() == "suicide & self-harm" for c in categories)

        result = {
            **row,
            "qwen_safety": safety,
            "qwen_categories": categories,
            "pred_binary": int(pred_positive),
            "raw_output": text,
        }
        results.append(result)
        print(
            f"[{idx:03d}/{len(records):03d}] "
            f"gold={row['source_label']:<18} "
            f"safety={str(safety):<13} "
            f"categories={categories}"
        )

    metrics = confusion(results)
    payload = {
        "paper": "https://doi.org/10.2196/88435",
        "guardrail_paper": "https://arxiv.org/abs/2510.14276",
        "model": args.model,
        "mapping": {
            "BHH suicidal_ideation": "Qwen Suicide & Self-Harm",
            "BHH self-harm": "Qwen Suicide & Self-Harm",
            "BHH no_crisis": "absence of Qwen Suicide & Self-Harm category",
        },
        "compatible_subset_size": len(compatible),
        "sample_size": len(results),
        "sample_distribution": dict(Counter(r["source_label"] for r in results)),
        "metrics": metrics,
        "results": results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nMetrics")
    print(json.dumps(metrics, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
