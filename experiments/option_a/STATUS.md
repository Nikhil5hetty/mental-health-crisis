# Option A Feasibility Status

## Status

### Baseline reproduction — PASS
The public *Between Help and Harm* annotation artifacts reproduce the paper's reported validation agreement to rounding.

| Rater family | Reproduced mean Cohen κ |
|---|---:|
| GPT-4o-mini | 0.6445 |
| GPT-5-nano | 0.6306 |
| Llama-4-Scout | 0.5810 |
| Human-human | 0.5530 |

Paper: https://doi.org/10.2196/88435

### Qwen3Guard extension — RUNNING / PIPELINE FEASIBILITY

The first guardrail extension has been implemented using:
- BHH expert-consensus validation data
- directly compatible labels only:
  - no_crisis
  - suicidal_ideation
  - self-harm
- Qwen3Guard-Gen-0.6B for CPU CI smoke testing
- deterministic generation
- direct category mapping to Qwen's combined `Suicide & Self-Harm` category

Qwen3Guard paper: https://arxiv.org/abs/2510.14276

The GitHub Actions workflow has successfully started. At the latest check it had completed runner setup, checkout, Python setup, and cache initialization and was installing inference dependencies.

## Blockers found

1. Qwen3Guard combines suicide and self-harm into one category.
2. Qwen3Guard has no direct substance-use crisis category.
3. Qwen's safety policy is aimed at harmful content, while BHH labels user clinical crisis state.
4. The current local assistant runtime is CPU-only and cannot install packages from the network; CI is therefore used for executable model testing.
5. Qwen3Guard-8B should be treated as a GPU experiment; 0.6B is only the pipeline smoke test.
6. Sixteen BHH-206 consensus records have null/unresolved labels and must not be silently treated as no-crisis.

## Decision gate

Option A remains feasible if the Qwen smoke test can:
- load the public model,
- parse outputs consistently,
- score the BHH overlap subset without runtime failures.

Even if clinical recall is poor, that would be a research result rather than a feasibility failure, because the taxonomy/policy mismatch is itself part of the research question.
