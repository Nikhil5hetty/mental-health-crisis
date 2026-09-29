# Option A Feasibility — Cross-Dataset Crisis Detection

## Goal

Establish whether the public artifacts from [Between Help and Harm](https://doi.org/10.2196/88435) are reproducible enough to serve as the baseline for the cross-dataset guardrail study.

This first experiment reproduces the paper's **validation-set crisis-label agreement** using the authors' committed outputs, before spending compute/API budget on fresh inference.

## Source artifacts

- [Paper](https://doi.org/10.2196/88435)
- [Authors' GitHub repository](https://github.com/ellisalicante/LLMs-Mental-Health-Crisis)
- [Hugging Face benchmark](https://huggingface.co/datasets/arnaiztech/llms-mental-health-crisis-benchmark)

The authors' validation setup contains:
- 206 validation conversations
- 4 human expert annotation files
- 3 runs of GPT-4o-mini
- 3 runs of GPT-5-nano
- 3 runs of Llama-4-Scout-17B-16E-Instruct

The paper computes pairwise Cohen's kappa between each model run and each human annotator, then averages across runs and human raters.

## Feasibility run performed

The committed annotation JSON files were fetched from the authors' public repository and the paper's agreement calculation was independently recomputed.

### Reproduced results

| Rater family | Paper | Reproduced |
|---|---:|---:|
| GPT-4o-mini | 0.645 | 0.6445 |
| GPT-5-nano | 0.631 | 0.6306 |
| Llama-4-Scout | 0.581 | 0.5810 |
| Human-human mean pairwise κ | ~0.553 | 0.5530 |

Raw mean agreement:
- GPT-4o-mini: 73.62%
- GPT-5-nano: 72.90%
- Llama-4-Scout: 68.08%
- Human-human: 65.86%

The results reproduce the published values to the paper's reported rounding precision.

## What this establishes

### PASS — public labels are internally reproducible
The public human and model annotation artifacts are aligned sufficiently to reproduce the main classifier-selection result.

### PASS — no model inference is required for baseline reproduction
The first baseline can be reproduced from stored outputs, so API availability is not a blocker for verifying the paper's reported agreement.

### PASS — exact classification prompt is public
The prompt used by the authors is present in `src/llm_prompting.py` in the source repository.

## Authors' environment

The public `environment.yml` specifies:
- Python 3.12
- openai
- datasets
- scikit-learn
- python-dotenv
- regex
- tabulate
- groq
- openpyxl
- statsmodels

Fresh classification inference additionally requires credentials configured in `.env`:
- `OPEN_AI_API_KEY`
- `GROQ_API_KEY`
- other provider keys for response experiments

## Current blockers for a full fresh-inference reproduction

### B1 — proprietary API access
GPT-4o-mini and GPT-5-nano require OpenAI API access and incur usage cost.

**Impact:** does not block reproduction of stored results; blocks regenerating the model labels from scratch.

### B2 — provider/model persistence
The original Llama-4-Scout run is routed through Groq in the authors' code. Availability, exact hosted revision, and inference behavior may change over time.

**Impact:** a fresh rerun may not be bit-for-bit comparable even with the same prompt.

### B3 — stochasticity is not explicitly pinned
The authors call chat-completions without explicitly setting temperature, seed, or other sampling parameters in the public classification function.

**Impact:** exact reruns may differ from the committed outputs; three repeated runs partially address this but do not make inference deterministic.

### B4 — environment is not fully locked
The repository provides `environment.yml` but does not pin exact package versions.

**Impact:** setup should be straightforward, but exact dependency-level reproducibility is weaker than a lockfile-based environment.

### B5 — source corpus regeneration is heavier than benchmark reuse
The authors' full construction path merges multiple Hugging Face datasets. For Option A, rebuilding all ~239k inputs is unnecessary if our goal is detector evaluation on the released benchmark.

**Decision:** use the released 206-example validation benchmark as the first gold-standard baseline rather than regenerating the entire corpus.

### B6 — human gold standard is not a single unquestionable label
Human-human agreement is only moderate (mean pairwise κ ≈ 0.553).

**Impact:** model-vs-ground-truth comparisons must acknowledge label uncertainty. Per-annotator or consensus sensitivity analyses may be warranted.

### B7 — taxonomy mismatch with guardrails
The BHH taxonomy is not identical to NeMoGuard, Qwen3Guard, Llama Guard, AEGIS, or VMHG.

**Impact:** a frozen mapping specification is required before cross-model evaluation.

## Next Option A feasibility step

1. Freeze a BHH-to-common-taxonomy mapping.
2. Extract the human-labelled BHH-206 common-taxonomy subset.
3. Run one open guardrail locally without proprietary API dependence.
4. Verify output parsing and taxonomy mapping.
5. Add a second guardrail.
6. Only after the pipeline is stable, add proprietary moderation/LLM baselines.
7. Add AEGIS as an external dataset.
8. Add Verily/VMHG only if research access is granted.

## Reproduce locally

Run:

```bash
python experiments/option_a/reproduce_bhh_agreement.py
```

The script downloads only the public annotation JSON files required for the validation agreement and computes the reported statistics.

No API key is required for this reproduction step.

## Expected output

```text
gpt-4o-mini     kappa ~= 0.6445    agreement ~= 73.62%
gpt-5-nano      kappa ~= 0.6306    agreement ~= 72.90%
llama-4-scout   kappa ~= 0.5810    agreement ~= 68.08%
human-human     kappa ~= 0.5530    agreement ~= 65.86%
```


## Qwen3Guard taxonomy feasibility finding

The first extension uses [Qwen3Guard](https://arxiv.org/abs/2510.14276).

Qwen3Guard's current safety taxonomy exposes a combined **Suicide & Self-Harm** category rather than separate suicide and self-harm labels. It does not expose a direct substance-misuse/withdrawal crisis category. Therefore the full five-class harmonized taxonomy cannot be evaluated against Qwen3Guard without inventing mappings.

For the initial feasibility experiment, the comparison is intentionally restricted to the direct overlap:

| BHH expert-consensus label | Qwen3Guard target |
|---|---|
| `suicidal_ideation` | `Suicide & Self-Harm` |
| `self-harm` | `Suicide & Self-Harm` |
| `no_crisis` | absence of `Suicide & Self-Harm` |

The BHH expert-consensus validation file contains:
- 94 `no_crisis`
- 34 `suicidal_ideation`
- 15 `self-harm`
- 16 unresolved/null consensus labels
- other crisis labels outside this Qwen smoke-test overlap

Thus the full directly compatible binary subset is **143 records**: 49 positive and 94 negative.

### Important semantic mismatch

Even this binary mapping is not perfectly equivalent. BHH defines crisis categories clinically, while Qwen3Guard's safety category is oriented toward harmful content such as advocating, encouraging, or detailing self-harm/suicide. A user *expressing* suicidal ideation may therefore be clinically positive in BHH but not necessarily unsafe under Qwen3Guard's moderation policy.

This is not a data-cleaning nuisance; it is a central research question:

> Can general-purpose content-safety guardrails detect user crisis state, or are their taxonomies optimized for harmful-content intent rather than clinical crisis recognition?

## Current executable smoke test

`run_qwen3guard_bhh.py` implements this mapping.

Because GitHub-hosted CI is CPU-only, the automated feasibility workflow uses **Qwen3Guard-Gen-0.6B** and a deterministic 12-record stratified smoke test (4 records each from no-crisis, suicidal ideation, and self-harm).

If this passes end-to-end, the exact same pipeline can be scaled to:
1. the full 143-record directly compatible subset;
2. Qwen3Guard-Gen-8B on GPU compute.

The smaller checkpoint is a pipeline feasibility check, not a substitute for the final 8B experiment.

### GitHub Actions

Workflow: `.github/workflows/option-a-qwen-smoke.yml`

The workflow:
1. installs the required Transformer stack;
2. downloads Qwen3Guard-Gen-0.6B;
3. downloads the public BHH human-consensus file;
4. runs deterministic inference;
5. writes `results/option_a_qwen_smoke.json`;
6. uploads the result as a workflow artifact.
