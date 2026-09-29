# Methodology

## Study design
Two tracks:
- Track A: cross-dataset generalization
- Track B: multi-turn crisis detection

## Unit of analysis
- Single-turn: one user message
- Multi-turn: one conversation trajectory as the primary independent unit

## Harmonized taxonomy
| ID | Class |
|---|---|
| 0 | No crisis |
| 1 | Suicide / suicidal ideation |
| 2 | Self-harm |
| 3 | Substance misuse / abuse / withdrawal |
| 4 | Violence toward others / violent thoughts |

Mappings must be frozen before inference.

## Dataset hierarchy
1. BHH-206 expert-labelled validation subset — primary
2. BHH-2046 — secondary
3. AEGIS2.0 compatible subset — external
4. Verily-1800 — optional
5. clinician-reviewed NVIDIA-794 — optional

## Detector families
### General LLM classifier
Record provider, exact model ID, prompts, temperature, seed if available, date, repetitions.

### Dedicated guardrails
Candidate systems:
- NeMoGuard
- Qwen3Guard-Gen
- Llama Guard
- Nemotron Content Safety Reasoning

### VMHG
Use only if sufficient implementation details are provided.

## Output normalization
Each model requires a predeclared mapping from native output to harmonized output. Mapping rules must be version-controlled.

## Track A experiments
### A1 — Binary crisis detection
Metrics:
- sensitivity
- specificity
- precision
- F1
- FPR
- FNR

### A2 — Category-level detection
Metrics:
- macro F1
- per-class recall
- per-class precision
- confusion matrix

### A3 — Cross-dataset robustness
Run identical configurations across datasets and measure absolute performance plus performance drift.

### A4 — Clinician-relabelling sensitivity
If exact Verily NVIDIA records and IDs are available, score identical predictions against original and clinician-corrected labels.

## Track B setup
Preferred scope:
- suicide
- self-harm

Target scale:
- 30–50 trajectories
- 6–10 turns each

Potential states:
- 0 no detectable crisis
- 1 ambiguous/concerning
- 2 identifiable crisis

## Context ablation
- C1 current message only
- C2 short user-history window
- C3 full user history

## Detection delay
DetectionDelay = T_model - T_reference

Interpretation:
- 0 = aligned with reference onset
- >0 = delayed
- <0 = early/premature

Always interpret together with pre-onset false-positive rate.

## Post-onset consistency
PostOnsetConsistency =
positive predictions at/after T_ref / evaluated turns at/after T_ref

This is an experimental metric, not a clinically validated one.

## Explicit-cue ablation
Compare:
- original trajectory
- same trajectory with the most explicit crisis cue removed or softened while preserving context

Rewritten examples should be independently reviewed.

## Annotation
Preferred:
- at least 2 independent annotators
- ideally clinical/domain-expert onset validation
- adjudication before model results are inspected

## Statistics
- paired comparisons for single-turn experiments
- bootstrap/resample at conversation level for multi-turn experiments
- confidence intervals for primary metrics

## Reproducibility
Store:
- model/checkpoint
- tokenizer
- inference library
- prompt
- parameters
- seed
- hardware
- timestamp
- dataset version
- label-map version
- raw output
- normalized prediction

## Minimum viable study
1. BHH-206
2. harmonized binary + common categories
3. NeMoGuard + Qwen3Guard + one Llama Guard
4. cross-model comparison
5. 20–30 multi-turn suicide/self-harm trajectories
6. current-message vs full-history
7. detection delay + FNR + false-positive analysis
