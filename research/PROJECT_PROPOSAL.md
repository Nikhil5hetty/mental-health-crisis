# Project Proposal

## Working title
**Evaluating LLM Safety Systems for Mental Health Crisis Detection Across Datasets and Conversational Contexts**

## Foundation papers
- [Nelson et al. (2026), *An AI-based mental health guardrail and dataset for identifying psychiatric crises in text-based conversations*](https://doi.org/10.1038/s41746-026-02579-5)
- [Arnaiz-Rodriguez et al. (2026), *Between Help and Harm: An Evaluation Study of Mental Health Crisis Handling by Large Language Models*](https://doi.org/10.2196/88435)

## Motivation
Recent work has examined psychiatric-crisis detection through both specialized guardrails and general-purpose LLMs. Two important gaps remain: cross-dataset generalization and longitudinal/multi-turn detection.

## Objective
Develop a reproducible evaluation framework comparing general-purpose LLM classifiers, dedicated safety guardrails, and—if accessible—a mental-health-specific guardrail across harmonized datasets and multi-turn contexts.

## Research questions
1. How well do crisis detectors generalize across independently constructed datasets?
2. How does performance vary across common crisis categories?
3. Does conversational context improve detection?
4. How early and stably do systems detect emerging crises?
5. How robust are systems when explicit crisis cues are absent?

## Scope
Included:
- text-based crisis detection
- common taxonomy
- open/reproducible guardrails as core baseline
- multi-turn suicide/self-harm as preferred longitudinal study

Excluded initially:
- diagnosis
- therapeutic chatbot construction
- full treatment-quality assessment
- large synthetic clinical corpora without expert review

## Harmonized taxonomy
1. No crisis
2. Suicide / suicidal ideation
3. Self-harm
4. Substance misuse / abuse / withdrawal
5. Violence toward others / violent thoughts

## Datasets
- BHH expert-labelled validation subset (n=206)
- BHH test subset (n=2,046)
- [AEGIS2.0](https://aclanthology.org/2025.naacl-long.306/) compatible subset
- Verily Mental Health Crisis Dataset v1.0 (n=1,800), if granted
- Verily clinician-reviewed NVIDIA subset (n=794), if granted

## Model groups
### A. General-purpose LLM classifiers
Use at least one BHH-compatible classifier setup with a frozen prompt and exact model identifier.

### B. Dedicated safety guardrails
- NeMoGuard
- Qwen3Guard-Gen-8B
- Llama Guard
- optional Nemotron Content Safety Reasoning
- optional OpenAI moderation

### C. Mental-health-specific guardrail
- VMHG, if provided by Verily.

## Direction 1: Cross-dataset generalization
Apply identical detector configurations to harmonized subsets from multiple datasets.

Metrics:
- recall/sensitivity
- specificity
- precision
- F1
- FNR/FPR
- per-category recall
- confidence intervals

## Direction 2: Multi-turn context-aware detection
Evaluate conversation prefixes under:
1. current user message only
2. short recent user-history window
3. full user-message history

Potential labels:
- no detectable crisis
- ambiguous/concerning
- identifiable crisis

Primary measures:
- classification performance
- detection delay
- post-onset consistency
- pre-onset false positives
- explicit-cue ablation

## Main risks
- Verily access unavailable
- taxonomy mismatch
- lack of clinically credible onset labels
- incompatible guardrail output taxonomies
- proprietary model drift
- synthetic-data artifacts
- statistical dependence between turns
- ethics/privacy requirements

## Mitigation
- baseline entirely from public data/open systems
- freeze mappings before inference
- use conversation as statistical unit
- separate human-labelled primary analysis from LLM-labelled secondary analysis
- record exact model/checkpoint/prompt/configuration/date

## Milestones
### Phase A — Foundation
- finalize RQs
- literature review
- harmonized taxonomy
- reproduce BHH-206 processing

### Phase B — Public baseline
- integrate NeMoGuard
- integrate Qwen3Guard
- add second guardrail
- evaluate on BHH-206

### Phase C — Cross-dataset
- BHH-2046 secondary analysis
- AEGIS subset
- compare generalization

### Phase D — Optional Verily extension
- add Verily-1800
- add NVIDIA-794
- add VMHG
- quantify effect of clinician relabelling

### Phase E — Context study
- identify/recover multi-turn trajectories
- establish onset annotation protocol
- run current-only/short-history/full-history experiments
- run explicit-cue ablation
- analyse delay and stability

### Phase F — Final report
- statistical analysis
- failure taxonomy
- limitations/reproducibility statement
