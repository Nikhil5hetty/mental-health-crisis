# Mental Health Crisis Detection in LLMs

## Working title
**Evaluating LLM Safety Systems for Mental Health Crisis Detection Across Datasets and Conversational Contexts**

## Project overview
This seminar project studies how reliably LLM safety systems identify mental-health crises, with two complementary research directions:

1. **Cross-dataset generalization** — whether crisis-detection performance transfers across independently constructed datasets when labels are harmonized.
2. **Context-aware / multi-turn detection** — whether accumulating conversational context improves the accuracy, timing, and stability of crisis detection.

The project is motivated by:
- Nelson et al. (2026), *An AI-based mental health guardrail and dataset for identifying psychiatric crises in text-based conversations*.
- Arnaiz-Rodriguez et al. (2026), *Between Help and Harm: An Evaluation Study of Mental Health Crisis Handling by Large Language Models*.

A core design principle is that the project should remain executable without request-only artifacts. Public datasets and open-weight guardrails form the baseline; Verily materials strengthen the study if access is granted.

## Research questions
- **RQ1:** How well do dedicated safety guardrails and general-purpose LLM classifiers generalize across independently constructed mental-health crisis datasets under a harmonized taxonomy?
- **RQ2:** Does increasing conversational context improve crisis-detection performance compared with evaluating the current message alone?
- **RQ3:** How early relative to an annotated crisis-onset point do different guardrails detect an emerging crisis?
- **RQ4:** How robust is detection when explicit suicide/self-harm cues are absent but contextual indicators accumulate across turns?

## Harmonized taxonomy
Initial shared categories:
- No crisis
- Suicide / suicidal ideation
- Self-harm
- Substance misuse / abuse / withdrawal
- Violence toward others / violent thoughts

## Candidate datasets
- **Between Help and Harm:** 206 expert-annotated validation examples + 2,046 larger test examples.
- **Verily Mental Health Crisis Dataset v1.0:** 1,800 clinician-reviewed simulated messages; available on researcher request.
- **Clinician-reviewed NVIDIA subset used by Verily:** 794 suicide/self-harm-related messages with clinician corrections.
- **AEGIS2.0:** public general safety dataset with 34,248 human–LLM interaction samples.

## Candidate model groups
### General-purpose LLM classifiers
At least one model/setup from *Between Help and Harm* for continuity.

### Dedicated guardrails
- NVIDIA NeMoGuard
- Qwen3Guard-Gen-8B
- Llama Guard family
- optional Nemotron Content Safety Reasoning
- optional OpenAI moderation baseline

### Mental-health-specific guardrail
- Verily Mental Health Guardrail (VMHG), if code/prompts/configuration are provided.

## Direction 1 — Cross-dataset evaluation
Evaluate the same detector families across harmonized datasets.

Primary metrics:
- sensitivity / recall
- specificity
- precision
- F1
- false-positive rate
- false-negative rate
- per-category recall
- confidence intervals and paired comparisons

## Direction 2 — Multi-turn context-aware detection
Evaluate evolving conversations using:
- current message only
- short user-history window
- full user-message history

Potential labels:
- 0 = no detectable crisis
- 1 = ambiguous / concerning
- 2 = identifiable crisis

Key measurements:
- classification metrics
- detection delay
- post-onset consistency
- explicit-cue ablation
- context-length ablation

## Repository structure
```
mental-health-crisis/
├── README.md
├── research/
│   ├── PROJECT_PROPOSAL.md
│   ├── LITERATURE_REVIEW.md
│   ├── METHODOLOGY.md
│   └── VERILY_DATA_REQUEST.md
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── src/
├── experiments/
├── notebooks/
└── results/
```

## Initial milestones
1. Finalize the harmonized taxonomy and exclusion rules.
2. Reproduce BHH-206 label processing and baseline analysis.
3. Run two open guardrails on the harmonized BHH subset.
4. Add at least one newer guardrail.
5. Build an AEGIS-compatible external validation subset.
6. Request Verily datasets and VMHG implementation.
7. If access is granted, perform direct cross-dataset evaluation.
8. Develop or recover a small multi-turn suicide/self-harm benchmark.
9. Run context-length and detection-delay experiments.
10. Complete failure analysis and final seminar report.

## Key references
- Nelson, B. W. et al. (2026). *An AI-based mental health guardrail and dataset for identifying psychiatric crises in text-based conversations*. **npj Digital Medicine**, 9, 407. https://doi.org/10.1038/s41746-026-02579-5
- Arnaiz-Rodriguez, A. et al. (2026). *Between Help and Harm: An Evaluation Study of Mental Health Crisis Handling by Large Language Models*. **JMIR Mental Health**, 13, e88435. https://doi.org/10.2196/88435
- Ghosh, S. et al. (2025). *AEGIS2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails*. **NAACL 2025**.
- Weilnhammer, V. et al. (2026). *A clinically validated framework for auditing AI chatbot behavior in mental health interactions*. **Nature Medicine**.

## Research ethics
This repository is for model-safety research. Crisis examples should be handled as research data, not clinical advice or diagnostic outputs. Any new dataset involving human annotators, clinicians, or sensitive real-world conversations should follow the university's ethics and privacy requirements.
