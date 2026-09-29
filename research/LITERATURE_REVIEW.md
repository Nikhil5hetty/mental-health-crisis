# Literature Review

## 1. Verily Mental Health Guardrail
Nelson et al. (2026) frame psychiatric-crisis detection as a dedicated guardrail problem rather than generic moderation.

Key contributions:
- clinician-informed crisis taxonomy
- 1,800-message Verily crisis dataset
- VMHG mental-health-specific guardrail
- external evaluation on a 794-message AEGIS-derived subset
- comparison with general-purpose moderation systems

Relevance:
The study motivates testing whether specialized mental-health guardrails generalize better than generic safety systems.

Availability:
The paper states data and code are available upon researcher request.

## 2. Between Help and Harm
Arnaiz-Rodriguez et al. (2026) examine both crisis classification and response handling.

Key contributions:
- six-category crisis taxonomy
- 206 expert-annotated validation examples
- 2,046 test examples
- source pool of more than 239,000 user messages from 12 public datasets
- general-purpose LLM classification
- downstream response-safety evaluation

Relevance:
The expert-labelled validation set is a strong public baseline for reproducible crisis-detection experiments.

## 3. AEGIS2.0
Ghosh et al. (2025) provide a broad AI-safety dataset and taxonomy:
- 34,248 human–LLM interaction samples
- 12 top-level hazard categories
- 9 finer-grained subcategories

Relevance:
AEGIS connects general content-safety research with Verily's external validation setup.

## 4. SIM-VAIL
Weilnhammer et al. (2026) provide a clinically validated multi-turn framework for auditing mental-health risks:
- 810 multi-turn conversations
- 9 chatbots
- 30 simulated user profiles
- 13 clinically grounded risk dimensions
- more than 90,000 turn-level ratings

Relevance:
SIM-VAIL shows that mental-health risk can accumulate over conversational trajectories. Our proposed extension focuses specifically on the detection layer: when does a safety system recognize that a trajectory has become a crisis?

## 5. Newer guardrail families
Relevant modern guardrails include:
- Qwen3Guard
- Llama Guard family
- NVIDIA Nemotron content-safety reasoning models

A seminar-scale project should prioritize diversity and reproducibility rather than maximizing model count.

## Research gap
A coherent remaining gap is:
**How robustly do safety systems detect mental-health crises when dataset construction, taxonomy, and conversational context change?**

This motivates:
1. cross-dataset generalization under a harmonized taxonomy
2. multi-turn crisis-onset detection under increasing context
