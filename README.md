# Mental Health Crisis Detection in LLMs

## Working title
**Evaluating LLM Safety Systems for Mental Health Crisis Detection Across Datasets and Conversational Contexts**

## Project overview
This seminar project studies how reliably LLM safety systems identify mental-health crises, with two complementary research directions:

1. **Cross-dataset generalization** — whether crisis-detection performance transfers across independently constructed datasets when labels are harmonized.
2. **Context-aware / multi-turn detection** — whether accumulating conversational context improves the accuracy, timing, and stability of crisis detection.

The project is motivated by:
- [Nelson et al. (2026), *An AI-based mental health guardrail and dataset for identifying psychiatric crises in text-based conversations*](https://doi.org/10.1038/s41746-026-02579-5).
- [Arnaiz-Rodriguez et al. (2026), *Between Help and Harm: An Evaluation Study of Mental Health Crisis Handling by Large Language Models*](https://doi.org/10.2196/88435).

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
- optional Nemotron content-safety reasoning
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

## Option A feasibility work
The first reproducibility experiment is documented in [experiments/option_a/README.md](experiments/option_a/README.md).

Using the public annotation artifacts from the *Between Help and Harm* repository, the paper's validation agreement was reproduced to rounding:
- GPT-4o-mini: Cohen κ ≈ **0.645**
- GPT-5-nano: Cohen κ ≈ **0.631**
- Llama-4-Scout: Cohen κ ≈ **0.581**
- Human-human mean pairwise Cohen κ ≈ **0.553**

## LLM word elicitation (WP2)
`experiments/words/elicit_llm.py` asks local open-weight models what probability each of the 19 CAPphrase terms conveys. It follows the RQ2 section of [docs/ANALYSIS_PLAN_WORDS.md](docs/ANALYSIS_PLAN_WORDS.md).

**What it does**
- Requests 4 models × 2 prompts (P1, P2) × 19 terms × 50 samples = 7,600 answers.
- Uses temperature 1.0, with seed 20261124 plus the sample index.
- Accepts a reply only if it is a bare number, optionally followed by `%` and a full stop (for example `70`, `70%`, `72.5`, `70.`), and the number lies between 0 and 100 (D19). Replies such as `## Step 1: ...`, `70-80` or `about 70` are invalid.
- If a reply is invalid, it asks the same prompt again in the same conversation, up to 2 times. After that the answer is recorded as missing.
- Stores Ollama's `done_reason` for the final attempt. `length` means the reply hit the 64-token cap. Files written before this column existed get it added on their next use, empty for the old rows (D18).
- Writes to `data/llm_answers/answers.csv`. This file is generated by us, so it may be committed.

**Install**
1. Install Ollama from https://ollama.com/download, or on macOS run `brew install ollama`.
2. Start the server with `ollama serve`, or open the Ollama app. It listens on `http://localhost:11434`.
3. Pull the models listed in `experiments/words/models.yaml` (about 20 GB of disk in total):
   ```
   ollama pull llama3.1:8b
   ollama pull qwen2.5:7b
   ollama pull gemma2:9b
   ollama pull mistral:7b
   ```
4. Install the Python dependency with `.venv/bin/pip install -r requirements.txt`. This adds PyYAML.

**Run**
```
.venv/bin/python -I experiments/words/elicit_llm.py --dry-run              # fake model, writes data/processed/llm_answers_dryrun.csv
.venv/bin/python -I experiments/words/elicit_llm.py --models llama3.1:8b --samples 2 --out /tmp/smoke.csv   # smoke test
.venv/bin/python -I experiments/words/elicit_llm.py                        # full run
```

**Resuming and model versions**
- The run is resumable: rows already in the CSV are skipped. If the run stops, rerun the same command.
- Each tag's digest is read from `/api/tags` and stored on every row.
- The script refuses to append answers for a tag whose digest has changed. Don't run `ollama pull` again in the middle of a study.

**Answers collected before D19.** The first harness took the first number in a reply, so a verbose reply such as `## Step 1: ...` was stored as 1. Run this once on such a file:
```
.venv/bin/python -I experiments/words/elicit_llm.py --requeue-invalid
```
- It moves every accepted row whose stored reply fails the strict rule to `data/llm_answers/superseded.csv`. The row is copied unchanged and a `reason` column is added. That file is append-only.
- It prints how many rows it moved per model, prompt and term.
- A normal run afterwards requests exactly the moved rows again.
- Rows that were already missing stay where they are. All their attempts failed the looser rule, so they also fail the strict one.

**Expected runtime.** The total has not been measured yet. Each answer is a few tokens, so a 7–9B model on an Apple-silicon laptop should take roughly 0.5–1.5 s per request. With re-asks, expect about 1–3 hours for all 7,600 answers. The script prints its progress and an ETA every 100 rows.

## Word-meaning lab (WP6)

An interactive Streamlit prototype for the word-meaning study. It has two pages: *What does a word mean?*
(citizen vs LLM word models) and a *Crisis-line simulator* (does the word mismatch flip the bot's action?). It
reads only the committed aggregate results in `results/words/`.

```bash
.venv/bin/streamlit run app/word_lab.py
```

Installation manual, user manual and known limitations: [app/README.md](app/README.md). Usability
self-evaluation: [docs/USABILITY.md](docs/USABILITY.md).

## Repository structure
```
mental-health-crisis/
├── README.md
├── research/
│   ├── PROJECT_PROPOSAL.md
│   ├── LITERATURE_REVIEW.md
│   ├── METHODOLOGY.md
│   ├── REFERENCES.md
│   └── VERILY_DATA_REQUEST.md
├── experiments/
│   └── option_a/
│       ├── README.md
│       └── reproduce_bhh_agreement.py
├── data/
│   └── README.md
├── src/
├── notebooks/
└── results/
```

## Key references
- [Nelson, B. W. et al. (2026). *An AI-based mental health guardrail and dataset for identifying psychiatric crises in text-based conversations*. npj Digital Medicine, 9, 407.](https://doi.org/10.1038/s41746-026-02579-5)
- [Arnaiz-Rodriguez, A. et al. (2026). *Between Help and Harm: An Evaluation Study of Mental Health Crisis Handling by Large Language Models*. JMIR Mental Health, 13, e88435.](https://doi.org/10.2196/88435)
- [Ghosh, S. et al. (2025). *AEGIS2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails*. NAACL 2025.](https://aclanthology.org/2025.naacl-long.306/)
- [Weilnhammer, V. et al. (2026). *A clinically validated framework for auditing AI chatbot behavior in mental health interactions*. Nature Medicine.](https://doi.org/10.1038/s41591-026-04577-2)

## Research ethics
This repository is for model-safety research. Crisis examples should be handled as research data, not clinical advice or diagnostic outputs. Any new dataset involving human annotators, clinicians, or sensitive real-world conversations should follow the university's ethics and privacy requirements.
