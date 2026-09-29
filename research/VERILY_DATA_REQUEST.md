# Verily Research Data / Code Request

Paper: [Nelson et al. (2026), *An AI-based mental health guardrail and dataset for identifying psychiatric crises in text-based conversations*](https://doi.org/10.1038/s41746-026-02579-5)

## Requested artifacts

### 1. Verily Mental Health Crisis Dataset v1.0
Request:
- all 1,800 records
- final crisis/non-crisis labels
- crisis-category labels
- clinician-adjudicated labels
- direct/ambiguous metadata if available
- stable identifiers
- data dictionary

### 2. Clinician-reviewed NVIDIA AEGIS subset
Request the exact 794 records used for external validation, including:
- stable record IDs
- mapping to original AEGIS2.0 IDs where permitted
- original labels
- final clinician-reviewed labels
- identification of changed records
- adjudication protocol if shareable

### 3. VMHG implementation
Request:
- Stage 1 prompt/configuration
- Stage 2 prompt/configuration
- exact underlying model/provider/version
- preprocessing code
- output parsing/mapping logic
- inference parameters
- evaluation scripts
- dependencies
- random seeds/repeated-run protocol

### 4. Comparison guardrail configuration
If available:
- exact OpenAI moderation version
- exact NVIDIA checkpoint
- category mappings used in the paper

## Draft request email

Subject: Research request for VMHG dataset, clinician-reviewed NVIDIA subset, and code

Dear Dr. Nelson,

I am a master's student working on a seminar project examining the robustness of LLM safety systems for mental-health crisis detection across datasets and conversational contexts.

Your 2026 npj Digital Medicine paper, [“An AI-based mental health guardrail and dataset for identifying psychiatric crises in text-based conversations”](https://doi.org/10.1038/s41746-026-02579-5), is one of the two primary studies forming the basis of our work.

The paper states that the study data and code are available upon researcher request. We would be grateful if you could share, subject to any applicable research-use agreement:

1. the Verily Mental Health Crisis Dataset v1.0, including final clinician-reviewed labels and crisis-category annotations;
2. the exact 794-message NVIDIA AEGIS subset used for external validation, including the clinician-corrected labels and, if possible, identifiers mapping records back to the public AEGIS dataset;
3. the VMHG implementation, prompts, model/configuration details, preprocessing and evaluation code; and
4. the comparison-model configuration/category mappings used for the OpenAI and NVIDIA guardrails.

Our proposed extension is to evaluate cross-dataset generalization under a harmonized crisis taxonomy and, in a second experiment, investigate how conversational context affects crisis detection in multi-turn interactions.

The materials would be used for academic research and reproducibility analysis only, in accordance with any conditions you specify.

Thank you for considering the request.

Kind regards,

Nikhil Shetty
MSc Computer Science / Data Science
University of Bern
