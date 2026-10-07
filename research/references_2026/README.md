# 2026 Research Reference Pack

This project deliberately restricts its **core research literature to 2026 publications**.

## Core papers

### 1. CounselBench — primary benchmark
**Yahan Li, Jifan Yao, John Bosco S. Bunyi, Adam C. Frank, Angel Hsing-Chi Hwang, Ruishan Liu.**  
*CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering.*  
ICLR 2026.

- Paper: https://proceedings.iclr.cc/paper_files/paper/2026/file/99946cb64d51ead9d3969db0af65ca2e-Paper-Conference.pdf
- Conference page: https://proceedings.iclr.cc/paper_files/paper/2026/hash/99946cb64d51ead9d3969db0af65ca2e-Abstract-Conference.html
- Code/data: https://github.com/llm-eval-mental-health/CounselBench

**Why we need it:** Main expert-labelled benchmark. It provides 2,000 expert evaluations across six clinically grounded dimensions and demonstrates that LLM judges can overrate responses and miss safety concerns.

---

### 2. When Can We Trust LLMs in Mental Health? — judge-reliability extension
**Abeer Badawi et al.**  
*When Can We Trust LLMs in Mental Health? Large-Scale Benchmarks for Reliable LLM Evaluation.*  
EACL 2026.

- Paper: https://aclanthology.org/2026.eacl-long.180.pdf
- ACL page: https://aclanthology.org/2026.eacl-long.180/
- Code/data: https://github.com/abeerbadawi/MentalBench-Align
- Dataset: https://huggingface.co/datasets/abadawi/MentalBench-Align

**Why we need it:** Directly studies reliability of LLM-as-a-judge in mental health. Introduces MentalBench-100k, MentalAlign-70k, and ICC-based agreement/bias analysis.

---

### 3. Between Help and Harm — crisis-specific response evaluator
**Adrian Arnaiz-Rodriguez et al.**  
*Between Help and Harm: An Evaluation Study of Mental Health Crisis Handling by Large Language Models.*  
JMIR Mental Health, 2026.

- Article: https://mental.jmir.org/2026/1/e88435/
- DOI: https://doi.org/10.2196/88435
- Code: https://github.com/ellisalicante/LLMs-Mental-Health-Crisis
- Dataset: https://huggingface.co/datasets/arnaiztech/llms-mental-health-crisis-benchmark

**Why we need it:** Gives a clinically informed crisis taxonomy and a 1–5 response-appropriateness evaluation protocol validated against experts. This is our strongest candidate for a crisis-specific evaluator comparison.

---

### 4. MentalHealthBench — newest expert-rubric benchmark
**Ali Malik et al.**  
*MentalHealthBench: An Expert-Informed Benchmark of AI Capabilities in Realistic Mental Health Conversations.*  
OpenAI, 2026.

- Paper: https://cdn.openai.com/ctf-cdn/MentalHealthBench_A_Comprehensive_Benchmark_of_AI_Capabilities_in_Realistic_Mental_Health_Conversations.pdf
- Publication page: https://openai.com/index/introducing-mentalhealthbench/

**Why we need it:** 1,215 realistic conversations with expert-written weighted rubrics from 80+ licensed mental-health experts. Especially relevant to testing newer automated graders and rubric-based evaluation.

---

### 5. Verily Mental Health Guardrail — mental-health-specific safety system
**Benjamin W. Nelson et al.**  
*An AI-based mental health guardrail and dataset for identifying psychiatric crises in text-based conversations.*  
npj Digital Medicine, 2026.

- Paper: https://www.nature.com/articles/s41746-026-02579-5.pdf
- Article: https://www.nature.com/articles/s41746-026-02579-5
- DOI: https://doi.org/10.1038/s41746-026-02579-5

**Why we need it:** Demonstrates why general-purpose safety taxonomies may not transfer cleanly to psychiatric crisis detection. Useful as background and as a comparison to mental-health-specific safety systems.

---

### 6. SIM-VAIL — multi-turn/contextual safety reference
**Veith Weilnhammer et al.**  
*A clinically validated framework for auditing AI chatbot behavior in mental health interactions.*  
Nature Medicine, 2026.

- Article: https://www.nature.com/articles/s41591-026-04577-2
- PDF: https://www.nature.com/articles/s41591-026-04577-2.pdf
- DOI: https://doi.org/10.1038/s41591-026-04577-2

**Why we need it:** Provides the strongest 2026 reference for multi-turn mental-health interaction auditing and trajectory-level safety analysis. It is secondary to the CounselBench-focused judge study.

---

## Priority for our project

### Essential
1. CounselBench
2. When Can We Trust LLMs in Mental Health? / MentalAlign-70k
3. Between Help and Harm
4. MentalHealthBench

### Supporting
5. Verily Mental Health Guardrail
6. SIM-VAIL

## Current research direction supported by these papers

> **How reliably do modern LLM-as-a-judge systems reproduce expert mental-health evaluations, and which clinically important response failures do they systematically overlook?**

The intended empirical path is:

1. Reproduce a subset of CounselBench's original judge evaluation.
2. Add one or more newer 2026 judge models.
3. Compare against CounselBench expert scores.
4. Use MentalAlign-70k methodology (ICC, agreement, bias) to strengthen judge-reliability analysis.
5. Test the BHH crisis-specific evaluation protocol on the subset of CounselBench cases where it is semantically applicable.
6. Optionally validate conclusions on MentalHealthBench.
