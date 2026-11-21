## ANALYSIS PLAN: RQ1, CounselBench descriptive, escalation demo (frozen 2026-11-21)

General: item-bootstrap 95% CIs (B = 2000, seed 20261121) for every estimate; results contain
ids and numbers only; all hypothesis tests are exploratory (no multiplicity correction).
Verdict rule as in RQ2: supported / not supported / inconclusive by the CI of the stated quantity.

### Part A — RQ1: is "crisis" better kept as a degree than a majority label?
Data: BHH crisis labels, 206 items (D4, D13). Missing LLM labels stay missing.
- mu_H = k_humans_crisis / 4; mu_L = k_llm_crisis / n_llm_observed.
- Crisp labels: human_majority (crisis / no_crisis / tie, as in the loader); LLM majority with the
  same rule on mu_L (> 0.5 crisis, < 0.5 no_crisis, = 0.5 tie).
- human_split = 0 < mu_H < 1.
Descriptive:
- D-A1: distribution of mu_H (0, .25, .5, .75, 1) and of mu_L; share of human_split items;
  share by human-majority category.
- D-A2: mean mu_L for each mu_H level (calibration table and figure).
- D-A3: category-level fuzzy agreement per item = Σ_c min(p_H(c), p_L(c)) / Σ_c max(p_H(c), p_L(c)),
  with p(c) = share of observed raters choosing category c (7 categories); compare with crisp exact
  category match (human majority category vs LLM majority category; ties excluded and counted).
Hypotheses:
- H1a: Spearman(mu_H, mu_L) > 0 among human_split items (CI of rho entirely above 0).
- H1b: The crisp disagreement rate (LLM majority ≠ human majority, both non-tie) is higher on
  human_split items than on unanimous items (CI of the risk difference entirely above 0).
  Also report the counts of ties that crisp metrics must drop.

### Part B — CounselBench expert uncertainty (descriptive only; D6)
Data: CounselBench-Eval, 400 replies × 5 experts.
- B1: per risk dimension, the distribution of k_yes (medical advice: k of 5 "yes"); share of split
  panels (1–4 of 5) vs unanimous; "I am not sure" prevalence per dimension.
- B2: Likert dimensions (overall, empathy, specificity, toxicity): per reply, the rating range and
  the IAA panel set (encode_scores, half_width 0.5); share of replies whose experts span ≥ 2 levels.
- B3: toxicity panel decoded with a 5-word codebook built like appropriateness_codebook(0.25),
  using CounselBench's own scale words (1 Not toxic at all, 2 Somewhat not toxic, 3 Neutral,
  4 Somewhat toxic, 5 Extremely toxic); share of contested panels (same contested rule as RQ2);
  share of replies where the crisp mean's 2-tuple word differs from the decoded word.
- B4: replies with exactly one expert copying a risky sentence (k_copied = 1) per risk type:
  the evidence a "flagged by ≥ 2 experts" answer key discards. Report counts and shares.
- B5: all of the above split by responder (gpt4, llama3, gemini, human).

### Part C — Escalation demo (Perceptual Computer decoder to a public-service action)
Join on item_id: RQ1 crisis degree (206 items) × RQ2 reply verdict (190 complete-case items) → 190 items.
Inputs (system side, deployable, no human data):
- Crisis level word from mu_L with T1 trapezoids on [0, 1]:
  None (0, 0, 0.1, 0.3); Possible (0.1, 0.3, 0.4, 0.6); Likely (0.4, 0.6, 0.7, 0.9); Clear (0.7, 0.9, 1, 1).
  Membership of each word at mu_L.
- Reply word memberships = S3 Jaccard similarities to the 5 appropriateness words.
Actions: A0 Continue; A1 Signpost helpline (143; 147 for young people); A2 Human review / handover
to a counsellor; A3 Urgent: human handover and show emergency numbers (144 / 112).
Rule table (crisis level × reply word → action):
| Crisis \ Reply | Harmful | Inappropriate | Partially | Mostly | Fully |
|---|---|---|---|---|---|
| None     | A2 | A2 | A0 | A0 | A0 |
| Possible | A2 | A2 | A1 | A1 | A0 |
| Likely   | A3 | A3 | A2 | A1 | A1 |
| Clear    | A3 | A3 | A3 | A2 | A1 |
Inference: rule strength = min(crisis membership, reply similarity); action strength = max over
its rules (max–min); output the strongest action; ties go to the MORE severe action.
Pipelines:
- P-fuzzy: inputs as above.
- P-crisp: crisis = LLM majority (crisis → Clear, no_crisis → None, tie → Possible) with
  membership 1; reply = S2 word with membership 1; same rule table.
- Oracle (reference): crisis word memberships from mu_H; reply similarities from decoding
  encode_scores([H1, H2]); same rule table and inference.
Metrics: action agreement with the oracle (exact; quadratic-weighted kappa on A0–A3);
under-escalation rate = share of items where the pipeline action < oracle action, overall and among
oracle A2/A3 items; over-escalation rate; action distribution per pipeline.
Hypothesis:
- H3: P-fuzzy under-escalation rate among oracle A2/A3 items ≤ P-crisp (paired difference CI).
Caveat to state: the oracle uses the same rule table, so this compares input representations
(fuzzy vs crisp), not the rule table itself. The rule table is a design proposal, not clinically validated.
Worked cases: select 8 item_ids by rule (no cherry-picking): the 4 items with the largest
oracle-minus-P-crisp action gap, then the 4 with the largest P-fuzzy-minus-P-crisp gap (ties by
lowest item_id). Report ids, degrees, words, similarities and actions only. No text.
## END PLAN
