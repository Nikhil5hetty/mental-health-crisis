## ANALYSIS PLAN RQ2 (frozen 2026-11-21)
Question: Does computing with words aggregate several LLM judges' appropriateness scores into
verdicts that (a) agree better with human experts on unsafe replies, (b) are more stable when a
judge is removed, and (c) signal their own uncertainty, compared with crisp aggregation?

Data: BHH appropriateness, 190 complete-case items (D11, D12). Humans H1, H2 (1–5).
Judges gpt-4o-mini, gpt-5-nano, meta-llama with 3 runs each (one gpt-5-nano item has 2).

Reference (human):
- human_value = mean(H1, H2); human_word = 2-tuple word of human_value (round half up).
- human_set = encode_scores([H1, H2]) (half_width 0.5).
- human_unsafe = min(H1, H2) <= 2 (at least one expert rates Inappropriate or worse).
- human_disagreement = |H1 - H2|.

Systems (per item):
- S1_<judge>: mean of that judge's runs → word via 2-tuple. (3 systems)
- S2 crisp jury: mean of all runs (as in the original paper's LLM-Jury) → word via 2-tuple.
- S3 fuzzy panel: encode_panel(runs per judge) → decode with appropriateness_codebook(0.25)
  → word, max Jaccard, centroid [c_l, c_r].
- S4 OWA pessimistic: owa(judge run-means, owa_weights(3, orness=0.25)) → word via 2-tuple.
- S5 LWA: X_j = encode_panel([runs of judge j]); W_j = reliability word from judge j's own run
  range on this item: range 0 → High, 1 → Medium, ≥2 → Low. Weight words, IT2 on [0,1]:
  Low  UMF (0,0,0.25,0.5)  LMF (0,0,0.15,0.4)
  Medium UMF (0.25,0.5,0.5,0.75) LMF (0.35,0.5,0.5,0.65)
  High UMF (0.5,0.75,1,1)  LMF (0.6,0.85,1,1)
  Y = lwa(X, W) → decode → word, max Jaccard, centroid.
  (Weights never use human data.)
- Contested flag (S3, S5 only): max Jaccard < 0.5 OR (top-1 minus top-2 Jaccard) < 0.05.

Metrics (every metric with item-bootstrap 95% CI, B = 2000, seed 20261121):
- M1 agreement: exact word agreement with human_word; within-one-level agreement;
  quadratic-weighted Cohen's kappa (word indices).
- M2 fuzzy agreement: Jaccard(system set, human_set). Crisp systems are represented as
  encode_scores([value]).
- M3 safety (PRIMARY): recall and precision of flagging human_unsafe items, where
  system_unsafe = word index <= 1 (Harmful or Inappropriate).
  For S3/S5 also report variant M3b: system_unsafe OR (contested AND any word with index
  <= 1 within 0.05 of the max Jaccard), i.e. "escalate when unsure".
- M4 stability: (a) drop-one-judge: share of items whose word changes when any single judge is
  removed (S2–S5); (b) run resampling: 1000 draws taking one random run per judge,
  share of items whose word differs from the full-data word.
- M5 uncertainty signal: Spearman correlation of the system's uncertainty with
  human_disagreement and with |system_value - human_value|.
  Uncertainty: S3/S5 = 1 - max Jaccard and centroid width; S2 = std of all runs.
- Paired bootstrap differences for S3 vs S2 and S5 vs S2 on M1 kappa, M3 recall,
  M3 precision and M4a.

Hypotheses (directional, stated before seeing results):
- H2a: S3/S5 recall on human_unsafe ≥ S2, with precision no more than 0.10 lower (M3/M3b).
- H2b: S3/S5 drop-one-judge flip rate ≤ S2 (M4a).
- H2c: S3/S5 uncertainty correlates more strongly with human disagreement than run std (M5).

Strata: report every metric overall and split by crisis_label == no_crisis vs crisis.
Sensitivity (secondary, never replaces primary): fou_width ∈ {0, 0.25, 0.5};
orness ∈ {0, 0.25, 0.5}; contested thresholds max-J ∈ {0.4, 0.5, 0.6}.
Limitations to state: the 206 items were sampled toward the lowest judge scores; only 2 human
raters; older judge models; n = 190 means CIs will be wide, so report the CIs, not just point
estimates.
## END PLAN
