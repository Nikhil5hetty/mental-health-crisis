## ANALYSIS PLAN: Same word, different meaning (frozen 2026-11-24)
Question: Do citizens and LLMs mean the same thing by probability words, and does a mismatch change
escalation decisions of a crisis-line Perceptual Computer?

Data: CAPphrase (Kucharski 2026, CC-BY 4.0; 5,174 respondents x 19 terms, 0-100); LLM answers generated
by us (WP2); Between Help and Harm (BHH) crisis degree and S3 reply words as in results/rq1 and results/rq2.

Universe: 0-100, step 0.5. Quantile trapezoid of a sample = (q05, q25, q75, q95), clipped to [0,100].

RQ1 (citizen word models)
- Pooled T1 word model per term = quantile trapezoid of all respondents.
- Citizen groups (primary): english_background {native; non-native = fluent + not fluent} x age
  {18-34; 35-54; 55+} = 6 groups ("Under 18" and missing metadata excluded from groups, kept in pooled).
  Groups with n < 30 for a term are excluded and reported.
- Citizen IT2 word model = UMF pointwise max, LMF pointwise min of the 6 group trapezoids.
- Sensitivity: (a) bootstrap IT2 (200 resamples of pooled data, seed 20261124, UMF max / LMF min);
  (b) excluding respondents with Almost No Chance >= Almost Certain (inattention).
- Report per term: IQR, FOU area (sum of UMF - LMF times step), centroid [cl, cr]; ranking of widest terms;
  pairwise group Jaccard; share of pairwise choices consistent with the same respondent's absolute values.

RQ2 (LLM word models)
- Models: 4 local open-weight models via Ollama (pinned tags + digests recorded), temperature 1.0,
  50 samples per term per model, seed base 20261124, prompt P1 (primary) below; P2 persona prompt secondary.
  P1: "On a scale from 0 to 100, what probability (in percent) does the phrase \"<term>\" convey? Answer with a single number only."
  P2: P1 prefixed with "You are a member of the public living in Switzerland."
- Parsing: first number in [0,100]; otherwise re-ask up to 2 times; then missing. Report invalid rate.
- Per-model T1 = quantile trapezoid; LLM IT2 = UMF max / LMF min over the 4 models.
- Metrics per term: Jaccard(citizen IT2, LLM IT2); Jaccard(pooled citizen T1, each model T1);
  centroid shift (LLM centre - citizen centre); width ratio (LLM IQR / citizen IQR).
  95% bootstrap CIs (B = 1000 over respondents and samples, seed 20261124).
- H2a: mean Jaccard of hedge terms {Realistic Possibility, Could Happen, May Happen, Might Happen,
  Probable, Likely} < mean Jaccard of extreme terms {Almost No Chance, Almost Certain, Will Happen}.
- H2b: LLM IQR < citizen IQR for a majority of terms (per model and pooled).

RQ3 (consequence for escalation)
- Crisis level words: None = "Remote Chance", Possible = "Realistic Possibility", Likely = "Likely",
  Clear = "Almost Certain". For each BHH message, mu_L (x100) is verbalised as the level word with the
  highest membership (mean of UMF and LMF; ties to the more severe word) under (i) the citizen codebook,
  (ii) the LLM codebook, (iii) each citizen group's T1 codebook.
- Action = rule table in src/cww/rules.py with crisis level = that word (membership 1) and reply word = S3 word.
- Metrics (190 complete items): word disagreement rate, action flip rate (LLM vs citizen), under-escalation
  rate (LLM action < citizen action), per citizen group; bootstrap CIs.
- H3: action flip rate LLM vs citizen > 0 (CI excludes 0).

All tests exploratory; verdict rule = CI as in earlier plans; deviations recorded in each SUMMARY.md.
## END PLAN
