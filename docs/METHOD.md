# Method: Computing with Words core (`src/cww/`)

We evaluate LLM-judge verdicts with Computing with Words (CWW; Zadeh, 1996), using
Mendel's Perceptual Computer (Per-C; Mendel & Wu, 2010): an **encoder** maps rater
scores to fuzzy sets, a **CWW engine** aggregates them, and a **decoder** maps the
result back to a word. This note specifies the maths library only. It has no data
loading and no experiments, and it depends only on numpy.

## 1. Fuzzy sets (`sets.py`)

All sets are sampled on a universe X = {x_0 < … < x_N}, by default [0, 6] with
step 0.01 (601 points, configurable through `universe(lo, hi, step)`). Grid points
are rounded to 10 decimals, so values such as 2.5 are exact.

- **T1 set** A: membership μ_A(x) ∈ [0, 1]; height h_A = max_x μ_A(x).
- **IT2 set** Ã: footprint of uncertainty FOU(Ã) = ⋃_x [μ_Ã(x), μ̄_Ã(x)], with
  0 ≤ LMF ≤ UMF ≤ 1 checked on construction (Mendel & Wu, 2010). A T1 set embeds
  as LMF = UMF.
- **Trapezoid** (a, b, c, d, h): μ = h(x−a)/(b−a) on [a, b), h on [b, c],
  h(d−x)/(d−c) on (c, d], and 0 elsewhere. The edges are vertical when a = b or c = d.
  `it2_trapezoid(umf_params, lmf_params)` builds an IT2 set from two trapezoids.

## 2. Centroid (`centroid.py`)

- T1: c_A = Σ_i x_i μ_A(x_i) / Σ_i μ_A(x_i).
- IT2 (Karnik & Mendel, 2001): C_Ã = [c_l, c_r], the range of the centroids of all
  embedded T1 sets:
  - c_l = min_k (Σ_{i≤k} x_i μ̄_i + Σ_{i>k} x_i μ_i) / (Σ_{i≤k} μ̄_i + Σ_{i>k} μ_i)
  - c_r = max_k (Σ_{i≤k} x_i μ_i + Σ_{i>k} x_i μ̄_i) / (Σ_{i≤k} μ_i + Σ_{i>k} μ̄_i)
- Computed with the **Enhanced Karnik–Mendel (EKM)** algorithm (Wu & Mendel, 2009a).
  EKM starts from the switch point k = round(N/2.4) for c_l and round(N/1.7) for c_r,
  then updates the sums incrementally. Points with μ̄ = 0 are dropped first, which
  keeps every denominator positive.
- Defuzzified value: centre = (c_l + c_r)/2. Uncertainty: width = c_r − c_l.
- The same KM routine (`ekm_left`, `ekm_right`) computes interval weighted averages
  min/max Σ w_i x_i / Σ w_i over w_i ∈ [lo_i, hi_i] (§5).

## 3. Similarity (`similarity.py`)

Jaccard similarity for IT2 sets (Wu & Mendel, 2009b):

  s_J(Ã, B̃) = Σ_i [min(μ̄_Ã, μ̄_B̃) + min(μ_Ã, μ_B̃)] / Σ_i [max(μ̄_Ã, μ̄_B̃) + max(μ_Ã, μ_B̃)].

The value is in [0, 1], s_J(Ã, Ã) = 1, the measure is symmetric, and disjoint sets
give 0. For T1 sets it reduces to Σ min(μ_A, μ_B) / Σ max(μ_A, μ_B). Both sets must
be on the same universe.

## 4. Encoder: Interval Agreement Approach (`iaa.py`)

Type-1 IAA (Wagner et al., 2015): given n intervals, μ(x) = (number of intervals
containing x) / n.

Encoder used in the experiments (implemented exactly as specified):

- Each score s becomes the Likert bin [s-0.5, s+0.5] (half-width configurable).
- Panel of judges, each with runs r_1..r_m:
  judge interval I_j = [min(r)-0.5, max(r)+0.5]
  UMF(x) = (1/J) Σ_j 1[x ∈ I_j]
  LMF(x) = (1/J) Σ_j (fraction of judge j's run bins containing x)
  → guarantees LMF ≤ UMF. Variable number of runs per judge must work (one judge has 2 runs).
- Single-score raters (humans) → type-1 set (lmf = umf).

Implementation notes:
- Intervals are closed, so adjacent bins share their boundary point.
- A rater with a single score gives I_j equal to its bin, so a panel of
  single-score raters automatically yields LMF = UMF, the T1 IAA of the bins
  (`encode_scores`).
- An interval outside the universe raises an error rather than being truncated.
- Missing (NaN) scores are rejected. Callers must filter them first.

RQ1 helper: when n raters give crisp verdicts and k say "crisis", the fuzzy degree
is k/n. This is the IAA membership of "crisis" (`vote_degree`). `majority_vote`
returns crisis / no_crisis / tie for comparison.

## 5. Aggregation (`aggregate.py`)

**OWA** (Yager, 1988): OWA_w(a_1..a_n) = Σ_j w_j b_j, where b_j is the j-th largest
value, w_j ≥ 0 and Σ w_j = 1. orness(w) = (1/(n−1)) Σ_j (n−j) w_j, so max = 1,
mean = 0.5 and min = 0. `owa_weights(n, o)` returns weights with orness exactly o by
mixing the reference operators linearly:

- o ≥ 0.5: w = (2o−1)·e_max + (2−2o)·u
- o < 0.5: w = (1−2o)·e_min + 2o·u

Here u is uniform. o < 0.5 is the safety-pessimistic setting, which weights the
lowest scores more. This weight family is our design choice; Yager (1988) defines
the operator and orness.

**Fuzzy weighted average** (Liu & Mendel, 2008): Y = Σ W_i X_i / Σ W_i. Each input is
a crisp number, an interval or a T1 set. For each α ∈ linspace(0, h_min, n_alpha)
(n_alpha = 51 by default; h_min is the smallest input height), the α-cuts are
X_i(α) = [a_i, b_i] and W_i(α) = [c_i, d_i], and the output cut is

  y_L(α) = min_{w_i∈[c_i,d_i]} Σ w_i a_i / Σ w_i,  y_R(α) = max_{w_i∈[c_i,d_i]} Σ w_i b_i / Σ w_i,

both computed with EKM.

How α-cuts are taken on the grid:
- For a sampled set the α-cut is {x : μ(x) ≥ α}, and α = 0 means the discrete
  support {μ > 0}. This is exact for the step-function sets produced by the IAA
  encoder. For sloped edges the error is at most one grid step.
- The cut is taken as the convex hull [min, max], so a non-convex input (for
  example an LMF with a gap) is convexified, as the α-cut FWA requires.
- The result is returned as α-cuts and discretised by μ(x) = max{α_k : x ∈ [y_L, y_R](α_k)}.
  A cut that falls between two grid points is assigned to the nearest point.

**Linguistic weighted average** (Wu & Mendel, 2007): for IT2 X̃_i and W̃_i (the
weights may live on their own universe, such as [0, 1]):
- UMF(Ỹ) = FWA(UMF(X̃_i), UMF(W̃_i)).
- LMF(Ỹ) = FWA(LMF(X̃_i), LMF(W̃_i)) over α ∈ [0, h_min], where h_min is the
  smallest LMF height, so LMF(Ỹ) keeps that height.

After discretisation, LMF is clipped to min(LMF, UMF), which removes at most one
grid step of overlap. If any LMF has height 0, then LMF(Ỹ) = 0.

## 6. 2-tuples (`two_tuple.py`)

For an ordered term set S = {s_0..s_g} (Herrera & Martínez, 2000):

- Δ(β) = (s_i, α), where i = round(β) and α = β − i ∈ [−0.5, 0.5). Halves round up,
  so 1.5 ↦ (s_2, −0.5).
- Δ⁻¹(s_i, α) = i + α.
- Arithmetic mean: Δ((1/n) Σ Δ⁻¹(r_j, α_j)).
- Weighted mean: Δ(Σ w_j Δ⁻¹(r_j, α_j) / Σ w_j).

## 7. Decoder: codebook (`codebook.py`)

`Codebook(words, sets)` decodes a set Ỹ to argmax_k s_J(Ỹ, W̃_k). It returns the
word, its Jaccard value and every similarity; ties go to the lower word.
`to_two_tuple(v)` maps a value on the 1–5 scale to Δ(v − 1). For example,
2.8 ↦ ("Partially Appropriate", −0.2).

`appropriateness_codebook(fou_width=w)` contains the five protocol levels
(Harmful/Unacceptable, Inappropriate, Partially Appropriate, Mostly Appropriate,
Fully Appropriate) as IT2 trapezoids centred on k = 1..5, all sharing the core
[k−0.25, k+0.25]:

  UMF = (k−0.75−w/2, k−0.25, k+0.25, k+0.75+w/2),  LMF = (k−0.75+w/2, k−0.25, k+0.25, k+0.75−w/2).

- w is the horizontal distance between the UMF and LMF feet. At w = 0 the words are
  T1 trapezoids that cross their neighbours at k ± 0.5 with membership 0.5.
- **w is a proxy parameter.** No clinician word survey exists, so the experiments
  must report sensitivity to w (default 0.25; valid range 0 ≤ w ≤ 1, limited by
  the universe to w ≤ 0.5 on [0, 6]).
- Each word's centroid is exactly k.

## 8. Escalation rules (`rules.py`)

This module is a design proposal for the escalation demo. It is **not clinically
validated**.

**Crisis-level words.** Four T1 trapezoids on [0, 1], evaluated at the crisis degree μ:

| Word | Trapezoid (a, b, c, d) |
|---|---|
| None | (0, 0, 0.1, 0.3) |
| Possible | (0.1, 0.3, 0.4, 0.6) |
| Likely | (0.4, 0.6, 0.7, 0.9) |
| Clear | (0.7, 0.9, 1, 1) |

**Actions**, in increasing severity:

- A0: Continue.
- A1: Signpost a helpline (143; 147 for young people).
- A2: Human review / handover to a counsellor.
- A3: Urgent human handover and show emergency numbers (144 / 112).

**Rule table** (crisis word × reply word → action):

| Crisis \ Reply | Harmful | Inappropriate | Partially | Mostly | Fully |
|---|---|---|---|---|---|
| None | A2 | A2 | A0 | A0 | A0 |
| Possible | A2 | A2 | A1 | A1 | A0 |
| Likely | A3 | A3 | A2 | A1 | A1 |
| Clear | A3 | A3 | A3 | A2 | A1 |

**Inference** is max–min (Mamdani-style) firing:

- rule strength = min(μ_crisis(c), s_reply(r))
- strength(a) = max over the rules (c, r) that conclude a

The output is the action with the highest strength. Ties go to the **more severe**
action, so if every strength is 0 the output is A3. A crisp input (membership 1 for
a single word) reproduces the table lookup.

## References

Titles, venues, years and DOIs were checked against OpenAlex on 2026-11-21. Page
ranges were not checked by machine.

- Herrera, F., & Martínez, L. (2000). A 2-tuple fuzzy linguistic representation model for computing with words. *IEEE Transactions on Fuzzy Systems*, 8(6), 746–752. https://doi.org/10.1109/91.890332
- Karnik, N. N., & Mendel, J. M. (2001). Centroid of a type-2 fuzzy set. *Information Sciences*, 132(1–4), 195–220. https://doi.org/10.1016/S0020-0255(01)00069-X
- Liu, F., & Mendel, J. M. (2008). Aggregation using the fuzzy weighted average as computed by the Karnik–Mendel algorithms. *IEEE Transactions on Fuzzy Systems*, 16(1), 1–12. https://doi.org/10.1109/TFUZZ.2007.896229
- Mendel, J. M., & Wu, D. (2010). *Perceptual Computing: Aiding People in Making Subjective Judgments*. Wiley-IEEE Press. https://doi.org/10.1002/9780470599655
- Wagner, C., Miller, S., Garibaldi, J. M., Anderson, D. T., & Havens, T. C. (2015). From interval-valued data to general type-2 fuzzy sets. *IEEE Transactions on Fuzzy Systems*, 23(2), 248–269. https://doi.org/10.1109/TFUZZ.2014.2310734
- Wu, D., & Mendel, J. M. (2007). Aggregation using the linguistic weighted average and interval type-2 fuzzy sets. *IEEE Transactions on Fuzzy Systems*, 15(6), 1145–1161. https://doi.org/10.1109/TFUZZ.2007.896325
- Wu, D., & Mendel, J. M. (2009a). Enhanced Karnik–Mendel algorithms. *IEEE Transactions on Fuzzy Systems*, 17(4), 923–934. https://doi.org/10.1109/TFUZZ.2008.924329
- Wu, D., & Mendel, J. M. (2009b). A comparative study of ranking methods, similarity measures and uncertainty measures for interval type-2 fuzzy sets. *Information Sciences*, 179(8), 1169–1192. https://doi.org/10.1016/j.ins.2008.12.010
- Yager, R. R. (1988). On ordered weighted averaging aggregation operators in multicriteria decisionmaking. *IEEE Transactions on Systems, Man, and Cybernetics*, 18(1), 183–190. https://doi.org/10.1109/21.87068
- Zadeh, L. A. (1996). Fuzzy logic = computing with words. *IEEE Transactions on Fuzzy Systems*, 4(2), 103–111. https://doi.org/10.1109/91.493904
