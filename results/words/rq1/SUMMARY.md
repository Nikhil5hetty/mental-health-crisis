# RQ1: Citizen probability-word models

Pooled: 5,174 respondents; 98,306 judgements; 19 terms.
Universe 0–100, step 0.5. NumPy linear quantiles (q05, q25, q75, q95), clipped after estimation.
Primary IT2: pointwise max/min of eligible demographic T1 models. Centroids use EKM.

## Group sizes

| Group | Minimum n per term | Maximum n per term |
|---|---:|---:|
| native / 18-34 | 757 | 757 |
| native / 35-54 | 2158 | 2158 |
| native / 55+ | 1038 | 1038 |
| non_native / 18-34 | 247 | 247 |
| non_native / 35-54 | 406 | 406 |
| non_native / 55+ | 101 | 101 |

467 respondents excluded from demographic groups only (missing metadata or under 18); retained in pooled models.
Group/term cells excluded for n < 30: 0.

## FOU ranking

| Rank | Term | FOU area | EKM interval | Centre |
|---:|---|---:|---|---:|
| 1 | Unlikely | 17.000000 | [14.158703, 22.510152] | 18.334428 |
| 2 | Probable | 14.597222 | [64.583333, 72.270731] | 68.427032 |
| 3 | Remote Chance | 14.000000 | [8.555556, 16.468113] | 12.511834 |
| 4 | Might Happen | 13.750000 | [37.279835, 44.020499] | 40.650167 |
| 5 | May Happen | 13.360417 | [37.500000, 44.624994] | 41.062497 |
| 6 | Chances are Slight | 13.000000 | [11.979798, 18.987573] | 15.483686 |
| 7 | Could Happen | 11.750000 | [36.078431, 42.070621] | 39.074526 |
| 8 | Likely | 10.000000 | [69.666667, 75.000000] | 72.333333 |
| 9 | Little Chance | 9.163073 | [11.404007, 15.726594] | 13.565300 |
| 10 | Very Good Chance | 8.400000 | [76.111111, 80.337321] | 78.224216 |
| 11 | Realistic Possibility | 8.183998 | [51.945730, 56.206212] | 54.075971 |
| 12 | Will Happen | 7.000000 | [92.804910, 96.822581] | 94.813745 |
| 13 | Almost Certain | 6.250000 | [90.708333, 94.128571] | 92.418452 |
| 14 | Improbable | 5.127551 | [12.949639, 15.605442] | 14.277541 |
| 15 | Better than Even | 4.500000 | [58.083333, 60.851180] | 59.467257 |
| 16 | Highly Likely | 4.000000 | [83.571429, 85.848485] | 84.709957 |
| 17 | Highly Unlikely | 4.000000 | [8.555556, 10.850776] | 9.703166 |
| 18 | About Even | 3.000000 | [48.080645, 50.500000] | 49.290323 |
| 19 | Almost No Chance | 0.250000 | [4.142857, 4.277778] | 4.210317 |

Widest: Unlikely; narrowest: Almost No Chance.

## Group differences

285 pairwise T1 Jaccards (15 per term when all six groups qualify).
| Group pair | Mean Jaccard across terms |
|---|---:|
| native / 18-34 vs native / 35-54 | 0.903296 |
| native / 18-34 vs native / 55+ | 0.834981 |
| native / 18-34 vs non_native / 18-34 | 0.763932 |
| native / 18-34 vs non_native / 35-54 | 0.773727 |
| native / 18-34 vs non_native / 55+ | 0.749259 |
| native / 35-54 vs native / 55+ | 0.888124 |
| native / 35-54 vs non_native / 18-34 | 0.752125 |
| native / 35-54 vs non_native / 35-54 | 0.765267 |
| native / 35-54 vs non_native / 55+ | 0.753114 |
| native / 55+ vs non_native / 18-34 | 0.752110 |
| native / 55+ vs non_native / 35-54 | 0.771636 |
| native / 55+ vs non_native / 55+ | 0.764096 |
| non_native / 18-34 vs non_native / 35-54 | 0.882942 |
| non_native / 18-34 vs non_native / 55+ | 0.833662 |
| non_native / 35-54 vs non_native / 55+ | 0.854555 |

## Pairwise consistency

Of 51,740 pair rows: 45,658 strictly consistent (88.245072%), 2,361 tied (4.563201%), and 3,721 inconsistent (7.191728%).
Denominator includes all pair rows, including the repeated pair in each respondent session; ties are not counted as consistent.

## Sensitivity

Bootstrap: 200 whole-respondent resamples with replacement, seed 20261124; same sampled respondents across all terms.
Inattention rule excludes 17 respondents (Almost No Chance >= Almost Certain), leaving 5157.
Filtered group/term cells excluded for n < 30: 0.

| Scenario | Mean FOU area | Mean Jaccard to primary IT2 |
|---|---:|---:|
| bootstrap | 3.023035 | 0.846298 |
| inattention_excluded | 9.014468 | 0.994436 |

## Deviations

None from the frozen RQ1 analysis plan. Missing/under-18 respondents remain in pooled and bootstrap samples; the inattention exclusion applies only to its sensitivity scenario.
Implementation choices: strict-choice share uses all rows as denominator; whole-respondent bootstrap preserves cross-term dependence; group eligibility is reassessed after sensitivity exclusion.
CSV files use 17 significant digits. Extra filtered and bootstrap codebooks are saved to support inspection; no individual-level records or identifiers are exported.
