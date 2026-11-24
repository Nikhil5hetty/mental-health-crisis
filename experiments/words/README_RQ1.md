# Citizen word models (WP3)

Run from any directory with the project venv:

```bash
.venv/bin/python -I experiments/words/run_rq1.py
.venv/bin/python -I -m pytest -q
```

Uses only local CAPphrase files and the existing CWW set, EKM, and similarity
implementations. No network calls or downloaded scripts are used. Outputs are
aggregate numbers plus term/group labels; no respondent identifiers, individual
responses, or demographic microdata are exported. The explicitly requested
aggregate results and figures live in `results/words/rq1/` and are committed.

The frozen RQ1 specification is `docs/ANALYSIS_PLAN_WORDS.md`. NumPy linear
quantiles estimate q05/q25/q75/q95, clipped to 0–100; memberships use the existing
trapezoid implementation on 201 points at spacing 0.5. FOU area is a rectangle
sum, not trapezoidal integration. Group counts are measured for every term;
missing/under-18 demographics stay in pooled data. Groups below 30 observations
are reported and omitted from that term's envelope. No groups are excluded in
the inspected primary or sensitivity data.

`human_group` is not inferred: group keys are exactly `english_group|age_group`,
using the CAPphrase loader's published-metadata mapping. The six eligible keys
are formed from native/non_native × 18-34/35-54/55+.

Bootstrap draws whole respondents with replacement, preserving each respondent's
19-term vector: 200 draws of the original sample size, `default_rng(20261124)`.
The same draw indices are used across terms; envelopes are taken over all 200
resulting trapezoids. This is a sensitivity envelope, not a 95% confidence band.
The attention sensitivity invokes the loader's explicit anchor rule, excluding
equality as well as reversal. It refits pooled T1, all group T1, and citizen IT2.

Pair consistency divides strict concordant choices by **all** published pair
rows; ties form a separate fraction. The repeated tenth pair is retained as
published. This statistic is not an independence-adjusted inferential test.

## Output contracts

- `codebooks/citizen_pooled_t1.csv`: term, x, mu.
- `codebooks/citizen_it2.csv`: term, x, umf, lmf.
- `codebooks/group_t1.csv`: term, group, x, mu.
- `metrics.csv`: metric, term, group, value. `pooled` rows report n, IQR,
  median, pooled centroid, demographic exclusions, citizen FOU area and EKM
  endpoints/centre; named-group rows report n and inclusion flags.
- `group_jaccard.csv`: term, group_a, group_b, jaccard (15 pairs × 19 terms).
- `consistency.csv`: n_pairs, consistent, ties, inconsistent and their shares.
- `sensitivity.csv`: scenario, metric, term, group, value. Includes bootstrap
  and attention-filtered IT2 metrics, similarities to primary models, filtered
  pooled metrics, exclusions, and per-term/group counts/inclusion flags.
- Extra inspectable models: `codebooks/bootstrap_it2.csv` and
  `codebooks/filtered/{citizen_it2,citizen_pooled_t1,group_t1}.csv`.
- `figures/fou_panels.png`: 19 panels ordered by pooled median (alphabetical
  ties), with pooled T1 dashed and the citizen FOU shaded.
- `figures/fou_ranking.png`: all terms ordered by citizen FOU area.
- `SUMMARY.md`: script-generated findings, denominator definitions, and deviations.

CSV values use 17 significant digits. Output ordering is deterministic and PNG
metadata contains no timestamps. Two reruns in the same pinned environment
must have identical file hashes. `--output` can select an isolated comparison
directory; the default is the committed result location. No output is read back
as an input to fitting.
