"""Probability word models on 0..100 at step 0.5.

Quantiles use numpy's default method, explicitly pinned as method='linear'.
The four quantiles are clipped after estimation, not the source observations.
Empty/nonfinite samples are rejected; no missing-value cleaning occurs here.
"""
import numpy as np
from .sets import IT2Set, trapezoid, universe, same_universe

X = universe(0, 100, .5)


def quantile_trapezoid(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty finite one-dimensional sample')
    q = np.clip(np.quantile(values, [.05, .25, .75, .95], method='linear'), 0, 100)
    return trapezoid(*q, x=X)


def envelope(models):
    models = list(models)
    if not models:
        raise ValueError('Cannot envelope zero models')
    for model in models:
        same_universe(models[0], model)
    memberships = np.stack([model.mu for model in models])
    return IT2Set(models[0].x, memberships.min(axis=0), memberships.max(axis=0))


def group_models(samples, min_n=30):
    """Return retained models and counts for EVERY candidate, including exclusions.

    Eligibility is per group/term sample size, inclusive at min_n=30.
    """
    if min_n < 1:
        raise ValueError('min_n must be positive')
    counts = {name: len(values) for name, values in samples.items()}
    models = {name: quantile_trapezoid(values) for name, values in samples.items()
              if counts[name] >= min_n}
    return models, counts
