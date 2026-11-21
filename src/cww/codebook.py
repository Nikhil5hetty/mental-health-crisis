"""Decoder: codebook of words, Jaccard decoding and 2-tuple output.

References
----------
Mendel, J. M., & Wu, D. (2010). Perceptual Computing. Wiley-IEEE Press (decoder).
Wu, D., & Mendel, J. M. (2009). Information Sciences 179(8) (Jaccard).
Herrera, F., & Martinez, L. (2000). IEEE TFS 8(6) (2-tuples).
"""
from __future__ import annotations

from collections.abc import Sequence

from . import two_tuple
from .sets import IT2Set, T1Set, _check_x, as_it2, it2_trapezoid, same_universe, DEFAULT_X
from .similarity import jaccard

APPROPRIATENESS_WORDS = [
    'Harmful/Unacceptable',
    'Inappropriate',
    'Partially Appropriate',
    'Mostly Appropriate',
    'Fully Appropriate',
]


class Codebook:
    """Ordered words s_0..s_g with IT2 sets, anchored at first_value, first_value+1, ..."""

    def __init__(self, words: Sequence[str], sets: Sequence[IT2Set], first_value: float = 1.0):
        if len(words) < 2 or len(words) != len(sets):
            raise ValueError('Need >= 2 words, one set per word')
        if len(set(words)) != len(words):
            raise ValueError('Words must be unique')
        self.words = list(words)
        self.sets = [as_it2(s) for s in sets]
        for s in self.sets[1:]:
            same_universe(self.sets[0], s)
        self.first_value = float(first_value)

    @property
    def g(self) -> int:
        return len(self.words) - 1

    def decode(self, y: T1Set | IT2Set) -> tuple[str, float, dict[str, float]]:
        """Return (word, s_J, {word: s_J}) with word = argmax_k s_J(Y, W_k).

        Ties go to the first (lowest) word in codebook order.
        """
        sims = {w: jaccard(y, s) for w, s in zip(self.words, self.sets)}
        best = max(self.words, key=lambda w: sims[w])  # max keeps the first on ties
        return best, sims[best], sims

    def to_two_tuple(self, value: float) -> tuple[str, float]:
        """Map a value on the word scale (1..5 by default) to (word, alpha).

        beta = value - first_value; (s_i, alpha) = Delta(beta) (Herrera & Martinez 2000).
        Example: 2.8 -> ('Partially Appropriate', -0.2).
        """
        index, alpha = two_tuple.delta(value - self.first_value, self.g)
        return self.words[index], alpha


def appropriateness_codebook(fou_width: float = 0.25, x=None) -> Codebook:
    """Codebook for the five BHH appropriateness levels, centred on k = 1..5.

    Word k is an IT2 trapezoid with a shared core [k - 0.25, k + 0.25]:
      UMF = (k - 0.75 - w/2, k - 0.25, k + 0.25, k + 0.75 + w/2)
      LMF = (k - 0.75 + w/2, k - 0.25, k + 0.25, k + 0.75 - w/2)
    w = ``fou_width`` is the horizontal distance between UMF and LMF feet.
    With w = 0 the word is the T1 trapezoid (k-0.75, k-0.25, k+0.25, k+0.75),
    whose edges cross neighbouring words at k +/- 0.5 with membership 0.5.
    w is a PROXY: there is no clinician word survey, so experiments should
    report sensitivity to it. Requires 0 <= w <= 1 and the UMF inside ``x``.
    """
    if not 0 <= fou_width <= 1:
        raise ValueError('fou_width must lie in [0, 1]')
    x = DEFAULT_X if x is None else _check_x(x)
    half = fou_width / 2
    if 1 - 0.75 - half < x[0] - 1e-12 or 5 + 0.75 + half > x[-1] + 1e-12:
        raise ValueError('Codebook UMF exceeds the universe; widen x or reduce fou_width')
    sets = [it2_trapezoid((k - 0.75 - half, k - 0.25, k + 0.25, k + 0.75 + half),
                          (k - 0.75 + half, k - 0.25, k + 0.25, k + 0.75 - half), x)
            for k in range(1, 6)]
    return Codebook(APPROPRIATENESS_WORDS, sets, first_value=1.0)
