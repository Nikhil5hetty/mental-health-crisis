"""Computing-with-Words core (Perceptual Computer: encoder -> CWW engine -> decoder).

numpy-only maths library. See docs/METHOD.md for formulas and references.
"""
from .aggregate import AlphaCuts, fwa_alpha_cut, lwa, lwa_cuts, orness, owa, owa_weights
from .centroid import centre, centroid_it2, centroid_t1, ekm_left, ekm_right, width
from .codebook import APPROPRIATENESS_WORDS, Codebook, appropriateness_codebook
from .iaa import encode_panel, encode_scores, iaa_t1, majority_vote, score_interval, vote_degree
from .sets import DEFAULT_X, IT2Set, T1Set, as_it2, it2_trapezoid, trapezoid, universe
from .similarity import jaccard, jaccard_t1
from . import two_tuple

__all__ = [
    'AlphaCuts', 'fwa_alpha_cut', 'lwa', 'lwa_cuts', 'orness', 'owa', 'owa_weights',
    'centre', 'centroid_it2', 'centroid_t1', 'ekm_left', 'ekm_right', 'width',
    'APPROPRIATENESS_WORDS', 'Codebook', 'appropriateness_codebook',
    'encode_panel', 'encode_scores', 'iaa_t1', 'majority_vote', 'score_interval', 'vote_degree',
    'DEFAULT_X', 'IT2Set', 'T1Set', 'as_it2', 'it2_trapezoid', 'trapezoid', 'universe',
    'jaccard', 'jaccard_t1', 'two_tuple',
]
