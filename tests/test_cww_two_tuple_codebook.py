"""Synthetic tests: 2-tuples and the appropriateness codebook decoder."""
import numpy as np
import pytest

from src.cww import (APPROPRIATENESS_WORDS, appropriateness_codebook, centre, encode_scores,
                     two_tuple)

G = 4  # five terms s_0..s_4


@pytest.mark.parametrize('beta', np.linspace(0, G, 401))
def test_delta_roundtrip_and_alpha_range(beta):
    t = two_tuple.delta(beta, G)
    assert two_tuple.delta_inv(t) == pytest.approx(beta)
    assert -0.5 <= t.alpha < 0.5
    assert 0 <= t.index <= G


def test_delta_half_rounds_up():
    assert two_tuple.delta(1.5, G) == (2, -0.5)
    assert two_tuple.delta(G, G) == (G, 0.0)
    with pytest.raises(ValueError):
        two_tuple.delta(4.5, G)


def test_mean_of_tuples_equals_delta_of_mean_beta():
    betas = [0.3, 2.6, 3.9, 1.5]
    tuples = [two_tuple.delta(b, G) for b in betas]
    assert two_tuple.delta_inv(two_tuple.mean(tuples, G)) == pytest.approx(np.mean(betas))
    weights = [1, 2, 3, 4]
    expected = np.dot(weights, betas) / sum(weights)
    assert two_tuple.delta_inv(two_tuple.weighted_mean(tuples, weights, G)) == pytest.approx(expected)


@pytest.mark.parametrize('fou_width', [0.0, 0.1, 0.25, 0.5])
def test_codebook_decodes_own_words(fou_width):
    book = appropriateness_codebook(fou_width)
    for word, s in zip(book.words, book.sets):
        decoded, similarity, sims = book.decode(s)
        assert decoded == word and similarity == pytest.approx(1.0)
        assert set(sims) == set(APPROPRIATENESS_WORDS)


def test_codebook_word_centroids_and_fou():
    book = appropriateness_codebook(0.25)
    for k, s in enumerate(book.sets, 1):
        assert centre(s) == pytest.approx(k, abs=1e-9)
        assert not s.is_type1
    assert all(s.is_type1 for s in appropriateness_codebook(0.0).sets)
    with pytest.raises(ValueError):
        appropriateness_codebook(0.8)  # UMF of word 1 would leave [0, 6]


def test_to_two_tuple():
    book = appropriateness_codebook()
    word, alpha = book.to_two_tuple(2.8)
    assert word == 'Partially Appropriate' and alpha == pytest.approx(-0.2)
    assert book.to_two_tuple(1.0) == ('Harmful/Unacceptable', 0.0)
    assert book.to_two_tuple(5.0) == ('Fully Appropriate', 0.0)


def test_decode_encoded_panel():
    book = appropriateness_codebook()
    assert book.decode(encode_scores([4, 4]))[0] == 'Mostly Appropriate'
