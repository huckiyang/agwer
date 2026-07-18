"""The auto-banded fast path must be invisible: exact everywhere.

pair_errors switches to a banded distance above _BAND_MIN elements. These
tests pin that the banded path returns exactly the plain Levenshtein
distance in every regime the band prior can get wrong: error rates far
above the prior, the gate boundary, empty inputs, strings and token lists.
"""

import random

import pytest
from rapidfuzz.distance import Levenshtein

import agwer
from agwer.align import _BAND_MIN, pair_errors


def _pair(n_words, wer, seed, vocab_size=500):
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(vocab_size)]
    ref = [rng.choice(vocab) for _ in range(n_words)]
    hyp = []
    for t in ref:
        r = rng.random()
        if r < wer * 0.6:
            hyp.append(rng.choice(vocab))
        elif r < wer * 0.8:
            continue
        elif r < wer:
            hyp.extend([t, rng.choice(vocab)])
        else:
            hyp.append(t)
    return ref, hyp


@pytest.mark.parametrize("wer", [0.0, 0.05, 0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n", [_BAND_MIN - 1, _BAND_MIN, _BAND_MIN + 1, 2_000])
def test_banded_equals_plain_all_regimes(n, wer):
    # error rates far above the 1/4 prior force band doubling; the result
    # must still be the exact distance
    ref, hyp = _pair(n, wer, seed=n + int(wer * 100))
    assert pair_errors([ref], [hyp]) == [Levenshtein.distance(ref, hyp)]


def test_banded_equals_plain_random_mix():
    rng = random.Random(7)
    refs, hyps = [], []
    for _ in range(300):
        r, h = _pair(rng.randint(0, 900), rng.random(), seed=rng.randint(0, 10**6))
        refs.append(r)
        hyps.append(h)
    assert pair_errors(refs, hyps) == [
        Levenshtein.distance(r, h) for r, h in zip(refs, hyps)
    ]


def test_banded_strings_char_level():
    # cer() feeds raw strings through the same path (1/8 prior)
    rng = random.Random(11)
    ref = " ".join(f"w{rng.randint(0, 200)}" for _ in range(1_500))
    for wer in (0.02, 0.3, 0.95):
        hyp = " ".join(
            f"w{rng.randint(0, 200)}" if rng.random() < wer else t
            for t in ref.split()
        )
        assert pair_errors([ref], [hyp]) == [Levenshtein.distance(ref, hyp)]


def test_banded_degenerate_inputs():
    long_ref = ["a"] * (_BAND_MIN + 50)
    assert pair_errors([long_ref], [[]]) == [len(long_ref)]
    assert pair_errors([[]], [long_ref]) == [len(long_ref)]
    assert pair_errors([long_ref], [long_ref]) == [0]
    # totally disjoint: distance == max length, far beyond the prior
    other = ["b"] * (_BAND_MIN + 50)
    assert pair_errors([long_ref], [other]) == [len(long_ref)]


def test_public_measures_unchanged_on_long_documents():
    # wer/cer through the public API agree with a naive plain-distance WER
    ref_t, hyp_t = _pair(3_000, 0.2, seed=3)
    ref, hyp = " ".join(ref_t), " ".join(hyp_t)
    assert agwer.wer(ref, hyp) == Levenshtein.distance(ref_t, hyp_t) / len(ref_t)
    assert agwer.cer(ref, hyp) == Levenshtein.distance(ref, hyp) / len(ref)
