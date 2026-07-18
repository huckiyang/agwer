"""Compositional oracle (o_cp, HyPoradise Sec. 5.2)."""

import random

import pytest

import agwer


def test_ocp_recovers_across_hypotheses():
    # No single hypothesis is correct, but the union of tokens covers the
    # reference: o_cp = 0 while o_nb > 0. This is the definitional
    # distinction between the two oracles.
    refs = ["a b c"]
    nbest = [["a b x", "c d e"]]
    assert agwer.compositional_oracle_wer(refs, nbest) == pytest.approx(0.0)
    assert agwer.oracle_wer(refs, nbest) > 0


def test_ocp_counts_uncovered_reference_tokens():
    refs = ["a b c"]
    nbest = [["a x", "a y"]]  # 'b' and 'c' occur in no hypothesis
    assert agwer.compositional_oracle_wer(refs, nbest) == pytest.approx(2 / 3)


def test_ocp_zero_when_any_hypothesis_correct():
    refs = ["hello world"]
    nbest = [["hello world", "hello word"]]
    assert agwer.compositional_oracle_wer(refs, nbest) == pytest.approx(0.0)


def test_ocp_never_exceeds_onb():
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(30)]
    refs, nbest = [], []
    for _ in range(300):
        ref = [rng.choice(vocab) for _ in range(rng.randint(3, 12))]
        refs.append(" ".join(ref))
        hyps = []
        for _ in range(5):
            hyps.append(" ".join(
                t if rng.random() > 0.3 else rng.choice(vocab) for t in ref))
        nbest.append(hyps)
    ocp = agwer.compositional_oracle_wer(refs, nbest)
    onb = agwer.oracle_wer(refs, nbest)
    assert ocp <= onb + 1e-12


def test_evaluate_populates_wer_compositional():
    refs = ["a b c d"]
    nbest = [["a b c x", "a b y d"]]
    out = agwer.evaluate(refs, ["a b c d"], nbest=nbest)
    assert out.wer_compositional == pytest.approx(0.0)   # union covers ref
    assert out.wer_oracle == pytest.approx(0.25)
    assert "wer_compositional" in out.as_dict()
    # onebest-only input: no n-best list, no compositional oracle
    out2 = agwer.evaluate(refs, ["a b c d"], onebest=["a b c x"])
    assert out2.wer_compositional is None


def test_ocp_uses_normalization():
    refs = ["Twenty-Two Dollars!"]
    nbest = [["twenty two", "dollars please"]]
    assert agwer.compositional_oracle_wer(refs, nbest) == pytest.approx(0.0)
