"""RIR (rho): constructed corpora with exactly known WERs."""

import pytest

import agwer


def test_rir_closes_gap_exactly():
    refs = ["a b c d"]
    nbest = [["a b c x", "a b c d"]]  # 1-best WER .25, oracle WER 0
    out = agwer.evaluate(refs, ["a b c d"], nbest=nbest)
    assert out.wer_1best == pytest.approx(0.25)
    assert out.wer_oracle == pytest.approx(0.0)
    assert out.rir == pytest.approx(1.0)


def test_rir_damage_regime_negative():
    refs = ["a b c d"]
    nbest = [["a b c x", "a b c d"]]
    out = agwer.evaluate(refs, ["q w e r"], nbest=nbest)  # corrected far worse
    assert out.rir < 0


def test_rir_partial_recovery():
    refs = ["a b c d e f g h"]
    nbest = [["a b c d e f x y", "a b c d e f g x"]]  # 1-best 2/8, oracle 1/8
    out = agwer.evaluate(refs, ["a b c d e f g x"], nbest=nbest)
    assert out.rir == pytest.approx(1.0)  # matches the oracle exactly


def test_rir_beyond_oracle():
    # Corrector produces the truth although no hypothesis contains it: rho > 1.
    refs = ["a b c d"]
    nbest = [["a x y d", "a x c d"]]  # 1-best 2 errors (.5), oracle 1 error (.25)
    out = agwer.evaluate(refs, ["a b c d"], nbest=nbest)
    assert out.rir == pytest.approx(2.0)
    assert out.rir > 1


def test_rir_none_without_headroom():
    refs = ["a b"]
    nbest = [["a x", "a x"]]  # oracle == 1-best, zero headroom
    out = agwer.evaluate(refs, ["a b"], nbest=nbest)
    assert out.rir is None
    assert out.headroom == pytest.approx(0.0)


def test_rir_functional_and_alias():
    refs = ["a b c d"]
    nbest = [["a b c x", "a b c d"]]
    assert agwer.rir(refs, ["a b c d"], nbest=nbest) == pytest.approx(1.0)
    assert agwer.rho(refs, ["a b c d"], nbest=nbest) == pytest.approx(1.0)


def test_rir_with_precomputed_oracle():
    refs = ["a b c d"]
    out = agwer.evaluate(
        refs, ["a b c d"], onebest=["a b c x"], oracle=["a b c d"]
    )
    assert out.rir == pytest.approx(1.0)


def test_rir_none_without_oracle_info():
    out = agwer.evaluate(["a b"], ["a b"], onebest=["a x"])
    assert out.rir is None and out.wer_oracle is None
    assert out.wer_corrected == pytest.approx(0.0)


def test_default_normalization_applies():
    refs = ["Hello, World!"]
    nbest = [["hello world", "hello duck"]]
    out = agwer.evaluate(refs, ["HELLO WORLD."], nbest=nbest)
    assert out.wer_corrected == pytest.approx(0.0)
    # raw scoring sees punctuation/case differences
    raw = agwer.evaluate(refs, ["HELLO WORLD."], nbest=nbest, normalize=None)
    assert raw.wer_corrected > 0
