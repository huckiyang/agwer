"""End-to-end evaluate(), input validation, per-item rows, serialization."""

import pytest

import agwer

REFS = ["the cat sat", "hello world", "speech is hard"]
NBEST = [
    ["the cat sad", "the cat sat", "a cat sad"],
    ["hello world", "hello word", "jello world"],
    ["speech is hard", "speech was hard", "peach is hard"],
]
CORRECTED = ["the cat sat", "hello world", "speech is hard"]


def test_evaluate_end_to_end():
    out = agwer.evaluate(REFS, CORRECTED, nbest=NBEST)
    assert out.n == 3
    assert out.wer_corrected == pytest.approx(0.0)
    assert out.wer_oracle == pytest.approx(0.0)   # truth is in every n-best list
    assert out.rir == pytest.approx(1.0)
    d = out.as_dict()
    for key in ("n", "wer_1best", "wer_corrected", "wer_oracle", "rir", "her", "edits"):
        assert key in d


def test_return_items_rows():
    out = agwer.evaluate(REFS, CORRECTED, nbest=NBEST, return_items=True)
    assert len(out.items) == 3
    assert {"index", "reference", "onebest", "corrected", "category"} <= set(out.items[0])


def test_return_items_token_rows():
    out = agwer.evaluate(
        REFS, CORRECTED, nbest=NBEST, her_granularity="token", return_items=True
    )
    assert {"helpful", "harmful", "missed"} <= set(out.items[0])


def test_oracle_helpers():
    picked = agwer.oracle_hypotheses(REFS, NBEST)
    assert picked == ["the cat sat", "hello world", "speech is hard"]
    assert agwer.oracle_wer(REFS, NBEST) == pytest.approx(0.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        agwer.evaluate(REFS, CORRECTED[:2], nbest=NBEST)
    with pytest.raises(ValueError):
        agwer.evaluate(REFS, CORRECTED, nbest=NBEST[:2])
    with pytest.raises(ValueError):
        agwer.evaluate(REFS, CORRECTED, onebest=["x"])


def test_missing_hypothesis_source_raises():
    with pytest.raises(ValueError):
        agwer.evaluate(REFS, CORRECTED)


def test_bad_granularity_raises():
    with pytest.raises(ValueError):
        agwer.evaluate(REFS, CORRECTED, nbest=NBEST, her_granularity="phoneme")


def test_empty_corpus_raises():
    with pytest.raises(ValueError):
        agwer.evaluate([], [], nbest=[])


def test_empty_nbest_list_raises():
    with pytest.raises(ValueError):
        agwer.evaluate(["a"], ["a"], nbest=[[]])


def test_empty_string_pairs_are_defined():
    # wer("", "") is 0 by convention; hallucination on silence is scored.
    out = agwer.evaluate(["", "a b"], ["", "a b"], onebest=["", "a x"])
    assert out.wer_corrected == pytest.approx(0.0)


def test_all_empty_corpus_wer_is_zero():
    # An all-silence corpus with all-empty outputs has zero error by convention.
    out = agwer.evaluate(["", ""], ["", ""], onebest=["", ""])
    assert out.wer_corrected == pytest.approx(0.0)
    assert out.wer_1best == pytest.approx(0.0)
    assert out.her is None
