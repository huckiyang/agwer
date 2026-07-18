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


def test_workers_merge_is_exact():
    # every aggregate is count-additive, so any worker count is bit-identical
    import random
    rng = random.Random(3)
    vocab = [f"w{i}" for i in range(50)]
    refs, nbest, corrected = [], [], []
    for _ in range(240):
        ref = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 12)))
        refs.append(ref)
        nbest.append([" ".join(rng.choice([t, rng.choice(vocab)])
                               for t in ref.split()) for _ in range(3)])
        corrected.append(" ".join(rng.choice([t, rng.choice(vocab)])
                                  for t in ref.split()))
    a = agwer.evaluate(refs, corrected, nbest=nbest)
    b = agwer.evaluate(refs, corrected, nbest=nbest, workers=2)
    assert a.as_dict() == b.as_dict()
    # token granularity + items too
    at = agwer.evaluate(refs, corrected, nbest=nbest,
                        her_granularity="token", return_items=True)
    bt = agwer.evaluate(refs, corrected, nbest=nbest,
                        her_granularity="token", return_items=True, workers=3)
    assert at.as_dict() == bt.as_dict()
    assert at.items == bt.items


def test_workers_with_cached_normalizer_and_lambda():
    # cached normalizers pickle transparently (fresh cache per worker)
    norm = agwer.EnglishTextNormalizer(cached=True)
    a = agwer.evaluate(["A b!"] * 4, ["a b"] * 4, onebest=["a x"] * 4, normalize=norm)
    b = agwer.evaluate(["A b!"] * 4, ["a b"] * 4, onebest=["a x"] * 4,
                       normalize=norm, workers=2)
    assert a.as_dict() == b.as_dict()
    # lambdas cannot cross process boundaries: clear error
    with pytest.raises(ValueError, match="picklable"):
        agwer.evaluate(["a b"] * 4, ["a b"] * 4, onebest=["a x"] * 4,
                       normalize=lambda s: s, workers=2)
