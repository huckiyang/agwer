"""Pre-tokenized list[list[str]] input: identical results, no surprises."""

import random

import pytest

import agwer
from agwer.align import tokenize


def _corpus(n, seed):
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(200)]
    refs, hyps = [], []
    for _ in range(n):
        ref = [rng.choice(vocab) for _ in range(rng.randint(1, 25))]
        hyp = [rng.choice(vocab) if rng.random() < 0.15 else t for t in ref]
        refs.append(" ".join(ref))
        hyps.append(" ".join(hyp))
    return refs, hyps


def test_word_measures_identical_on_pretokenized():
    refs, hyps = _corpus(300, seed=9)
    refs_tok = [tokenize(s) for s in refs]
    hyps_tok = [tokenize(s) for s in hyps]
    for fn in (agwer.wer, agwer.mer, agwer.wip, agwer.wil, agwer.ser):
        assert fn(refs_tok, hyps_tok) == fn(refs, hyps), fn.__name__


def test_single_word_utterances_stay_strings():
    # list of single-word STRINGS is still a list of utterances,
    # not a token list: no behavior change for existing users
    assert agwer.wer(["a", "b"], ["b", "a"]) == 1.0


def test_pretokenized_requires_no_normalize():
    with pytest.raises(ValueError, match="normalize=None"):
        agwer.wer([["a", "b"]], [["a", "b"]], normalize=agwer.default_normalize)


def test_cer_rejects_token_lists():
    with pytest.raises(ValueError, match="pass strings"):
        agwer.cer([["a", "b"]], [["a", "b"]])


def test_empty_and_degenerate():
    assert agwer.wer([], []) == 0.0
    assert agwer.wer([[]], [[]]) == 0.0
    assert agwer.wer([["a"]], [[]]) == 1.0
    assert agwer.wer([[]], [["a"]]) == float("inf")
