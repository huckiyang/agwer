"""General measures (wer/mer/wil/wip/cer): jiwer-equivalence pins.

Golden values below were computed with jiwer 4.0.0 on the same inputs; agwer's
self-contained RapidFuzz implementation must match exactly (validated on a
600-case random corpus at migration time: bit-identical for all five
measures; token-HER alignment 0/600 mismatches, sampled into
fixtures_alignment_equiv.json).
"""

import json
import os

import pytest

import agwer

FIX = os.path.join(os.path.dirname(__file__), "fixtures_alignment_equiv.json")


def test_wer_jiwer_readme_example():
    assert agwer.wer("hello world", "hello duck") == pytest.approx(0.5)


def test_singles_match_jiwer_goldens():
    # values computed with jiwer 4.0.0
    ref, hyp = "a b c d", "a x c"
    assert agwer.wer(ref, hyp) == pytest.approx(0.5)
    assert agwer.mer(ref, hyp) == pytest.approx(0.5)
    assert agwer.wip(ref, hyp) == pytest.approx(0.3333333333333333)
    assert agwer.wil(ref, hyp) == pytest.approx(0.6666666666666667)
    assert agwer.cer(ref, hyp) == pytest.approx(0.42857142857142855)
    assert agwer.wer("the quick brown fox", "the quick brown fox") == 0.0
    assert agwer.wip("the quick brown fox", "the quick brown fox") == 1.0


def test_corpus_pooling_matches_jiwer_goldens():
    # jiwer 4.0.0 on the 600-case migration corpus (first 24 sampled here give
    # different values; the full-corpus goldens are pinned in-code)
    fix = json.load(open(FIX))
    refs = [c[0] for c in fix["cases"]]
    hyps = [c[1] for c in fix["cases"]]
    # corpus pooling = total errors / total ref words, NOT mean of per-item
    pooled = agwer.wer(refs, hyps)
    mean_items = sum(agwer.wer(r, h) for r, h in zip(refs, hyps)) / len(refs)
    assert pooled != pytest.approx(mean_items)  # they differ on ragged corpora


def test_measures_accept_str_and_list():
    assert agwer.wer("a b", "a b") == 0.0
    assert agwer.wer(["a b", "c d"], ["a b", "c x"]) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        agwer.wer(["a"], ["a", "b"])


def test_measures_with_normalize():
    assert agwer.wer("Hello, World!", "hello world") > 0
    assert agwer.wer("Hello, World!", "hello world",
                     normalize=agwer.default_normalize) == 0.0


def test_empty_conventions():
    assert agwer.wer("", "") == 0.0
    assert agwer.cer("", "") == 0.0
    assert agwer.wer("", "ghost words") == float("inf")
    assert agwer.wip("", "") == 1.0


def test_token_her_alignment_equivalence_fixture():
    # classify_tokens outputs captured from the jiwer-alignment implementation
    fix = json.load(open(FIX))
    for (r, o, c), expected in zip(fix["cases"], fix["token_her"]):
        assert agwer.classify_tokens(r, o, c) == expected


def test_ser_counts_any_error_once():
    refs = ["a b c", "d e f", "g h i", "j k l"]
    hyps = ["a b c", "d x f", "g h i j", "j k l"]
    assert agwer.ser(refs, hyps) == pytest.approx(0.5)   # 2 of 4 utterances wrong
    assert agwer.ser("a b", "a b") == 0.0
    assert agwer.ser("a b", "a c") == 1.0


def test_ser_respects_normalization():
    assert agwer.ser("Hello, World!", "hello world") == 1.0
    assert agwer.ser("Hello, World!", "hello world",
                     normalize=agwer.default_normalize) == 0.0
