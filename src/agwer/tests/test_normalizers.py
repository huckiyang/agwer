"""Normalizers: golden equivalence with the original Whisper implementation,
behavior spot checks, and integration with evaluate()."""

import json
import os

import pytest

import agwer
from agwer.normalizers import BasicTextNormalizer, EnglishTextNormalizer

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures_whisper_golden.json")


# ---------------- golden: byte-identical to the original Whisper normalizers

def _golden(section):
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)[section]


def test_english_matches_original_whisper():
    norm = EnglishTextNormalizer()
    for src, expected in _golden("english").items():
        assert norm(src) == expected, f"mismatch for {src!r}"


def test_basic_matches_original_whisper():
    norm = BasicTextNormalizer()
    for src, expected in _golden("basic").items():
        assert norm(src) == expected, f"mismatch for {src!r}"


def test_basic_diacritics_matches_original_whisper():
    norm = BasicTextNormalizer(remove_diacritics=True)
    for src, expected in _golden("basic_diacritics").items():
        assert norm(src) == expected, f"mismatch for {src!r}"


# ---------------- behavior spot checks (readable documentation)

def test_english_flagship_behaviors():
    norm = EnglishTextNormalizer()
    assert norm("twenty two dollars and fifty cents") == "$22.50"
    assert norm("fifty percent of one hundred is fifty") == "50% of 100 is 50"
    assert norm("the colour of the aluminium armour") == "the color of the aluminum armor"
    assert norm("Mr. Brown won't go") == "mister brown will not go"
    assert norm("um, I think, uh, we should go") == "i think we should go"
    assert norm("hello [noise] world (laughter) again") == "hello world again"


def test_cached_is_equivalent():
    plain = EnglishTextNormalizer()
    cached = EnglishTextNormalizer(cached=True)
    for src in _golden("english"):
        assert cached(src) == plain(src)
    # repeated call hits the LRU and stays correct
    assert cached("twenty two dollars") == plain("twenty two dollars")


# ---------------- integration: normalization changes the metric verdict

def test_evaluate_with_english_normalizer():
    refs = ["it costs $22.50 today", "mister brown left"]
    onebest = ["it costs twenty two dollars and fifty cents today", "mr. brown left"]
    # a "corrector" that only rewrote surface forms should not be penalized
    out = agwer.evaluate(
        refs, list(onebest), onebest=onebest, normalize=EnglishTextNormalizer()
    )
    assert out.wer_corrected == pytest.approx(0.0)
    # under the conservative default normalizer these count as errors
    out_default = agwer.evaluate(refs, list(onebest), onebest=onebest)
    assert out_default.wer_corrected > 0
