"""HER: utterance-level (paper-reported) and token-level (formal) accounting."""

import pytest

import agwer

# ---------- utterance granularity (paper-reported values) ----------

def test_her_utterance_mixed():
    refs = ["a b c", "d e f", "g h i"]
    onebest = ["a b c", "d x f", "g h i"]      # item0 correct, item1 has an error
    corrected = ["a b z", "d e f", "g h i"]    # broke item0, fixed item1, left item2
    out = agwer.evaluate(refs, corrected, onebest=onebest)
    assert out.edits.harmful == 1
    assert out.edits.helpful == 1
    assert out.edits.no_edit == 1
    assert out.her == pytest.approx(0.5)


def test_her_none_when_always_abstaining():
    refs = ["a b", "c d"]
    onebest = ["a x", "c d"]
    out = agwer.evaluate(refs, list(onebest), onebest=onebest)  # no edits at all
    assert out.her is None
    assert out.edits.no_edit == 2


def test_her_utterance_neutral_edit():
    # Changed the sentence but WER unchanged: neutral, not consequential.
    refs = ["a b c"]
    onebest = ["a x c"]
    corrected = ["a y c"]
    out = agwer.evaluate(refs, corrected, onebest=onebest)
    assert out.edits.neutral == 1
    assert out.her is None


# ---------- token granularity (formal per-edit definition) ----------

def test_her_token_fix_and_break():
    refs = ["a b c d"]
    onebest = ["a x c d"]     # token b wrong
    corrected = ["a b c y"]   # fixed b, broke d
    out = agwer.evaluate(refs, corrected, onebest=onebest, her_granularity="token")
    assert out.edits.helpful == 1
    assert out.edits.harmful == 1
    assert out.her == pytest.approx(0.5)


def test_her_token_missed_is_not_consequential():
    refs = ["a b c"]
    onebest = ["a x c"]
    corrected = ["a q c"]     # still wrong on token b: an edit, but not consequential
    out = agwer.evaluate(refs, corrected, onebest=onebest, her_granularity="token")
    assert out.edits.missed == 1
    assert out.her is None


def test_her_token_spurious_insertion_is_harmful():
    refs = ["a b"]
    onebest = ["a b"]
    corrected = ["a b z"]
    out = agwer.evaluate(refs, corrected, onebest=onebest, her_granularity="token")
    assert out.edits.harmful == 1
    assert out.her == pytest.approx(1.0)


def test_her_token_removing_insertion_is_helpful():
    refs = ["a b"]
    onebest = ["a b z"]
    corrected = ["a b"]
    out = agwer.evaluate(refs, corrected, onebest=onebest, her_granularity="token")
    assert out.edits.helpful == 1
    assert out.her == pytest.approx(0.0)


def test_her_functional_wrapper_granularities():
    # fix one token, break another: net WER is unchanged, so the utterance
    # granularity sees a neutral edit (HER None) while the token granularity
    # sees one helpful + one harmful edit (HER 0.5). This asymmetry is the
    # documented difference between the two accountings.
    refs = ["a b c d"]
    onebest = ["a x c d"]
    corrected = ["a b c y"]
    assert agwer.her(refs, onebest, corrected) is None
    assert agwer.her(refs, onebest, corrected, granularity="token") == pytest.approx(0.5)


def test_her_token_raw_whitespace_tokenization_matches_jiwer():
    # jiwer treats "a\tb" as ONE token (space is the only delimiter); the
    # token bookkeeping must agree with the alignment or counts get skewed.
    out = agwer.evaluate(
        ["a\tb"], ["a b"], onebest=["a b"], normalize=None,
        her_granularity="token",
    )
    assert out.edits.helpful == 0 and out.edits.harmful == 0
    assert out.edits.missed == 1  # the single ref token 'a\tb' is unmatched
    assert out.her is None
