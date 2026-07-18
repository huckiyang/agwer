"""Edit decomposition for the Harmful Edit Rate (HER).

Given (reference, 1-best, corrected), classify what the corrector's edits did.
Two granularities are provided:

* ``"utterance"`` - each utterance the corrector *edited* is labeled by its net
  WER effect: ``helpful`` (WER decreased), ``harmful`` (WER increased,
  i.e. over-correction), or ``neutral`` (changed but WER unchanged). Utterances
  left untouched are ``no_edit``. This is the granularity used for the HER
  values reported in the Voice Memory paper (e.g. 64% -> 35% on wsj).

* ``"token"`` - the paper's formal per-edit account. Both hypotheses are
  aligned to the reference; every reference token is scored as correct or not
  under each hypothesis, and each *changed outcome* is one edit:
  ``helpful`` (1-best wrong -> corrected right), ``harmful`` (1-best right ->
  corrected wrong). Reference tokens that stay wrong are ``missed``. Insertion
  changes are counted as edits too (extra insertions in the corrected output
  are harmful; removed 1-best insertions are helpful).

In both cases::

    HER = harmful / (harmful + helpful)

and HER is ``None`` when the corrector made no consequential edits
(the act/abstain reading: a corrector that always abstains has no HER).

Alignment runs on RapidFuzz Levenshtein opcodes over word tokens (split on
spaces, empty tokens dropped) - the same C++ alignment jiwer uses internally,
so classifications are equivalent (pinned by the alignment-equivalence
fixture in tests/).
"""

from __future__ import annotations

from dataclasses import dataclass

from agwer.align import equal_flags, pair_errors, tokenize

__all__ = ["EditCounts", "classify_utterance", "classify_tokens"]


@dataclass
class EditCounts:
    """Aggregated edit outcomes. ``her`` is harmful / (harmful + helpful)."""

    granularity: str
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    missed: int = 0
    no_edit: int = 0

    @property
    def consequential(self) -> int:
        return self.helpful + self.harmful

    @property
    def her(self):
        if self.consequential == 0:
            return None
        return self.harmful / self.consequential

    def as_dict(self) -> dict:
        return {
            "granularity": self.granularity,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "neutral": self.neutral,
            "missed": self.missed,
            "no_edit": self.no_edit,
            "her": self.her,
        }


def _pair_wer(reference: str, hypothesis: str) -> float:
    """Single-pair WER: word edit errors / reference length.

    Conventions: wer('', '') == 0; with an empty reference and a non-empty
    hypothesis the raw error (insertion) count is returned.
    """
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    errors = pair_errors([ref], [hyp])[0]
    if len(ref) == 0:
        return float(errors)
    return errors / len(ref)


def classify_utterance(reference: str, onebest: str, corrected: str) -> dict:
    """Net utterance-level classification (paper-reported granularity).

    All three strings are assumed to be already normalized.
    """
    edited = onebest != corrected
    wer_ob = _pair_wer(reference, onebest)
    wer_co = _pair_wer(reference, corrected)
    if not edited:
        category = "no_edit"
    elif wer_co < wer_ob:
        category = "helpful"
    elif wer_co > wer_ob:
        category = "harmful"
    else:
        category = "neutral"
    return {
        "category": category,
        "edited": edited,
        "wer_1best": wer_ob,
        "wer_corrected": wer_co,
    }


def _ref_token_hits(reference: str, hypothesis: str) -> tuple[list, int]:
    """Per-reference-token correctness under ``hypothesis`` + #insertions.

    Returns (ok, n_insertions) where ok[i] is True iff reference token i is
    matched exactly (an ``equal`` alignment span) in the hypothesis.
    """
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    ok, _, n_ins = equal_flags(ref, hyp)
    return ok, n_ins


def classify_tokens(reference: str, onebest: str, corrected: str) -> dict:
    """Token-level edit decomposition (the paper's formal definition).

    All three strings are assumed to be already normalized.
    """
    ob_ok, ob_ins = _ref_token_hits(reference, onebest)
    co_ok, co_ins = _ref_token_hits(reference, corrected)
    helpful = harmful = missed = kept = 0
    for was_ok, is_ok in zip(ob_ok, co_ok):
        if was_ok and not is_ok:
            harmful += 1
        elif not was_ok and is_ok:
            helpful += 1
        elif not was_ok and not is_ok:
            missed += 1
        else:
            kept += 1
    # Insertion deltas are edits too: adding spurious tokens is harmful,
    # removing 1-best insertions is helpful.
    if co_ins > ob_ins:
        harmful += co_ins - ob_ins
    elif ob_ins > co_ins:
        helpful += ob_ins - co_ins
    return {
        "helpful": helpful,
        "harmful": harmful,
        "missed": missed,
        "kept_correct": kept,
        "insertions_1best": ob_ins,
        "insertions_corrected": co_ins,
    }
