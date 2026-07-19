"""Classic ASR measures: WER, MER, WIL, WIP, CER, SER.

Semantics are jiwer-compatible and bit-identical on the same tokenization
(pinned by golden tests). Each function accepts a single string or a list of
strings for both arguments; lists are pooled corpus-level (total counts, not
per-utterance averages). By default strings are scored exactly as given;
pass ``normalize=agwer.default_normalize`` (or any callable) to normalize
first.

WER and CER take the fast path (:func:`agwer.align.pair_errors`): the error
count is the C-level edit distance and no alignment is materialized. MER,
WIP, and WIL need hit counts, so they walk the pooled alignment.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

from agwer.align import pair_errors, pooled_counts, tokenize_all
from agwer.text import apply_normalize

__all__ = ["wer", "mer", "wil", "wip", "cer", "ser"]

Strings = Union[str, Sequence[str]]


def _prep(reference: Strings, hypothesis: Strings,
          normalize: Optional[Callable[[str], str]]) -> tuple:
    refs = [reference] if isinstance(reference, str) else list(reference)
    hyps = [hypothesis] if isinstance(hypothesis, str) else list(hypothesis)
    if len(refs) != len(hyps):
        raise ValueError(
            f"reference ({len(refs)}) and hypothesis ({len(hyps)}) "
            "must have the same length"
        )
    return apply_normalize(refs, normalize), apply_normalize(hyps, normalize)


def _is_tokens(x) -> bool:
    """A pre-tokenized batch: a sequence whose entries are token lists."""
    return not isinstance(x, str) and len(x) > 0 and not isinstance(x[0], str)


def _prep_tokens(reference, hypothesis,
                 normalize: Optional[Callable[[str], str]]) -> tuple:
    """Token lists for the word-level measures.

    Accepts strings / lists of strings (normalized then tokenized here) or
    pre-tokenized ``list[list[str]]`` batches, which skip both steps — the
    fast path for repeated batch scoring (tokenize once, score many times).
    """
    if _is_tokens(reference) or _is_tokens(hypothesis):
        if normalize is not None:
            raise ValueError(
                "pre-tokenized input requires normalize=None; "
                "normalize the strings before tokenizing them"
            )
        refs, hyps = list(reference), list(hypothesis)
        if len(refs) != len(hyps):
            raise ValueError(
                f"reference ({len(refs)}) and hypothesis ({len(hyps)}) "
                "must have the same length"
            )
        return refs, hyps
    refs, hyps = _prep(reference, hypothesis, normalize)
    return tokenize_all(refs), tokenize_all(hyps)


def _error_rate(errors: int, n_ref: int) -> float:
    if n_ref == 0:
        return 0.0 if errors == 0 else float("inf")
    return errors / n_ref


def wer(reference: Strings, hypothesis: Strings,
        normalize: Optional[Callable[[str], str]] = None) -> float:
    """Word error rate: (S + D + I) / reference words. 0 when both empty.

    Batch fast path: pass pre-tokenized ``list[list[str]]`` for both sides
    (with ``normalize=None``) to skip per-call tokenization — tokenize a
    corpus once, score it many times.
    """
    refs_tok, hyps_tok = _prep_tokens(reference, hypothesis, normalize)
    errors = sum(pair_errors(refs_tok, hyps_tok))
    return _error_rate(errors, sum(map(len, refs_tok)))


def cer(reference: Strings, hypothesis: Strings,
        normalize: Optional[Callable[[str], str]] = None) -> float:
    """Character error rate (spaces count as characters). 0 when both empty."""
    if _is_tokens(reference) or _is_tokens(hypothesis):
        raise ValueError("cer scores characters; pass strings, not token lists")
    refs, hyps = _prep(reference, hypothesis, normalize)
    errors = sum(pair_errors(refs, hyps))
    return _error_rate(errors, sum(map(len, refs)))


def mer(reference: Strings, hypothesis: Strings,
        normalize: Optional[Callable[[str], str]] = None) -> float:
    """Match error rate: (S + D + I) / (H + S + D + I)."""
    refs, hyps = _prep_tokens(reference, hypothesis, normalize)
    h, s, d, i = pooled_counts(refs, hyps)
    total = h + s + d + i
    return (s + d + i) / total if total else 0.0


def wip(reference: Strings, hypothesis: Strings,
        normalize: Optional[Callable[[str], str]] = None) -> float:
    """Word information preserved: (H/N_ref) * (H/N_hyp)."""
    refs, hyps = _prep_tokens(reference, hypothesis, normalize)
    h, s, d, i = pooled_counts(refs, hyps)
    n_ref, n_hyp = h + s + d, h + s + i
    if n_ref == 0 or n_hyp == 0:
        return 1.0 if (n_ref == 0 and n_hyp == 0) else 0.0
    return (h / n_ref) * (h / n_hyp)


def wil(reference: Strings, hypothesis: Strings,
        normalize: Optional[Callable[[str], str]] = None) -> float:
    """Word information lost: 1 - WIP."""
    return 1.0 - wip(reference, hypothesis, normalize)


def ser(reference: Strings, hypothesis: Strings,
        normalize: Optional[Callable[[str], str]] = None) -> float:
    """Sentence (utterance) error rate: the fraction of utterances whose
    token sequence differs from the reference at all. The strictest
    utterance-level measure; standard in ASR reporting."""
    refs_tok, hyps_tok = _prep_tokens(reference, hypothesis, normalize)
    wrong = sum(1 for r, h in zip(refs_tok, hyps_tok) if r != h)
    return wrong / len(refs_tok) if refs_tok else 0.0
