"""Entity F1: keyword-level accuracy that overall WER hides.

A transcript at a comfortable 5% WER that breaks the dollar amount is a
failed transcript. Entity F1 restricts scoring to the information-carrying
tokens: a reference entity counts as recalled iff it lies in an ``equal``
span of the minimal word-level alignment (the same engine as WER), and a
hypothesis entity counts as precise under the mirrored condition.

The entity subset is explicit: pass ``entities=`` (a set of token strings)
or ``predicate=`` (token -> bool). :func:`numeric_tokens` is a ready-made
predicate for digits and spelled numbers, the tokens that break amounts,
dates, and confirmation codes.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Union

from agwer.align import equal_flags, tokenize
from agwer.text import default_normalize

__all__ = ["entity_f1", "numeric_tokens"]

Strings = Union[str, Sequence[str]]

_SPELLED = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion",
}


def numeric_tokens(token: str) -> bool:
    """Digits and spelled numbers: the default information-carrying subset."""
    return any(c.isdigit() for c in token) or token in _SPELLED


def entity_f1(
    reference: Strings,
    hypothesis: Strings,
    entities: Optional[Iterable[str]] = None,
    predicate: Optional[Callable[[str], bool]] = None,
    normalize: Optional[Callable[[str], str]] = default_normalize,
) -> dict:
    """Entity-level precision, recall, and F1 over an explicit token subset.

    Provide exactly one of ``entities=`` (token set) or ``predicate=``
    (token -> bool). Accepts single strings or lists (pooled corpus-level).

    Returns ``recall`` (recalled reference entities / reference entities),
    ``precision`` (correct hypothesis entities / hypothesis entities),
    ``f1``, ``entity_wer`` (1 - recall: the subset miss rate), and supports.
    Values are ``None`` when the corresponding side contains no entities.
    """
    if (entities is None) == (predicate is None):
        raise ValueError("provide exactly one of entities= or predicate=")
    pred = set(entities).__contains__ if entities is not None else predicate

    refs = [reference] if isinstance(reference, str) else list(reference)
    hyps = [hypothesis] if isinstance(hypothesis, str) else list(hypothesis)
    if len(refs) != len(hyps):
        raise ValueError(
            f"reference ({len(refs)}) and hypothesis ({len(hyps)}) "
            "must have the same length"
        )
    norm = (lambda s: s) if normalize is None else normalize

    ref_total = ref_hit = hyp_total = hyp_hit = 0
    for r, h in zip(refs, hyps):
        rt, ht = tokenize(norm(r)), tokenize(norm(h))
        r_ok, h_ok, _ = equal_flags(rt, ht)
        for tok, ok in zip(rt, r_ok):
            if pred(tok):
                ref_total += 1
                ref_hit += ok
        for tok, ok in zip(ht, h_ok):
            if pred(tok):
                hyp_total += 1
                hyp_hit += ok

    recall = ref_hit / ref_total if ref_total else None
    precision = hyp_hit / hyp_total if hyp_total else None
    f1 = None
    if recall is not None and precision is not None and (recall + precision) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "entity_wer": None if recall is None else 1.0 - recall,
        "ref_entities": ref_total,
        "hyp_entities": hyp_total,
    }
