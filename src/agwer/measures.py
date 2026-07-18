"""Measures: general ASR similarity measures plus the agentic one-liners.

General measures (jiwer-compatible semantics, self-contained on RapidFuzz):
:func:`wer`, :func:`mer`, :func:`wil`, :func:`wip`, :func:`cer`. Each accepts
a single string or a list of strings for both arguments; lists are pooled
corpus-level (total counts, not per-utterance averages), matching jiwer. By
default strings are scored exactly as given; pass
``normalize=agwer.default_normalize`` (or any callable) to normalize first.

The agentic one-liners are thin wrappers over
:func:`agwer.process.process_agentic`; use that (or :func:`agwer.evaluate`)
directly when you need more than one number, so alignments are computed once.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

from rapidfuzz.distance import Levenshtein

from agwer.process import AgenticOutput, oracle_select, process_agentic
from agwer.transforms import apply_normalize, default_normalize

__all__ = ["wer", "mer", "wil", "wip", "cer",
           "evaluate", "rir", "rho", "her",
           "oracle_wer", "oracle_hypotheses", "compositional_oracle_wer"]

Strings = Union[str, Sequence[str]]


def _as_list(x: Strings) -> list:
    return [x] if isinstance(x, str) else list(x)


def _counts(reference: Strings, hypothesis: Strings,
            normalize: Optional[Callable[[str], str]], chars: bool):
    """Pooled alignment counts (hits, substitutions, deletions, insertions)."""
    refs, hyps = _as_list(reference), _as_list(hypothesis)
    if len(refs) != len(hyps):
        raise ValueError(
            f"reference ({len(refs)}) and hypothesis ({len(hyps)}) "
            "must have the same length"
        )
    refs = apply_normalize(refs, normalize)
    hyps = apply_normalize(hyps, normalize)
    hits = subs = dels = ins = 0
    for r, h in zip(refs, hyps):
        if not chars:
            r = [t for t in r.split(" ") if t]
            h = [t for t in h.split(" ") if t]
        for op in Levenshtein.opcodes(r, h):
            n_src = op.src_end - op.src_start
            if op.tag == "equal":
                hits += n_src
            elif op.tag == "replace":
                subs += n_src
            elif op.tag == "delete":
                dels += n_src
            else:  # insert
                ins += op.dest_end - op.dest_start
    return hits, subs, dels, ins


def wer(reference: Strings, hypothesis: Strings,
        normalize: Optional[Callable[[str], str]] = None) -> float:
    """Word error rate: (S + D + I) / reference words. 0 when both empty."""
    h, s, d, i = _counts(reference, hypothesis, normalize, chars=False)
    n_ref = h + s + d
    if n_ref == 0:
        return 0.0 if i == 0 else float("inf")
    return (s + d + i) / n_ref


def mer(reference: Strings, hypothesis: Strings,
        normalize: Optional[Callable[[str], str]] = None) -> float:
    """Match error rate: (S + D + I) / (H + S + D + I)."""
    h, s, d, i = _counts(reference, hypothesis, normalize, chars=False)
    total = h + s + d + i
    return (s + d + i) / total if total else 0.0


def wip(reference: Strings, hypothesis: Strings,
        normalize: Optional[Callable[[str], str]] = None) -> float:
    """Word information preserved: (H/N_ref) * (H/N_hyp)."""
    h, s, d, i = _counts(reference, hypothesis, normalize, chars=False)
    n_ref, n_hyp = h + s + d, h + s + i
    if n_ref == 0 or n_hyp == 0:
        return 1.0 if (n_ref == 0 and n_hyp == 0) else 0.0
    return (h / n_ref) * (h / n_hyp)


def wil(reference: Strings, hypothesis: Strings,
        normalize: Optional[Callable[[str], str]] = None) -> float:
    """Word information lost: 1 - WIP."""
    return 1.0 - wip(reference, hypothesis, normalize)


def cer(reference: Strings, hypothesis: Strings,
        normalize: Optional[Callable[[str], str]] = None) -> float:
    """Character error rate (spaces count as characters). 0 when both empty."""
    h, s, d, i = _counts(reference, hypothesis, normalize, chars=True)
    n_ref = h + s + d
    if n_ref == 0:
        return 0.0 if i == 0 else float("inf")
    return (s + d + i) / n_ref


def evaluate(
    references: Sequence[str],
    corrected: Sequence[str],
    nbest: Optional[Sequence[Sequence[str]]] = None,
    onebest: Optional[Sequence[str]] = None,
    oracle: Optional[Sequence[str]] = None,
    normalize: Optional[Callable[[str], str]] = default_normalize,
    her_granularity: str = "utterance",
    return_items: bool = False,
) -> AgenticOutput:
    """Compute all agentic metrics for a corpus. See :func:`process_agentic`."""
    return process_agentic(
        references,
        corrected,
        nbest=nbest,
        onebest=onebest,
        oracle=oracle,
        normalize=normalize,
        her_granularity=her_granularity,
        return_items=return_items,
    )


def rir(
    references: Sequence[str],
    corrected: Sequence[str],
    nbest: Optional[Sequence[Sequence[str]]] = None,
    onebest: Optional[Sequence[str]] = None,
    oracle: Optional[Sequence[str]] = None,
    normalize: Optional[Callable[[str], str]] = default_normalize,
) -> Optional[float]:
    """Recoverable Information Ratio (rho), Voice Memory paper Eq. (1).

        rho = (WER_1best - WER_corrected) / (WER_1best - WER_oracle)

    rho = 1 exactly closes the 1-best-to-oracle gap; rho < 0 is the damage
    regime (correcting is worse than keeping the 1-best); rho > 1 means the
    corrector recovered tokens present in no hypothesis and beat the oracle
    bound. Returns ``None`` when there is no headroom (oracle == 1-best).
    """
    return process_agentic(
        references, corrected, nbest=nbest, onebest=onebest, oracle=oracle,
        normalize=normalize,
    ).rir


rho = rir  # the paper's symbol


def her(
    references: Sequence[str],
    onebest: Sequence[str],
    corrected: Sequence[str],
    normalize: Optional[Callable[[str], str]] = default_normalize,
    granularity: str = "utterance",
) -> Optional[float]:
    """Harmful Edit Rate: of the corrector's consequential edits, the fraction
    that broke a correct token (over-correction).

    ``granularity="utterance"`` reproduces the paper's reported values;
    ``granularity="token"`` follows the paper's formal per-edit definition.
    Returns ``None`` when the corrector made no consequential edits.
    """
    return process_agentic(
        references, corrected, onebest=onebest, normalize=normalize,
        her_granularity=granularity,
    ).her


def oracle_hypotheses(
    references: Sequence[str],
    nbest: Sequence[Sequence[str]],
    normalize: Optional[Callable[[str], str]] = default_normalize,
) -> list:
    """Per-utterance minimum-error hypothesis from each n-best list."""
    return oracle_select(references, nbest, normalize=normalize)


def oracle_wer(
    references: Sequence[str],
    nbest: Sequence[Sequence[str]],
    normalize: Optional[Callable[[str], str]] = default_normalize,
) -> float:
    """Corpus WER of the n-best oracle o_nb (the reranking lower bound)."""
    onebest = [h[0] for h in nbest]
    return process_agentic(
        references, onebest, nbest=nbest, normalize=normalize
    ).wer_oracle


def compositional_oracle_wer(
    references: Sequence[str],
    nbest: Sequence[Sequence[str]],
    normalize: Optional[Callable[[str], str]] = default_normalize,
) -> float:
    """Corpus WER of the compositional oracle o_cp (HyPoradise Sec. 5.2): the
    WER achievable composing any sequence from the tokens occurring in the
    n-best list -- the upper bound of *correction* using in-list elements.
    Always <= oracle_wer. Each reference token absent from every hypothesis
    costs one error; everything else is recoverable for free."""
    onebest = [h[0] for h in nbest]
    return process_agentic(
        references, onebest, nbest=nbest, normalize=normalize
    ).wer_compositional
