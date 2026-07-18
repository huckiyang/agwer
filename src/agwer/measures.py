"""Functional one-liners for the agentic WER measures.

These are thin wrappers over :func:`agwer.process.process_agentic`; use that
(or :func:`agwer.evaluate`) directly when you need more than one number, so the
alignments are computed once.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import jiwer

from agwer.process import AgenticOutput, oracle_select, process_agentic
from agwer.transforms import apply_normalize, default_normalize

__all__ = ["evaluate", "rir", "rho", "her", "oracle_wer", "oracle_hypotheses",
           "compositional_oracle_wer"]


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
    refs_n = apply_normalize(references, normalize)
    picked = oracle_select(references, nbest, normalize=normalize)
    return jiwer.wer(refs_n, picked)


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
