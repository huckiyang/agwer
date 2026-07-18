"""Core processing for agentic WER metrics.

One call — :func:`process_agentic` — computes everything the Voice Memory
paper's evaluation needs from (references, n-best lists, corrected outputs):

* corpus WER of the 1-best, the n-best oracle, the compositional oracle,
  and the corrected output,
* the Recoverable Information Ratio (RIR / rho),
* the Harmful Edit Rate (HER) with its edit decomposition,
* optional per-utterance rows for error analysis.

Performance design: word-level edit distances are computed directly with
RapidFuzz (the same C++ engine jiwer uses), batched over the whole corpus, so
the full agentic evaluation costs a handful of batch calls rather than
per-utterance Python round-trips. Corpus WER is the global error ratio
(total edit errors / total reference words) and is bit-identical to
``jiwer.wer`` on the same tokenization. Token-granularity HER needs full
alignments and uses ``jiwer.process_words`` per pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from rapidfuzz.distance import Levenshtein

from agwer.edits import EditCounts, classify_tokens
from agwer.transforms import apply_normalize, default_normalize

__all__ = ["AgenticOutput", "process_agentic", "oracle_select"]

_EPS = 1e-9


@dataclass
class AgenticOutput:
    """Everything agwer computes for one corpus."""

    n: int
    wer_1best: float
    wer_corrected: float
    wer_oracle: Optional[float]
    headroom: Optional[float]
    rir: Optional[float]
    her: Optional[float]
    edits: EditCounts
    wer_compositional: Optional[float] = None
    items: Optional[list] = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "wer_1best": self.wer_1best,
            "wer_corrected": self.wer_corrected,
            "wer_oracle": self.wer_oracle,
            "wer_compositional": self.wer_compositional,
            "headroom": self.headroom,
            "rir": self.rir,
            "her": self.her,
            "edits": self.edits.as_dict(),
        }


# --------------------------------------------------------------- primitives

def _tokens(strings: Sequence[str]) -> list[list[str]]:
    """jiwer-compatible tokenization: split on spaces, drop empty tokens."""
    return [[t for t in s.split(" ") if t] for s in strings]


def _pair_errors(refs_tok: list, hyps_tok: list) -> list[int]:
    """Word edit errors (S+D+I) for each (ref, hyp) pair.

    A plain loop over rapidfuzz's C-backed distance: profiling (see the
    efficiency review in ENGINEERING_PLAN 3.7) showed the edit distance is
    ~2% of evaluate() runtime, and a numpy/cpdist batch path was worth
    <=3 ms per 100k utterances -- not its complexity.
    """
    return [Levenshtein.distance(r, h) for r, h in zip(refs_tok, hyps_tok)]


def _corpus_wer(errors: Sequence[int], ref_lens: Sequence[int]) -> float:
    """Global error ratio; identical to jiwer's corpus WER."""
    total_ref = sum(ref_lens)
    total_err = sum(errors)
    if total_ref == 0:
        return 0.0 if total_err == 0 else float("inf")
    return total_err / total_ref


def _item_wer(errors: int, ref_len: int) -> float:
    # Empty-reference convention for per-item *display* values: report the raw
    # error count (jiwer's single-pair call would report 1.0). Corpus WER and
    # HER categories are unaffected; only this per-item number differs.
    return errors / ref_len if ref_len > 0 else float(errors)


def _oracle_pick(
    refs_tok: list, nbest_norm: list[list[str]]
) -> tuple[list[str], list[int]]:
    """Per-utterance minimum-error hypothesis; one flat batch call for all
    (ref, candidate) pairs so ragged n-best lists are fine."""
    flat_refs, flat_hyps = [], []
    for ref_tok, hyps in zip(refs_tok, nbest_norm):
        flat_refs.extend([ref_tok] * len(hyps))
        flat_hyps.append(hyps)
    flat_err = _pair_errors(flat_refs, _tokens([h for hs in flat_hyps for h in hs]))
    picked, picked_err, pos = [], [], 0
    for hyps in nbest_norm:
        errs = flat_err[pos:pos + len(hyps)]
        pos += len(hyps)
        best = min(range(len(hyps)), key=errs.__getitem__)  # first-min tie-break
        picked.append(hyps[best])
        picked_err.append(errs[best])
    return picked, picked_err


def _compositional_errors(refs_tok: list, nbest_norm: list) -> list[int]:
    """Per-utterance minimum errors when composing ANY sequence from the
    tokens occurring in the n-best list (HyPoradise o_cp): each reference
    token whose word appears in no hypothesis costs exactly one error
    (substitution/deletion); insertions never help, so covered tokens are
    free. Token inventory is type-level (unlimited reuse), which makes
    o_cp <= o_nb by construction."""
    errors = []
    for ref_tok, hyps in zip(refs_tok, nbest_norm):
        vocab = {tok for hyp in hyps for tok in hyp.split(" ") if tok}
        errors.append(sum(1 for tok in ref_tok if tok not in vocab))
    return errors


def oracle_select(
    references: Sequence[str],
    nbest: Sequence[Sequence[str]],
    normalize: Optional[Callable[[str], str]] = default_normalize,
) -> list[str]:
    """Per-utterance best hypothesis (fewest word errors) from each n-best list.

    Defines the reranking ceiling: no hypothesis selection can beat it.
    Returned hypotheses are in *normalized* form when a normalizer is given.
    """
    if len(references) != len(nbest):
        raise ValueError(
            f"references ({len(references)}) and nbest ({len(nbest)}) "
            "must have the same length"
        )
    if any(len(h) == 0 for h in nbest):
        raise ValueError("every n-best list must contain at least one hypothesis")
    refs_tok = _tokens(apply_normalize(references, normalize))
    nbest_norm = [apply_normalize(hyps, normalize) for hyps in nbest]
    picked, _ = _oracle_pick(refs_tok, nbest_norm)
    return picked


# ------------------------------------------------------------------- driver

def process_agentic(
    references: Sequence[str],
    corrected: Sequence[str],
    nbest: Optional[Sequence[Sequence[str]]] = None,
    onebest: Optional[Sequence[str]] = None,
    oracle: Optional[Sequence[str]] = None,
    normalize: Optional[Callable[[str], str]] = default_normalize,
    her_granularity: str = "utterance",
    return_items: bool = False,
) -> AgenticOutput:
    """Compute RIR, HER, and the underlying WERs for a corpus.

    Args:
        references: ground-truth transcripts.
        corrected: the corrector's outputs (the system under evaluation).
        nbest: per-utterance n-best hypothesis lists; ``nbest[i][0]`` must be
            the 1-best. When given, the 1-best and the oracle are derived
            from it.
        onebest: 1-best hypotheses; required when ``nbest`` is not given.
        oracle: precomputed oracle hypotheses (optional; otherwise selected
            from ``nbest``). Without ``nbest`` and ``oracle``, RIR is ``None``
            and only HER (+WERs) is computed.
        normalize: text normalizer applied to every string
            (default: paper-compatible :func:`agwer.default_normalize`;
            pass ``None`` to score raw strings).
        her_granularity: ``"utterance"`` (paper-reported values) or
            ``"token"`` (the paper's formal per-edit definition).
        return_items: attach per-utterance rows to ``.items``.
    """
    if her_granularity not in ("utterance", "token"):
        raise ValueError("her_granularity must be 'utterance' or 'token'")
    if nbest is None and onebest is None:
        raise ValueError("provide either nbest (with nbest[i][0] the 1-best) or onebest")
    n = len(references)
    if len(corrected) != n:
        raise ValueError(
            f"references ({n}) and corrected ({len(corrected)}) "
            "must have the same length"
        )
    if nbest is not None and len(nbest) != n:
        raise ValueError(f"references ({n}) and nbest ({len(nbest)}) must match")
    if onebest is not None and len(onebest) != n:
        raise ValueError(f"references ({n}) and onebest ({len(onebest)}) must match")
    if n == 0:
        raise ValueError("empty corpus")
    if nbest is not None and any(len(h) == 0 for h in nbest):
        raise ValueError("every n-best list must contain at least one hypothesis")

    refs_n = apply_normalize(references, normalize)
    corr_n = apply_normalize(corrected, normalize)
    ob_raw = onebest if onebest is not None else [h[0] for h in nbest]
    ob_n = apply_normalize(ob_raw, normalize)

    refs_tok = _tokens(refs_n)
    ref_lens = [len(t) for t in refs_tok]
    err_ob = _pair_errors(refs_tok, _tokens(ob_n))
    err_co = _pair_errors(refs_tok, _tokens(corr_n))
    wer_1best = _corpus_wer(err_ob, ref_lens)
    wer_corrected = _corpus_wer(err_co, ref_lens)

    wer_oracle = headroom = rir = wer_compositional = None
    if nbest is not None:
        nbest_norm = [apply_normalize(hyps, normalize) for hyps in nbest]
        wer_compositional = _corpus_wer(
            _compositional_errors(refs_tok, nbest_norm), ref_lens)
    if oracle is not None:
        oracle_n = apply_normalize(oracle, normalize)
        wer_oracle = _corpus_wer(_pair_errors(refs_tok, _tokens(oracle_n)), ref_lens)
    elif nbest is not None:
        _, picked_err = _oracle_pick(refs_tok, nbest_norm)
        wer_oracle = _corpus_wer(picked_err, ref_lens)
    if wer_oracle is not None:
        headroom = wer_1best - wer_oracle
        if abs(headroom) > _EPS:
            rir = (wer_1best - wer_corrected) / headroom

    counts = EditCounts(granularity=her_granularity)
    items = [] if return_items else None
    for i in range(n):
        if her_granularity == "utterance":
            # net effect per edited utterance; ref length is shared, so
            # comparing error counts is comparing WER.
            if ob_n[i] == corr_n[i]:
                cat = "no_edit"
            elif err_co[i] < err_ob[i]:
                cat = "helpful"
            elif err_co[i] > err_ob[i]:
                cat = "harmful"
            else:
                cat = "neutral"
            setattr(counts, cat, getattr(counts, cat) + 1)
            row = {
                "category": cat,
                "edited": ob_n[i] != corr_n[i],
                "wer_1best": _item_wer(err_ob[i], ref_lens[i]),
                "wer_corrected": _item_wer(err_co[i], ref_lens[i]),
            }
        else:
            row = classify_tokens(refs_n[i], ob_n[i], corr_n[i])
            counts.helpful += row["helpful"]
            counts.harmful += row["harmful"]
            counts.missed += row["missed"]
        if return_items:
            items.append({"index": i, "reference": refs_n[i], "onebest": ob_n[i],
                          "corrected": corr_n[i], **row})

    return AgenticOutput(
        n=n,
        wer_1best=wer_1best,
        wer_corrected=wer_corrected,
        wer_oracle=wer_oracle,
        headroom=headroom,
        rir=rir,
        her=counts.her,
        edits=counts,
        wer_compositional=wer_compositional,
        items=items,
    )
