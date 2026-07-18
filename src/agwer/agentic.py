"""Agentic metrics: RIR (rho), HER, and the oracle bounds.

One call — :func:`process_agentic` (alias :func:`evaluate`) — computes
everything the Voice Memory paper's evaluation needs from (references,
n-best lists, corrected outputs):

* corpus WER of the 1-best, the n-best oracle, the compositional oracle,
  and the corrected output,
* the Recoverable Information Ratio (RIR / rho),
* the Harmful Edit Rate (HER) with its edit decomposition,
* optional per-utterance rows for error analysis.

Performance design: word-level edit distances come from the alignment core
(:mod:`agwer.align`), batched over the whole corpus, so the full agentic
evaluation costs a handful of batch calls rather than per-utterance Python
round-trips. Corpus WER is the global error ratio (total edit errors /
total reference words) and is bit-identical to jiwer on the same
tokenization.

Every quantity agwer aggregates is *count-additive*, so large corpora can be
split across processes and merged exactly: pass ``workers=N``. Chunks run in
separate Python processes (``spawn`` start method), which parallelizes the
normalization/tokenization-dominated pipeline across performance cores —
worthwhile from roughly 100k utterances up; below that the process startup
outweighs the gain.
"""

from __future__ import annotations

import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Callable, Optional, Sequence

from agwer.align import pair_errors, tokenize_all
from agwer.edits import EditCounts, classify_tokens
from agwer.text import apply_normalize, default_normalize

__all__ = [
    "AgenticOutput",
    "process_agentic",
    "evaluate",
    "oracle_select",
    "rir",
    "rho",
    "her",
    "oracle_wer",
    "oracle_hypotheses",
    "compositional_oracle_wer",
]

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

def _ratio(errors: int, ref_words: int) -> float:
    """Global error ratio; identical to jiwer's corpus WER."""
    if ref_words == 0:
        return 0.0 if errors == 0 else float("inf")
    return errors / ref_words


def _item_wer(errors: int, ref_len: int) -> float:
    # Empty-reference convention for per-item *display* values: report the raw
    # error count (jiwer's single-pair call would report 1.0). Corpus WER and
    # HER categories are unaffected; only this per-item number differs.
    return errors / ref_len if ref_len > 0 else float(errors)


def _oracle_pick(
    refs_tok: list, nbest_norm: list
) -> tuple:
    """Per-utterance minimum-error hypothesis; one flat batch call for all
    (ref, candidate) pairs so ragged n-best lists are fine."""
    flat_refs, flat_hyps = [], []
    for ref_tok, hyps in zip(refs_tok, nbest_norm):
        flat_refs.extend([ref_tok] * len(hyps))
        flat_hyps.append(hyps)
    flat_err = pair_errors(
        flat_refs, tokenize_all([h for hs in flat_hyps for h in hs])
    )
    picked, picked_err, pos = [], [], 0
    for hyps in nbest_norm:
        errs = flat_err[pos:pos + len(hyps)]
        pos += len(hyps)
        best = min(range(len(hyps)), key=errs.__getitem__)  # first-min tie-break
        picked.append(hyps[best])
        picked_err.append(errs[best])
    return picked, picked_err


def _compositional_errors(refs_tok: list, nbest_norm: list) -> list:
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
) -> list:
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
    refs_tok = tokenize_all(apply_normalize(references, normalize))
    nbest_norm = [apply_normalize(hyps, normalize) for hyps in nbest]
    picked, _ = _oracle_pick(refs_tok, nbest_norm)
    return picked


# --------------------------------------------------------- chunk computation

def _chunk_sums(args: tuple) -> dict:
    """Compute one chunk's additive ingredients (module-level: picklable).

    Everything returned is a sum or a count, so chunk results merge exactly.
    """
    (references, corrected, nbest, onebest, oracle, normalize,
     her_granularity, return_items, offset) = args

    n = len(references)
    refs_n = apply_normalize(references, normalize)
    corr_n = apply_normalize(corrected, normalize)
    ob_raw = onebest if onebest is not None else [h[0] for h in nbest]
    ob_n = apply_normalize(ob_raw, normalize)

    refs_tok = tokenize_all(refs_n)
    ref_lens = [len(t) for t in refs_tok]
    err_ob = pair_errors(refs_tok, tokenize_all(ob_n))
    err_co = pair_errors(refs_tok, tokenize_all(corr_n))

    err_or = err_cp = None
    if nbest is not None:
        nbest_norm = [apply_normalize(hyps, normalize) for hyps in nbest]
        err_cp = sum(_compositional_errors(refs_tok, nbest_norm))
    if oracle is not None:
        oracle_n = apply_normalize(oracle, normalize)
        err_or = sum(pair_errors(refs_tok, tokenize_all(oracle_n)))
    elif nbest is not None:
        _, picked_err = _oracle_pick(refs_tok, nbest_norm)
        err_or = sum(picked_err)

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
            items.append({"index": offset + i, "reference": refs_n[i],
                          "onebest": ob_n[i], "corrected": corr_n[i], **row})

    return {
        "n": n,
        "ref_words": sum(ref_lens),
        "err_ob": sum(err_ob),
        "err_co": sum(err_co),
        "err_or": err_or,
        "err_cp": err_cp,
        "counts": counts,
        "items": items,
    }


def _merge_counts(chunks: list, granularity: str) -> EditCounts:
    merged = EditCounts(granularity=granularity)
    for c in chunks:
        for f in ("helpful", "harmful", "neutral", "missed", "no_edit"):
            setattr(merged, f, getattr(merged, f) + getattr(c["counts"], f))
    return merged


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
    workers: int = 1,
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
        workers: number of processes. Results are identical for any value
            (count-additive merge); >1 pays process startup, so it wins from
            roughly 100k utterances up. The ``normalize`` callable must be
            picklable (module-level functions and all agwer normalizers are;
            lambdas are not). On macOS/Windows call from a
            ``if __name__ == "__main__"`` guard, as with any multiprocessing.
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
    if oracle is not None and len(oracle) != n:
        raise ValueError(f"references ({n}) and oracle ({len(oracle)}) must match")
    if n == 0:
        raise ValueError("empty corpus")
    if nbest is not None and any(len(h) == 0 for h in nbest):
        raise ValueError("every n-best list must contain at least one hypothesis")

    workers = max(1, min(int(workers), n))
    if workers == 1:
        chunks = [_chunk_sums((references, corrected, nbest, onebest, oracle,
                               normalize, her_granularity, return_items, 0))]
    else:
        size = math.ceil(n / workers)
        jobs = []
        for start in range(0, n, size):
            end = min(start + size, n)
            jobs.append((
                list(references[start:end]),
                list(corrected[start:end]),
                [list(h) for h in nbest[start:end]] if nbest is not None else None,
                list(onebest[start:end]) if onebest is not None else None,
                list(oracle[start:end]) if oracle is not None else None,
                normalize, her_granularity, return_items, start,
            ))
        try:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn")
            ) as pool:
                chunks = list(pool.map(_chunk_sums, jobs))
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            raise ValueError(
                "workers>1 requires a picklable normalize callable "
                "(module-level functions and agwer normalizers work; "
                "lambdas and closures do not)"
            ) from e

    ref_words = sum(c["ref_words"] for c in chunks)
    wer_1best = _ratio(sum(c["err_ob"] for c in chunks), ref_words)
    wer_corrected = _ratio(sum(c["err_co"] for c in chunks), ref_words)

    wer_oracle = headroom = rir_val = wer_compositional = None
    if chunks[0]["err_cp"] is not None:
        wer_compositional = _ratio(sum(c["err_cp"] for c in chunks), ref_words)
    if chunks[0]["err_or"] is not None:
        wer_oracle = _ratio(sum(c["err_or"] for c in chunks), ref_words)
        headroom = wer_1best - wer_oracle
        if abs(headroom) > _EPS:
            rir_val = (wer_1best - wer_corrected) / headroom

    counts = _merge_counts(chunks, her_granularity)
    items = None
    if return_items:
        items = [row for c in chunks for row in c["items"]]

    return AgenticOutput(
        n=n,
        wer_1best=wer_1best,
        wer_corrected=wer_corrected,
        wer_oracle=wer_oracle,
        headroom=headroom,
        rir=rir_val,
        her=counts.her,
        edits=counts,
        wer_compositional=wer_compositional,
        items=items,
    )


evaluate = process_agentic


# ----------------------------------------------------------- one-liners

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
