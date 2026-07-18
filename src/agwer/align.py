"""Alignment core: one engine, every metric is a view over it.

agwer computes each metric from the same three primitives, so tokenization
and alignment semantics cannot drift between metrics:

* :func:`tokenize` / :func:`tokenize_all` — the single tokenization
  convention (split on spaces, drop empty tokens; jiwer-compatible).
* :func:`pair_errors` — batch edit errors (S+D+I) per pair, the fast path:
  RapidFuzz's C-level bit-parallel distance, no alignment materialized.
  Works on token lists (word level) and raw strings (character level).
* :func:`pooled_counts` — corpus-pooled (hits, substitutions, deletions,
  insertions) from the minimal alignments, for measures that need hits.
* :func:`equal_flags` — per-token correctness flags from the minimal
  alignment, for metrics that ask *which* tokens survived (HER edit
  classification, entity F1, hallucination attribution).

``pair_errors`` and the opcode-based primitives describe the same optimal
alignment, so counts agree exactly: ``distance == S + D + I`` always.
"""

from __future__ import annotations

from typing import Sequence

from rapidfuzz.distance import Levenshtein

__all__ = [
    "tokenize",
    "tokenize_all",
    "pair_errors",
    "pooled_counts",
    "equal_flags",
]


def tokenize(s: str) -> list:
    """Split on spaces, drop empty tokens (jiwer-compatible)."""
    return [t for t in s.split(" ") if t]


def tokenize_all(strings: Sequence[str]) -> list:
    return [[t for t in s.split(" ") if t] for s in strings]


# Auto-banding: above _BAND_MIN elements the distance runs banded
# (Ukkonen-style band via rapidfuzz score_hint, O(d*n/64) instead of
# O(n*m/64)). The hint only chooses the starting band; rapidfuzz doubles it
# until the true distance fits, so results are exact for ANY hint (pinned by
# tests down to rapidfuzz 3.6). The starting band is a length-difference
# floor plus an error-rate prior: 1/4 of the reference for word tokens, 1/8
# for character strings (character error rates run about half of word error
# rates). Measured: 2-4x on 16k+ word documents, ~10x on long-document CER,
# no change below the gate; a pathological corpus far above the prior
# (~90% error) pays up to ~1.3x for the doubling.
_BAND_MIN = 256


def _distance(r, h) -> int:
    n, m = len(r), len(h)
    if n <= _BAND_MIN:
        return Levenshtein.distance(r, h)
    hint = max(m - n if m > n else n - m,
               n >> (3 if isinstance(r, str) else 2))
    return Levenshtein.distance(r, h, score_hint=hint)


def pair_errors(refs, hyps) -> list:
    """Edit errors (S+D+I) for each pair; C-level distance, no alignment.

    Pass token lists for word-level errors or raw strings for
    character-level errors (spaces count as characters). Long inputs run
    banded automatically (see ``_BAND_MIN`` above); results are exact.
    """
    return [_distance(r, h) for r, h in zip(refs, hyps)]


def pooled_counts(refs: Sequence[str], hyps: Sequence[str],
                  chars: bool = False) -> tuple:
    """Corpus-pooled (hits, substitutions, deletions, insertions)."""
    hits = subs = dels = ins = 0
    for r, h in zip(refs, hyps):
        if not chars:
            r, h = tokenize(r), tokenize(h)
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


def equal_flags(ref_tok: list, hyp_tok: list) -> tuple:
    """(ref_ok, hyp_ok, n_insertions) from the minimal word alignment.

    ``ref_ok[i]`` / ``hyp_ok[j]`` are True iff that token lies in an
    ``equal`` span; ``n_insertions`` counts hypothesis tokens inserted
    against the reference.
    """
    ref_ok = [False] * len(ref_tok)
    hyp_ok = [False] * len(hyp_tok)
    n_ins = 0
    for op in Levenshtein.opcodes(ref_tok, hyp_tok):
        if op.tag == "equal":
            for i in range(op.src_start, op.src_end):
                ref_ok[i] = True
            for j in range(op.dest_start, op.dest_end):
                hyp_ok[j] = True
        elif op.tag == "insert":
            n_ins += op.dest_end - op.dest_start
    return ref_ok, hyp_ok, n_ins
