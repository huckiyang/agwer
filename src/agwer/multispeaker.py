"""cpWER: multi-speaker word error rate (concatenated minimum-permutation).

The standard metric for speaker-attributed multi-speaker ASR (meetings,
conversations; the CHiME benchmark family). Each speaker's utterances are
concatenated into one stream, and the speaker permutation minimizing the
summed word-level edit distance between reference and hypothesis streams is
found (MeetEval, arXiv:2307.11394, Eq. 3):

    cpWER = min over speaker permutations of total errors / reference words

When the hypothesis over- or under-estimates the number of speakers, the
smaller side is padded with empty streams: an unmatched reference speaker
costs all its words as deletions, an unmatched hypothesis speaker all its
words as insertions.

Inputs are ``{speaker: text}`` dicts (text may also be a list of utterance
strings, concatenated in the given order) or plain lists (index = speaker).
Timestamps are not used; order utterances by start time upstream if needed.
Semantics agree with meeteval's ``cp_word_error_rate`` on the same inputs
(pinned by golden tests), except the empty-reference convention, which
follows :func:`agwer.wer` (0.0 when both sides are empty, ``inf`` when only
the reference is).

The assignment is solved exactly with the Jonker-Volgenant shortest
augmenting path algorithm (O(n^3)), so there is no speaker-count limit in
practice; a 20x20 cost matrix solves in microseconds.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from rapidfuzz.distance import Levenshtein

from agwer.align import tokenize

__all__ = ["cpwer", "cp_statistics"]

Streams = Union[Dict[object, Union[str, Sequence[str]]], Sequence[Union[str, Sequence[str]]]]


def _streams(x: Streams, normalize: Optional[Callable[[str], str]]) -> list:
    """[(speaker, tokens)] with utterances concatenated in input order."""
    items = list(x.items()) if isinstance(x, dict) else list(enumerate(x))
    norm = (lambda s: s) if normalize is None else normalize
    out = []
    for spk, val in items:
        utts = [val] if isinstance(val, str) else list(val)
        toks: list = []
        for u in utts:
            toks.extend(tokenize(norm(u)))
        out.append((spk, toks))
    return out


def _assign(cost: List[List[int]]) -> list:
    """Exact minimum-cost assignment on a square matrix: column per row
    (Jonker-Volgenant shortest augmenting path, O(n^3))."""
    n = len(cost)
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    match = [0] * (n + 1)   # match[j] = row assigned to column j (1-based)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        match[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = match[j0], INF, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
    out = [0] * n
    for j in range(1, n + 1):
        out[match[j] - 1] = j - 1
    return out


def _cp_solve(reference: Streams, hypothesis: Streams,
              normalize: Optional[Callable[[str], str]]) -> dict:
    refs = _streams(reference, normalize)
    hyps = _streams(hypothesis, normalize)
    n = max(len(refs), len(hyps))
    if n == 0:
        return {"errors": 0, "ref_words": 0, "assignment": [],
                "missed_speakers": 0, "falarm_speakers": 0,
                "scored_speakers": 0}
    # cost[i][j]: plain (unbanded) distance — off-diagonal pairs are
    # deliberately bad matches, the one regime where the band prior is wrong
    cost = [[0] * n for _ in range(n)]
    for i in range(n):
        rt = refs[i][1] if i < len(refs) else []
        for j in range(n):
            ht = hyps[j][1] if j < len(hyps) else []
            cost[i][j] = Levenshtein.distance(rt, ht)
    cols = _assign(cost)
    assignment = []
    for i, j in enumerate(cols):
        rk = refs[i][0] if i < len(refs) else None
        hk = hyps[j][0] if j < len(hyps) else None
        if rk is not None or hk is not None:
            assignment.append((rk, hk))
    return {
        "errors": sum(cost[i][j] for i, j in enumerate(cols)),
        "ref_words": sum(len(t) for _, t in refs),
        "assignment": assignment,
        "missed_speakers": max(0, len(refs) - len(hyps)),
        "falarm_speakers": max(0, len(hyps) - len(refs)),
        "scored_speakers": len(refs),
    }


def cpwer(reference: Streams, hypothesis: Streams,
          normalize: Optional[Callable[[str], str]] = None) -> float:
    """Concatenated minimum-permutation word error rate.

    ``reference`` and ``hypothesis`` are ``{speaker: text}`` dicts (text may
    be one string or a list of utterance strings) or plain lists of per-
    speaker texts. Returns min total errors / total reference words.
    """
    s = _cp_solve(reference, hypothesis, normalize)
    if s["ref_words"] == 0:
        return 0.0 if s["errors"] == 0 else float("inf")
    return s["errors"] / s["ref_words"]


def cp_statistics(reference: Streams, hypothesis: Streams,
                  normalize: Optional[Callable[[str], str]] = None) -> dict:
    """cpWER with its full accounting.

    Returns ``cpwer``, ``errors``, ``ref_words``, ``assignment`` (list of
    ``(ref_speaker, hyp_speaker)`` pairs; ``None`` marks an empty-stream
    pad on the unmatched side), and ``missed_speakers`` /
    ``falarm_speakers`` / ``scored_speakers`` counts.
    """
    s = _cp_solve(reference, hypothesis, normalize)
    if s["ref_words"] == 0:
        s["cpwer"] = 0.0 if s["errors"] == 0 else float("inf")
    else:
        s["cpwer"] = s["errors"] / s["ref_words"]
    return s
