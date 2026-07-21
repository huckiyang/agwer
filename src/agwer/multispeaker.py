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

__all__ = ["cpwer", "cp_statistics", "tcpwer", "tcp_statistics"]

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


# ------------------------------------------------------------------ tcpWER

def _char_intervals(words: list, start: float, end: float) -> list:
    """Per-word intervals proportional to character length (MeetEval
    'character_based')."""
    if len(words) == 1:
        return [(start, end)]
    total = sum(len(w) for w in words)
    per = (end - start) / total
    out, pos = [], 0
    for w in words:
        out.append((start + per * pos, start + per * (pos + len(w))))
        pos += len(w)
    return out


def _tc_streams(segments, normalize, points: bool, collar: float) -> dict:
    """{speaker: (tokens, intervals)}; segments sorted by start per speaker.

    Reference side (``points=False``): character-based word intervals.
    Hypothesis side (``points=True``): interval centers, expanded ±collar
    (MeetEval 'character_based_points' + collar).
    """
    norm = (lambda s: s) if normalize is None else normalize
    by_spk: dict = {}
    for seg in segments:
        try:
            spk, words = seg["speaker"], seg["words"]
            start, end = seg["start_time"], seg["end_time"]
        except KeyError as e:
            raise ValueError(f"segment missing key {e}") from e
        if end < start:
            raise ValueError(f"end_time < start_time in segment {seg!r}")
        by_spk.setdefault(spk, []).append((start, end, words))
    out = {}
    for spk, segs in by_spk.items():
        segs.sort(key=lambda s: s[0])          # stable: input order on ties
        toks: list = []
        ivs: list = []
        for start, end, words in segs:
            ws = tokenize(norm(words))
            if not ws:
                continue
            for w, (a, b) in zip(ws, _char_intervals(ws, start, end)):
                toks.append(w)
                if points:
                    c = (a + b) / 2
                    ivs.append((c - collar, c + collar))
                else:
                    ivs.append((a, b))
        out[spk] = (toks, ivs)
    return out


def _tc_full(ref, hyp, riv, hiv) -> int:
    """Naive O(n*m) time-constrained Levenshtein (fallback + fuzz reference).

    Substitution/match is allowed only when intervals strictly overlap."""
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        rs, re = riv[i - 1]
        rw = ref[i - 1]
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            hs, he = hiv[j - 1]
            best = min(prev[j], cur[j - 1]) + 1
            if hs < re and he > rs:
                d = prev[j - 1] + (rw != hyp[j - 1])
                if d < best:
                    best = d
            cur[j] = best
        prev = cur
    return prev[m]


def _tc_banded(ref, hyp, riv, hiv) -> int:
    """Time-windowed DP, exact for ANY input order: a column opens once its
    hypothesis word starts before the prefix-maximum reference end (it could
    have overlapped something seen), and closes once its running-maximum end
    is at or below the suffix-minimum of remaining reference starts (it can
    never overlap anything again). On time-sorted streams this reduces
    O(n*m) to O(n * words-per-collar-window); local disorder (overlapping
    speech, jitter) only widens the band. Closed columns are
    deletion-forwarded like meeteval's pruned kernel; exactness vs the
    naive DP is pinned by an unrestricted fuzz test."""
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        return n + m
    run_end = []
    e = -_INF
    for _, b in hiv:
        if b > e:
            e = b
        run_end.append(e)
    sufmin_hs = [0.0] * m                # min hypothesis start from j onward
    e = _INF
    for j in range(m - 1, -1, -1):
        if hiv[j][0] < e:
            e = hiv[j][0]
        sufmin_hs[j] = e
    sufmin_rs = [0.0] * n                # min reference start from i onward
    e = _INF
    for i in range(n - 1, -1, -1):
        if riv[i][0] < e:
            e = riv[i][0]
        sufmin_rs[i] = e
    D = list(range(m + 1))               # D[0][j] = j insertions
    lo = 0                               # closed hypothesis words
    act = 0                              # opened columns (D index)
    remax = -_INF                        # prefix-max reference end
    for i in range(1, n + 1):
        rs, re = riv[i - 1]
        rw = ref[i - 1]
        if re > remax:
            remax = re
        while act < m and sufmin_hs[act] < remax:  # open before closing:
            act += 1                         # every closed column was open
            D[act] = D[act - 1] + 1          # D[i-1][j]: insertion extension
        while lo < act and run_end[lo] <= sufmin_rs[i - 1]:
            lo += 1
        diag = D[lo]                     # D[i-1][lo]
        D[lo] = i if lo == 0 else D[lo] + 1   # boundary: deletion-forward
        left = D[lo]
        for j in range(lo + 1, act + 1):
            up = D[j]
            best = (up if up < left else left) + 1
            a, b = hiv[j - 1]
            if a < re and b > rs:
                d = diag + (rw != hyp[j - 1])
                if d < best:
                    best = d
            D[j] = best
            diag = up
            left = best
    return D[act] + (m - act)            # tail columns: insertions only


_INF = float("inf")


def _tc_distance(ref, hyp, riv, hiv) -> int:
    return _tc_banded(ref, hyp, riv, hiv)


def _tcp_solve(reference, hypothesis, collar, normalize) -> dict:
    refs = list(_tc_streams(reference, normalize, False, 0.0).items())
    hyps = list(_tc_streams(hypothesis, normalize, True, collar).items())
    n = max(len(refs), len(hyps))
    if n == 0:
        return {"errors": 0, "ref_words": 0, "assignment": [],
                "missed_speakers": 0, "falarm_speakers": 0,
                "scored_speakers": 0}
    cost = [[0] * n for _ in range(n)]
    for i in range(n):
        rt, riv = refs[i][1] if i < len(refs) else ([], [])
        for j in range(n):
            ht, hiv = hyps[j][1] if j < len(hyps) else ([], [])
            cost[i][j] = _tc_distance(rt, ht, riv, hiv)
    cols = _assign(cost)
    assignment = []
    for i, j in enumerate(cols):
        rk = refs[i][0] if i < len(refs) else None
        hk = hyps[j][0] if j < len(hyps) else None
        if rk is not None or hk is not None:
            assignment.append((rk, hk))
    return {
        "errors": sum(cost[i][j] for i, j in enumerate(cols)),
        "ref_words": sum(len(t) for _, (t, _) in refs),
        "assignment": assignment,
        "missed_speakers": max(0, len(refs) - len(hyps)),
        "falarm_speakers": max(0, len(hyps) - len(refs)),
        "scored_speakers": len(refs),
    }


def tcpwer(reference, hypothesis, collar: float = 5.0,
           normalize=None) -> float:
    """Time-constrained cpWER (MeetEval tcpWER, the CHiME-7/8 metric).

    Like :func:`cpwer`, but words may only match when their time intervals
    overlap: reference words get character-based intervals within their
    segment, hypothesis words get their interval center expanded by
    ``collar`` seconds (default 5.0, the recommended value). Inputs are
    lists of segments ``{"speaker", "words", "start_time", "end_time"}``.
    Semantics match meeteval's ``tcp_word_error_rate`` defaults (pinned by
    golden tests); the empty-reference convention follows :func:`agwer.wer`.
    """
    s = _tcp_solve(reference, hypothesis, collar, normalize)
    if s["ref_words"] == 0:
        return 0.0 if s["errors"] == 0 else float("inf")
    return s["errors"] / s["ref_words"]


def tcp_statistics(reference, hypothesis, collar: float = 5.0,
                   normalize=None) -> dict:
    """tcpWER with the full accounting (see :func:`cp_statistics`)."""
    s = _tcp_solve(reference, hypothesis, collar, normalize)
    if s["ref_words"] == 0:
        s["tcpwer"] = 0.0 if s["errors"] == 0 else float("inf")
    else:
        s["tcpwer"] = s["errors"] / s["ref_words"]
    return s
