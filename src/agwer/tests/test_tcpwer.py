"""tcpWER: banded DP fuzzed against the naive DP, golden-pinned to meeteval."""

import json
import pathlib
import random

import pytest

import agwer
from agwer.multispeaker import _tc_banded, _tc_distance, _tc_full

FIXTURE = pathlib.Path(__file__).parent / "fixtures_tcpwer_golden.json"


def _rand_stream(rng, n, t0=0.0, vocab=20):
    toks, ivs, t = [], [], t0
    for _ in range(n):
        t += rng.uniform(0.0, 2.0)
        d = rng.uniform(0.1, 3.0)
        toks.append(f"w{rng.randrange(vocab)}")
        ivs.append((t, t + d))
        t += d * rng.uniform(0.0, 1.0)
    return toks, ivs


def test_distance_dispatcher_equals_naive_fuzz():
    # the public contract: _tc_distance == naive DP on ANY input, sorted or
    # not (the point-transform can produce non-monotone centers, which the
    # dispatcher must route to the full DP)
    rng = random.Random(13)
    banded_hits = 0
    for trial in range(600):
        rt, riv = _rand_stream(rng, rng.randrange(0, 30))
        ht, hiv = _rand_stream(rng, rng.randrange(0, 30),
                               t0=rng.uniform(-5, 5))
        if rng.random() < 0.4:      # point intervals with collar, like hyp side
            c = rng.uniform(0, 5)
            hiv = [((a + b) / 2 - c, (a + b) / 2 + c) for a, b in hiv]
        assert _tc_distance(rt, ht, riv, hiv) == _tc_full(rt, ht, riv, hiv), trial
        # and the banded kernel itself, whenever its precondition holds
        if (all(riv[i][0] <= riv[i + 1][0] for i in range(len(riv) - 1)) and
                all(hiv[i][0] <= hiv[i + 1][0] for i in range(len(hiv) - 1))):
            banded_hits += 1
            assert _tc_banded(rt, ht, riv, hiv) == _tc_full(rt, ht, riv, hiv), trial
    assert banded_hits > 200        # the banded path is actually exercised


def test_meeteval_golden():
    cases = json.loads(FIXTURE.read_text())
    assert len(cases) >= 50
    for c in cases:
        s = agwer.tcp_statistics(c["ref"], c["hyp"], collar=c["collar"])
        assert s["errors"] == c["errors"], c
        assert s["ref_words"] == c["length"]
        assert s["missed_speakers"] == c["missed_speaker"]
        assert s["falarm_speakers"] == c["falarm_speaker"]


def test_collar_semantics():
    ref = [{"speaker": "A", "words": "a b", "start_time": 0, "end_time": 2}]
    ok = [{"speaker": "1", "words": "a b", "start_time": 0, "end_time": 2}]
    far = [{"speaker": "1", "words": "a b", "start_time": 100, "end_time": 102}]
    near = [{"speaker": "1", "words": "a b", "start_time": 4, "end_time": 6}]
    assert agwer.tcpwer(ref, ok, collar=5) == 0.0
    # right words, wrong century: everything becomes deletions + insertions
    assert agwer.tcpwer(ref, far, collar=5) == 2.0
    # 4 s away: inside a 5 s collar, outside a 1 s collar
    assert agwer.tcpwer(ref, near, collar=5) == 0.0
    assert agwer.tcpwer(ref, near, collar=1) == 2.0


def test_speaker_permutation_and_counts():
    ref = [{"speaker": "alice", "words": "x y z", "start_time": 0, "end_time": 3},
           {"speaker": "bob", "words": "u v", "start_time": 3, "end_time": 5}]
    hyp = [{"speaker": "s0", "words": "u v", "start_time": 3, "end_time": 5},
           {"speaker": "s1", "words": "x y z", "start_time": 0, "end_time": 3}]
    s = agwer.tcp_statistics(ref, hyp, collar=5)
    assert s["tcpwer"] == 0.0
    assert sorted(s["assignment"]) == [("alice", "s1"), ("bob", "s0")]
    missing = agwer.tcp_statistics(ref, hyp[:1], collar=5)
    assert missing["missed_speakers"] == 1 and missing["errors"] == 3


def test_degenerate():
    assert agwer.tcpwer([], []) == 0.0
    ref = [{"speaker": "A", "words": "a", "start_time": 0, "end_time": 1}]
    assert agwer.tcpwer(ref, []) == 1.0
    assert agwer.tcpwer([], ref) == float("inf")
    with pytest.raises(ValueError, match="missing key"):
        agwer.tcpwer([{"speaker": "A", "words": "a"}], ref)
