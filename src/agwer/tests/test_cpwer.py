"""cpWER: golden-pinned to meeteval, assignment solver pinned to brute force.

The fixture was generated with meeteval's cp_word_error_rate (the reference
implementation, arXiv:2307.11394) on timestamp-free dict inputs, where both
implementations concatenate utterances in input order. Validation is on
errors/length, never on the assignment itself (ties are broken differently
by different solvers, legitimately).
"""

import itertools
import json
import pathlib
import random

import pytest
from rapidfuzz.distance import Levenshtein

import agwer
from agwer.multispeaker import _assign

FIXTURE = pathlib.Path(__file__).parent / "fixtures_cpwer_golden.json"


def test_meeteval_golden():
    cases = json.loads(FIXTURE.read_text())
    assert len(cases) >= 80
    for c in cases:
        s = agwer.cp_statistics(c["ref"], c["hyp"])
        assert s["errors"] == c["errors"], (c["ref"], c["hyp"])
        assert s["ref_words"] == c["length"]
        assert s["missed_speakers"] == c["missed_speaker"]
        assert s["falarm_speakers"] == c["falarm_speaker"]
        assert s["scored_speakers"] == c["scored_speaker"]
        if c["error_rate"] is not None:
            assert agwer.cpwer(c["ref"], c["hyp"]) == pytest.approx(c["error_rate"])


def test_assignment_solver_matches_brute_force():
    # the Hungarian solve must equal exhaustive permutation search
    rng = random.Random(5)
    for _ in range(200):
        n = rng.randint(1, 5)
        cost = [[rng.randint(0, 30) for _ in range(n)] for _ in range(n)]
        best = min(
            sum(cost[i][p[i]] for i in range(n))
            for p in itertools.permutations(range(n))
        )
        cols = _assign(cost)
        assert sorted(cols) == list(range(n))       # a real permutation
        assert sum(cost[i][cols[i]] for i in range(n)) == best


def test_hand_computed():
    # perfect with permuted speaker labels
    assert agwer.cpwer({"a": "x y z", "b": "u v"},
                       {"1": "u v", "2": "x y z"}) == 0.0
    # under-clustering: speaker b's 3 words are all deletions
    assert agwer.cpwer({"a": "x y z", "b": "u v w"}, {"1": "x y z"}) == 0.5
    # over-clustering: extra hypothesis speaker is all insertions
    assert agwer.cpwer({"a": "x y z"}, {"1": "x y z", "2": "u v"}) == pytest.approx(2 / 3)
    # greedy-vs-optimal trap: the optimal permutation is not the per-row argmin
    s = agwer.cp_statistics({"a": "p q r s", "b": "p q"},
                            {"1": "p q", "2": "p q r x"})
    both = min(
        Levenshtein.distance("p q r s".split(), "p q".split())
        + Levenshtein.distance("p q".split(), "p q r x".split()),
        Levenshtein.distance("p q r s".split(), "p q r x".split())
        + Levenshtein.distance("p q".split(), "p q".split()),
    )
    assert s["errors"] == both == 1


def test_input_forms_and_normalize():
    # lists index speakers; utterance lists concatenate in order
    assert agwer.cpwer(["x y", "u v"], ["u v", "x y"]) == 0.0
    assert agwer.cpwer({"a": ["x y", "z"]}, {"h": "x y z"}) == 0.0
    # normalize applied per utterance
    assert agwer.cpwer({"a": "Hello, World!"}, {"h": "hello world"},
                       normalize=agwer.default_normalize) == 0.0


def test_degenerate():
    assert agwer.cpwer({}, {}) == 0.0
    assert agwer.cpwer({"a": ""}, {"h": ""}) == 0.0
    assert agwer.cpwer({"a": ""}, {"h": "x y"}) == float("inf")
    s = agwer.cp_statistics({"a": "x"}, {})
    assert s["cpwer"] == 1.0 and s["missed_speakers"] == 1
    assert s["assignment"] == [("a", None)]
