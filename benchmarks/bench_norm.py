"""Normalization benchmark: agwer vendored normalizers vs original Whisper.

Two workloads:
  * unique   - every utterance distinct (batch evaluation)
  * agentic  - dictation/eval loop with recurring strings (~10% unique),
               where the cached=True LRU should shine.

(Equivalence with the original Whisper normalizers is pinned by the golden
tests in tests/test_normalizers.py; this benchmark measures agwer's own
normalizers.)

Run:  .venv/bin/python benchmarks/bench_norm.py [--n 20000]
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import agwer
from agwer.normalizers import BasicTextNormalizer, EnglishTextNormalizer

TEMPLATES = [
    "Mr. {n} paid twenty two dollars and fifty cents for the colour TV",
    "um, I think we need {n} per cent more by June first",
    "she finished in twenty second place with {n} points",
    "the programme cost one thousand euros in nineteen eighty {n}",
    "call me at five five five {n} about the aluminium order",
    "it's three point one four miles from the theatre [noise]",
    "he'd been waiting since {n} o'clock, uh, roughly",
    "one hundred and {n} people can't be wrong",
]


def make_utterances(n: int, unique_ratio: float, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    n_unique = max(1, int(n * unique_ratio))
    uniques = [
        rng.choice(TEMPLATES).format(n=rng.randint(1, 99)) + f" #{i}"
        for i in range(n_unique)
    ]
    return [rng.choice(uniques) for _ in range(n)]


def timeit(fn, repeats: int = 3) -> float:
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench_normalizer(name: str, norm, utts: list[str]) -> None:
    t = timeit(lambda: [norm(u) for u in utts])
    rate = len(utts) / t
    print(f"     {name:34s} {t * 1000:9.1f} ms   {rate:11,.0f} utt/s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()

    print(f"agwer {agwer.__version__} normalization benchmark "
          f"(n={args.n:,} utterances, median of 3)\n")

    for label, ratio in (("unique workload (batch eval)", 1.0),
                         ("agentic workload (10% unique strings)", 0.1)):
        utts = make_utterances(args.n, ratio)
        print(f"  == {label} ==")
        bench_normalizer("agwer.default_normalize", agwer.default_normalize, utts)
        bench_normalizer("BasicTextNormalizer", BasicTextNormalizer(), utts)
        bench_normalizer("EnglishTextNormalizer", EnglishTextNormalizer(), utts)
        bench_normalizer(
            "EnglishTextNormalizer(cached=True)", EnglishTextNormalizer(cached=True), utts
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
