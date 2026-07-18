"""Self-benchmark: ``python -m agwer.bench [--sizes 1000 10000]``.

Seeded synthetic ASR corpus (5-best), reports wall-clock for the full agentic
evaluation and the plain corpus-WER path on *this* machine, so published
performance claims are verifiable anywhere. Prints a fastwer comparison row
when fastwer is installed (it is not a dependency).

(Design borrowed from sglang's shipped ``bench_*`` modules: benchmarks live in
the package, not in the repo, so every install can measure itself.)
"""

from __future__ import annotations

import argparse
import platform
import random
import statistics
import time

import agwer


def make_corpus(n: int, seed: int = 0):
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(2000)]

    def sent():
        return [rng.choice(vocab) for _ in range(rng.randint(5, 20))]

    def corrupt(toks, p):
        out = []
        for t in toks:
            r = rng.random()
            if r < p * 0.6:
                out.append(rng.choice(vocab))
            elif r < p * 0.8:
                continue
            elif r < p:
                out.extend([t, rng.choice(vocab)])
            else:
                out.append(t)
        return out or [rng.choice(vocab)]

    refs, nbest, corrected = [], [], []
    for _ in range(n):
        r = sent()
        refs.append(" ".join(r))
        hyps = [" ".join(corrupt(r, 0.08))]
        hyps += [" ".join(corrupt(r, rng.uniform(0.08, 0.18))) for _ in range(4)]
        nbest.append(hyps)
        corrected.append(" ".join(corrupt(r, 0.05)))
    return refs, nbest, corrected


def median_time(fn, repeats: int = 5) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agwer.bench")
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print(f"agwer {agwer.__version__} | {platform.machine()} {platform.system()} "
          f"| python {platform.python_version()} | median of 5\n")
    try:
        import fastwer
    except ImportError:
        fastwer = None

    for n in args.sizes:
        refs, nbest, corrected = make_corpus(n, seed=args.seed)
        refs_n = [agwer.default_normalize(r) for r in refs]
        corr_n = [agwer.default_normalize(c) for c in corrected]

        t_eval = median_time(
            lambda r=refs, c=corrected, nb=nbest: agwer.evaluate(
                r, c, nbest=nb, workers=args.workers))
        t_tok = median_time(
            lambda r=refs, c=corrected, nb=nbest: agwer.evaluate(
                r, c, nbest=nb, her_granularity="token"))
        t_wer = median_time(
            lambda r=refs_n, c=corr_n: agwer.evaluate(
                r, c, onebest=c, normalize=None).wer_corrected)

        print(f"n={n:,} utterances x 5-best")
        print(f"  evaluate() [WERx3+oracle+RIR+HER]  {t_eval*1000:9.1f} ms")
        print(f"  evaluate(her_granularity='token')  {t_tok*1000:9.1f} ms")
        print(f"  corpus WER only (pre-normalized)   {t_wer*1000:9.1f} ms")
        if fastwer is not None:
            t_fw = median_time(lambda r=refs_n, c=corr_n: fastwer.score(c, r))
            print(f"  fastwer corpus WER (comparison)    {t_fw*1000:9.1f} ms")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
