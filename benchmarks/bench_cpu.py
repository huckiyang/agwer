"""CPU benchmark: agwer vs jiwer vs fastwer.

Two questions, kept separate on purpose:

A. Engine speed - corpus WER on identical, pre-normalized strings
   (jiwer/RapidFuzz vs fastwer C++ DP; agwer rides the jiwer engine and adds
   its normalization pass).

B. Agentic-metrics speed - the full paper evaluation (WER 1-best + corrected,
   n-best oracle, RIR, HER) via agwer.evaluate() vs the naive per-item loop of
   the paper's original reference implementation. fastwer/jiwer have no
   counterpart here (marked n/a) - this is what agwer exists for.

Run:  .venv/bin/python benchmarks/bench_cpu.py [--sizes 1000 10000 50000]
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import jiwer

import agwer

try:
    import fastwer
except ImportError:  # fastwer is an optional comparison, not a dependency
    fastwer = None

REPEATS = 5
NAIVE_TIME_GUARD_S = 120.0  # skip naive baseline above this projected cost


# ---------------------------------------------------------------- corpus gen
def make_corpus(n: int, seed: int = 0):
    """Seeded ASR-like corpus: refs, 5-best (nbest[0]=1-best), corrected."""
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(2000)]

    def sent():
        return [rng.choice(vocab) for _ in range(rng.randint(5, 20))]

    def corrupt(toks, p):
        out = []
        for t in toks:
            r = rng.random()
            if r < p * 0.6:
                out.append(rng.choice(vocab))          # substitution
            elif r < p * 0.8:
                continue                                # deletion
            elif r < p:
                out.extend([t, rng.choice(vocab)])      # insertion
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


# ---------------------------------------------------------------- timing
def timeit(fn, repeats: int = REPEATS):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ------------------------------------------------- naive reference baseline
def naive_reference_eval(refs, nbest, corrected):
    """The paper's original evaluation pattern: per-item jiwer calls
    (skillopt_asr metrics.oracle_hypotheses + harmful_edit.corpus_her)."""
    onebest = [h[0] for h in nbest]
    # corpus WERs (this part was already corpus-level in the reference code)
    wer_1best = jiwer.wer(refs, onebest)
    wer_corr = jiwer.wer(refs, corrected)
    # oracle: per-item, per-hypothesis jiwer.wer
    oracle = []
    for ref, hyps in zip(refs, nbest):
        oracle.append(min(hyps, key=lambda h: jiwer.wer(ref, h)))
    wer_oracle = jiwer.wer(refs, oracle)
    rho = (wer_1best - wer_corr) / (wer_1best - wer_oracle)
    # HER: per-item wer pair + per-item process_words (token accuracies),
    # exactly as harmful_edit.classify_item did
    cats = {"helpful": 0, "harmful": 0, "neutral": 0, "no_edit": 0}
    for ref, ob, co in zip(refs, onebest, corrected):
        w_ob, w_co = jiwer.wer(ref, ob), jiwer.wer(ref, co)
        jiwer.process_words([ref], [ob])
        jiwer.process_words([ref], [co])
        if ob == co:
            cats["no_edit"] += 1
        elif w_co < w_ob:
            cats["helpful"] += 1
        elif w_co > w_ob:
            cats["harmful"] += 1
        else:
            cats["neutral"] += 1
    her = cats["harmful"] / (cats["helpful"] + cats["harmful"])
    return rho, her


# ---------------------------------------------------------------- benchmark
def bench(sizes):
    print(f"platform: {get_platform()}")
    print(f"versions: agwer {agwer.__version__} | jiwer {getattr(jiwer, '__version__', '?')} "
          f"| fastwer {'yes' if fastwer else 'not installed'} | repeats={REPEATS} (median)\n")

    for n in sizes:
        refs, nbest, corrected = make_corpus(n)
        # pre-normalized copies so section A compares engines on identical input
        refs_n = [agwer.default_normalize(r) for r in refs]
        corr_n = [agwer.default_normalize(c) for c in corrected]

        # correctness agreement (identical inputs)
        w_j = jiwer.wer(refs_n, corr_n)
        w_a = agwer.evaluate(refs_n, corr_n, onebest=corr_n, normalize=None).wer_corrected
        assert abs(w_j - w_a) < 1e-12
        agree = "yes"
        if fastwer is not None:
            w_f = fastwer.score(corr_n, refs_n) / 100.0
            agree = "yes" if abs(w_j - w_f) < 1e-4 else f"NO ({w_j:.6f} vs {w_f:.6f})"

        t_jiwer = timeit(lambda r=refs_n, c=corr_n: jiwer.wer(r, c))
        t_fast = (timeit(lambda r=refs_n, c=corr_n: fastwer.score(c, r))
                  if fastwer is not None else None)
        # agwer's actual internal WER path (batched rapidfuzz) on the same input
        t_agwer_wer = timeit(lambda r=refs_n, c=corr_n: agwer.evaluate(
            r, c, onebest=c, normalize=None).wer_corrected)

        t_evaluate = timeit(
            lambda r=refs, c=corrected, nb=nbest: agwer.evaluate(r, c, nbest=nb))
        t_evaluate_tok = timeit(
            lambda r=refs, c=corrected, nb=nbest: agwer.evaluate(
                r, c, nbest=nb, her_granularity="token"))

        # naive baseline with a projected-cost guard
        t_probe0 = time.perf_counter()
        naive_reference_eval(refs[:200], [h for h in nbest[:200]], corrected[:200])
        probe = time.perf_counter() - t_probe0
        projected = probe * n / 200
        if projected < NAIVE_TIME_GUARD_S:
            t_naive = timeit(
                lambda r=refs, nb=nbest, c=corrected: naive_reference_eval(r, nb, c),
                repeats=3)
            naive_cell = f"{t_naive:8.2f}s"
            speedup = f"{t_naive / t_evaluate:5.1f}x"
        else:
            naive_cell = f"~{projected:7.0f}s (skipped)"
            speedup = f"~{projected / t_evaluate:4.0f}x"

        print(f"== n = {n:,} utterances (5-best) | engines agree on WER: {agree} ==")
        print("  A. corpus WER (pre-normalized input)")
        print(f"     jiwer (RapidFuzz)          {t_jiwer*1000:9.1f} ms")
        if t_fast is not None:
            print(f"     fastwer (C++ DP)           {t_fast*1000:9.1f} ms")
        print(f"     agwer (batched rapidfuzz)  {t_agwer_wer*1000:9.1f} ms")
        print("  B. full agentic eval (WERx3 + oracle + RIR + HER)")
        print(f"     agwer.evaluate()           {t_evaluate*1000:9.1f} ms")
        print(f"     agwer.evaluate(token HER)  {t_evaluate_tok*1000:9.1f} ms")
        print(f"     naive per-item (paper ref) {naive_cell}   -> agwer speedup {speedup}")
        print("     jiwer / fastwer                 n/a (no RIR/HER support)")
        print()


def get_platform():
    import platform
    return f"{platform.machine()} | {platform.system()} | python {platform.python_version()}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 50000])
    args = ap.parse_args()
    bench(args.sizes)
