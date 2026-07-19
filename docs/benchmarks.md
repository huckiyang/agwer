# Benchmarks

All numbers on this page are measured, not estimated. The harness lives in
our development repository; the data is real: NVIDIA Canary 1-best decodes
of AMI meeting speech (5,469 segments, 22.2% corpus WER) from the Voice
Memory error analysis. Every engine receives identical pre-normalized
strings, WER agreement across engines is asserted before any timing, and we
report the minimum of 3 runs on an Apple Silicon M-series machine (14
cores). Versions: agwer 0.2.0, jiwer 4.0.0, fastwer 0.1.3.

## Long-context ASR: engine scaling with document length

Long-form ASR (meetings, podcasts, earnings calls) is scored at the document
level, and per-pair edit distance is quadratic in document length. That
quadratic term is where engines part ways. We concatenate consecutive real
AMI decodes into documents of a target length and hold the total workload
fixed at ~400k reference words per row, so any time growth comes purely from
document length:

| words per document | documents | fastwer (C++ DP) | jiwer | **agwer** |
|---|---|---|---|---|
| 25 | 11,966 | 57 ms | 176 ms | **54 ms** |
| 250 | 1,544 | 161 ms | 159 ms | **58 ms** |
| 1,000 | 397 | 502 ms | 166 ms | **78 ms** |
| 4,000 | 100 | 1,999 ms | 214 ms | **151 ms** |

Three observations:

1. **Plain dynamic programming does not survive long documents.** fastwer's
   classic O(n·m) DP grows ~35× as documents go from 25 to 4,000 words at
   constant total workload. The bit-parallel Myers algorithm underneath
   agwer and jiwer processes alignment columns 64 at a time, so it grows
   only ~3×. At 4,000-word documents agwer is **13× faster than fastwer**.
2. **agwer and jiwer share the same C++ engine (RapidFuzz), and agwer's
   thinner Python layer shows.** agwer calls the distance kernel directly on
   pre-tokenized batches; jiwer runs its transform pipeline per call. Same
   results (asserted bit-identical), roughly 2 to 3× less overhead.
3. **On very short utterances the engines converge.** At tiny corpora of
   short commands, fastwer's lean DP is competitive (we have measured it
   slightly ahead at the 1k-utterance scale). Long context is where the
   engine choice matters.

## Extreme case: one voice-coding dictation session (1k to 64k words)

Dictating to a coding agent produces the hardest scoring workload: one
continuous session-length document instead of many short utterances. This
benchmark scores a single seeded synthetic session (package names and
commands mixed with English glue, technical terms split into heard words,
~20% WER) as one document. Versions here: agwer 0.4.2, jiwer 4.0.0; WER
agreement between agwer and jiwer is asserted bit-identical before any
timing; median of 5 runs on the same M-series machine.

| session length | fastwer (C++ DP) | jiwer | agwer 0.4.1 | **agwer 0.4.2 (auto-banded)** | vs jiwer |
|---|---|---|---|---|---|
| 1,000 words | 1.3 ms | 0.34 ms | 0.11 ms | **0.09 ms** | 3.8× |
| 4,000 words | 21.3 ms | 1.91 ms | 0.83 ms | **0.46 ms** | 4.2× |
| 16,000 words | 498 ms | 15.6 ms | 9.9 ms | **3.5 ms** | 4.4× |
| 64,000 words | 9,163 ms | 188 ms | 153 ms | **36.9 ms** | 5.1× |

Character error rate on the same 64,000-word session (~345k characters):
**agwer 433 ms vs jiwer 4,151 ms (9.6×)**, values identical.

Three observations:

1. **Quadratic DP is disqualified at session length.** A long dictation
   session (64,000 words) costs fastwer 9.2 seconds per scoring call, 248×
   agwer; the bit-parallel engines stay far under 200 ms. If your
   evaluation loop rescores after every agent edit, that difference is the
   difference between interactive and not.
2. **The 0.4.2 auto-band is where the widening lead comes from.** Above
   256 elements, agwer starts the distance computation in an
   Ukkonen-style band sized by an error-rate prior (a quarter of the
   reference for words, an eighth for characters) instead of the full DP
   width; the band doubles until the true distance provably fits, so every
   result stays exact — pinned by tests from 0% to 100% error rates. Real
   transcripts sit far below the prior, so the band rarely doubles, and
   the work drops from O(n·m/64) to O(d·n/64). A pathological corpus
   (~90% error) pays about 1.3× over the unbanded path; typical ASR
   corpora gain 2–10×. Below the gate, short utterances take the exact
   same code path as before.
3. **Fleets of sessions scale across Apple Silicon cores.** Scoring 2,000
   such sessions (2M words) with `evaluate()`: 479 ms single worker,
   **140 ms with `workers=8`**, against 1,047 ms for jiwer on the same
   corpus — 7.5× end to end. The count-additive design merges chunk
   results exactly, so the parallel numbers are bit-identical to serial.

## Large-batch evaluation: many entries

The complementary axis to document length: entry count. The workload is
the real 30-session WSJ corrector batch (the same records as the
[example dataset](https://huggingface.co/datasets/huckiyang/agwer_asr_batch_test_v0))
replicated to size — short utterances, many entries, the LLM-output batch
regime. WER agreement across all engines and both agwer paths is asserted
before timing; agwer 0.4.7, median of 3, M-series.

| entries | fastwer | jiwer | agwer (strings) | **agwer (pre-tokenized)** | agwer `workers=8` |
|---|---|---|---|---|---|
| 10,020 | 12 ms | 56 ms | 16 ms | **3 ms** | 67 ms |
| 100,020 | 118 ms | 859 ms | 288 ms | **33 ms** | 118 ms |
| 1,000,020 | 1,194 ms | 8,690 ms | 2,844 ms | **335 ms** | 658 ms |

On short entries the per-call cost is tokenization, not edit distance:
materializing a million Python token lists dwarfs the C-level alignment
(fastwer avoids it by fusing tokenize+DP inside C++, which is why it
leads the plain string column at scale). agwer's answer, new in 0.4.7,
is to let you pay that cost **once**: every word-level measure accepts
pre-tokenized `list[list[str]]` input directly.

```python
import agwer

refs_tok = [agwer.tokenize(s) for s in refs]   # once per corpus
hyps_tok = [agwer.tokenize(s) for s in hyps]

agwer.wer(refs_tok, hyps_tok)   # identical value, no per-call tokenization
agwer.ser(refs_tok, hyps_tok)   # mer / wil / wip work the same way
```

Tokenize once and every subsequent scoring call runs 3.6 to 4x faster
than fastwer at every scale, single-threaded — the alignment engine was
never the bottleneck. This is exactly the shape of an agent evaluation
loop: the corpus is fixed, the outputs change, and rescoring should not
re-pay tokenization. (Normalize before tokenizing; the fast path
requires `normalize=None` so the two can never silently disagree.)

The number the comparison cannot show: the **full agentic evaluation**
(three WERs, both oracles, RIR, HER) over the same 1M entries × 5-best
runs in **5.6 s** with `workers=8`. No other engine computes those
quantities at all.

## Shared-reference DP reuse: 0.4.7 vs 0.4.8

In the agentic pipeline every quantity aligns against the same reference,
so 0.4.8 computes the n-best normalization, tokenization, and the flat
alignment pass once and derives the 1-best distances, oracle pick, and
compositional-oracle vocabulary from it — bit-identical outputs, pinned
by an A/B harness over every code path before release. Measured on the
published PyPI wheels, same machine, same replicated WSJ data (median of
5 at 10k, 3 at 100k):

| tier (10k / 100k entries) | 0.4.7 | 0.4.8 | change |
|---|---|---|---|
| single measure: corpus `wer()` | 17.6 / 298.0 ms | 16.0 / 299.1 ms | unchanged (not touched) |
| single measure, pre-tokenized | 3.2 / 32.3 ms | 3.1 / 33.1 ms | unchanged (not touched) |
| n-best measure: `oracle_wer()` | 300.7 / 3,687.7 ms | **248.4 / 3,104.8 ms** | 17% / 16% faster |
| full metrics: `evaluate()` | 298.4 / 3,610.3 ms | **248.6 / 3,095.6 ms** | 17% / 14% faster |
| full JSONL pipeline: `agwer file.jsonl` | 368.9 / 3,820.8 ms | **321.6 / 3,221.5 ms** | 13% / 16% faster |

The single-measure rows are the honest control: `wer()` never had the
duplication, so it does not move. Everything that touches an n-best list
gets the 14 to 17%, end to end through the CLI.

## agwer on CPU and Apple Silicon

The single-worker column is the portable CPU path: pure RapidFuzz, the same
code any x86 or ARM machine runs. On Apple Silicon, agwer additionally ships
native `arm64-darwin` wheels and scales across performance cores with
`evaluate(..., workers=N)`, since every aggregate it computes is
count-additive and merges exactly.

Long-form workload (1,000-word documents, ~2M reference words, full
`evaluate()`):

| setup | time | speedup |
|---|---|---|
| CPU path, single worker | 1.26 s | 1.0× |
| Apple Silicon, 4 workers | 0.39 s | 3.2× |
| Apple Silicon, 8 workers | **0.24 s** | **5.1×** |

Short-utterance workload (5-best agentic evaluation, from the main page):

| setup | time | speedup |
|---|---|---|
| 1M utterances, single worker | 27.9 s | 1.0× |
| 1M utterances, 8 workers | **5.2 s** | 5.4× |

Reproduce the in-package benchmark on your own machine:

```bash
python -m agwer.bench --workers 8
```

## The same numbers as charts

![Engine comparison: normal sentences and one 64,000-word agent dictation session](images/bench_engines.svg)

agwer is the fastest engine in both regimes, and the gap widens exactly
where agents live: long dictation sessions. On Apple Silicon the same
count-additive design then scales across performance cores with
`workers=N`, with results bit-identical to a single worker:

![CPU vs Apple Silicon: 2M-word session corpus and 1M-utterance agentic evaluation](images/bench_silicon.svg)
