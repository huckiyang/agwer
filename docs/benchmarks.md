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
~20% WER) as one document. Versions here: agwer 0.4.1, jiwer 4.0.0; WER
agreement between agwer and jiwer is asserted bit-identical before any
timing; median of 5 runs on the same M-series machine.

| session length | fastwer (C++ DP) | jiwer | **agwer** | vs jiwer | vs fastwer |
|---|---|---|---|---|---|
| 1,000 words | 1.3 ms | 0.35 ms | **0.11 ms** | 3.2× | 12× |
| 4,000 words | 21.2 ms | 1.94 ms | **0.83 ms** | 2.3× | 26× |
| 16,000 words | 498 ms | 15.7 ms | **9.9 ms** | 1.6× | 50× |
| 64,000 words | 9,181 ms | 185 ms | **153 ms** | 1.2× | 60× |

Three observations:

1. **Quadratic DP is disqualified at session length.** A long dictation
   session (64,000 words) costs fastwer 9.2 seconds per scoring call, 60×
   agwer; the bit-parallel engines stay well under 200 ms. If your
   evaluation loop rescores after every agent edit, that difference is the
   difference between interactive and not.
2. **agwer's fast path is why it leads jiwer.** Since 0.4.1, `wer()`
   computes the C-level bit-parallel distance directly and never
   materializes an alignment; jiwer builds one per call. The lead is
   largest at interactive lengths (3.2× at 1,000 words) and converges as
   the shared C kernel dominates (1.2× at 64,000).
3. **Fleets of sessions scale across Apple Silicon cores.** Scoring 2,000
   such sessions (2M words) with `evaluate()`: 611 ms single worker,
   **155 ms with `workers=8`** (3.9×), against 1,049 ms for jiwer on the
   same corpus — 6.8× end to end. The count-additive design merges chunk
   results exactly, so the parallel numbers are bit-identical to serial.

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
