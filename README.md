# agwer — agentic WER

**agwer** evaluates ASR error correctors — LLMs (or voice agents) that read
$n$-best speech-recognition hypotheses and decide **when to edit and when to
abstain**. Plain WER cannot see this act/abstain decision; agwer adds the two
measures introduced in the *Voice Memory* paper:

| measure | question it answers |
|---|---|
| **RIR** (ρ, Recoverable Information Ratio) | *How much of the 1-best→oracle gap did the correction close?* ρ=1 closes the gap; **ρ>1 beats the n-best oracle** (recovers tokens present in no hypothesis); **ρ<0 is the damage regime** (correcting was worse than doing nothing). |
| **HER** (Harmful Edit Rate) | *Of the edits the corrector actually made, what fraction broke a correct token?* Isolates **over-correction**, the dominant failure of unconstrained LLM correction. |

$$\rho=\frac{\mathrm{WER}_{1\text{-best}}-\mathrm{WER}_{\hat y}}{\mathrm{WER}_{1\text{-best}}-\mathrm{WER}_{\text{oracle}}}\qquad
\mathrm{HER}=\frac{\#\,\text{harmful edits}}{\#\,\text{harmful}+\#\,\text{helpful edits}}$$

Built on [jiwer](https://github.com/jitsi/jiwer) semantics with a batched
RapidFuzz (C++) hot path. On an M-series laptop (`python -m agwer.bench`):

| corpus (5-best) | full `evaluate()` (WER×3 + oracle + RIR + HER) | vs. per-utterance loop |
|---|---|---|
| 10k utterances | **0.21 s** | 3.9× faster |
| 50k utterances | **1.13 s** | 3.7× faster |

Reproduce on your machine: `python -m agwer.bench` (add fastwer/jiwer
comparisons with `benchmarks/bench_cpu.py`).

## Install

```bash
pip install agwer        # or: uv add agwer
```

## Quickstart

```python
import agwer

references = ["the cat sat", "hello world"]
nbest = [                       # nbest[i][0] is the ASR 1-best
    ["the cat sad", "the cat sat"],
    ["hello world", "hello word"],
]
corrected = ["the cat sat", "hello world"]   # your corrector's output

out = agwer.evaluate(references, corrected, nbest=nbest)
print(out.rir)            # 1.0   -> closed the n-best gap exactly
print(out.her)            # None  -> no harmful edits were made
print(out.wer_corrected)  # 0.0
print(out.edits.as_dict())
```

One-liners: `agwer.rir(...)` (alias `agwer.rho`), `agwer.her(...)`,
`agwer.oracle_wer(...)`, `agwer.oracle_hypotheses(...)`,
`agwer.compositional_oracle_wer(...)`.

With an n-best list, `evaluate()` reports **both HyPoradise oracles**
(§5.2 of [arXiv:2309.15701](https://arxiv.org/abs/2309.15701)):
`out.wer_oracle` — the n-best oracle *o_nb* (best single hypothesis, the
reranking bound) — and `out.wer_compositional` — the compositional oracle
*o_cp* (composing any sequence from tokens occurring in the list, the
correction bound; a reference token absent from every hypothesis costs one
error, so `o_cp <= o_nb` by construction). RIR is defined against *o_nb*,
matching the Voice Memory paper.

No n-best list available? Pass `onebest=` instead — HER and the WERs still
work; RIR needs the oracle (from `nbest=` or a precomputed `oracle=`).

### CLI

```bash
agwer results.jsonl                 # {"reference","corrected","nbest"} per line
agwer results.jsonl --json --her-granularity token
```

## The two HER granularities (read this once)

* `her_granularity="utterance"` (**default**) — every *edited* utterance is
  labeled by its net WER effect (helpful / harmful / neutral). This is the
  accounting behind the paper's reported values (e.g. 64% → 35% harmful on
  financial news).
* `her_granularity="token"` — the paper's formal per-edit account: both
  hypotheses are aligned to the reference and every changed token outcome is
  one edit (fixed-a-wrong-token = helpful, broke-a-correct-token = harmful;
  spurious insertions are harmful). Finer, and stricter about mixed edits.

A sentence where the corrector fixes one token and breaks another is *neutral*
at utterance granularity but *helpful=1, harmful=1* at token granularity —
choose the accounting that matches your claim, and report which one you used.

HER is `None` when the corrector made no consequential edits: a corrector that
always abstains has no harmful-edit rate. RIR is `None` when the n-best list
has no headroom (oracle = 1-best); there was nothing to recover.

## Normalization

Normalization is the main reason WER numbers are incomparable across papers,
so agwer ships the standards. Every normalizer is a plain
`Callable[[str], str]` passed as `normalize=`:

| normalizer | what | speed* |
|---|---|---|
| `agwer.default_normalize` (default) | conservative: lowercase, keep apostrophes, strip other punctuation (the Voice Memory paper / HyPoradise convention) | 639k utt/s |
| `BasicTextNormalizer()` | Whisper basic: symbols, brackets, optional diacritic folding (language-agnostic) | 205k utt/s |
| `EnglishTextNormalizer()` | **the Whisper English normalizer** — de-facto ASR-eval standard: spelled numbers→digits, currency, contractions, British→American | 29k utt/s (1.3× the original) |
| `EnglishTextNormalizer(cached=True)` | + LRU for recurring strings (agent dictation loops, gate re-scoring) | up to 16M utt/s on repeats |

\* M-series laptop, `benchmarks/bench_norm.py`; `normalize=None` scores raw strings.

```python
refs = ["it costs $22.50 today"]
hyps = ["it costs twenty two dollars and fifty cents today"]
agwer.evaluate(refs, hyps, onebest=hyps).wer_corrected                # > 0 (surface mismatch)
agwer.evaluate(refs, hyps, onebest=hyps,
               normalize=agwer.EnglishTextNormalizer()).wer_corrected  # 0.0
```

The Whisper normalizers are vendored (MIT, © 2022 OpenAI, attribution in
`agwer/normalizers/`) with **behavior pinned byte-identical to the original**
by golden tests — agwer only removed the third-party dependencies and
precompiled the patterns. Report which normalizer you used; it is part of the
metric.

## Reproducibility

`tests/test_paper_reference.py` pins agwer's outputs to golden values produced
by the paper's original evaluation code. The default settings
(`her_granularity="utterance"`, `normalize=agwer.default_normalize`) reproduce
the published numbers.

## Roadmap

Semantic (embedding-based) agentic metrics with hardware-accelerated inference
— Apple Silicon (MLX), CUDA GPU, and TPU backends — are planned.

## Citation

If you use RIR/ρ or HER, please cite the Voice Memory paper
(*Exploring Voice Memory for Agentic Speech Recognition*, under review, 2026 —
citation entry will be updated at camera-ready) and this package.

## License

Apache-2.0.
