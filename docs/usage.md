# Usage

## Installation

```bash
uv add agwer          # uv (recommended)
pip install agwer     # pip, Python >= 3.9
```

## Classic measures

`wer`, `mer`, `wil`, `wip`, `cer`, `ser` accept a string or a list of strings for
both arguments. Lists are pooled corpus-level (total errors / total reference
words), matching jiwer's semantics (validated bit-identical, pinned in tests).

```python
import agwer

reference  = "please schedule the quarterly budget review for tuesday march twenty first at nine thirty and invite the design team"
hypothesis = "please schedule the quarterly budget review for tuesday march twenty first at nine thirty and invite the desire team"

agwer.wer(reference, hypothesis)                # 0.0526 — one broken word in nineteen
agwer.wer(["a b", "c d"], ["a b", "c x"])       # 0.25 (pooled corpus-level)
agwer.cer(reference, hypothesis)                # character error rate
```

By default strings are scored exactly as given. Pass any
`Callable[[str], str]` as `normalize=`:

```python
agwer.wer("Hello, World!", "hello world", normalize=agwer.default_normalize)  # 0.0
```

For repeated batch scoring, tokenize once with `agwer.tokenize` and pass
pre-tokenized `list[list[str]]` batches — identical values, no per-call
tokenization (see the [benchmarks](benchmarks.md) for the numbers):

```python
refs_tok = [agwer.tokenize(s) for s in refs]
hyps_tok = [agwer.tokenize(s) for s in hyps]
agwer.wer(refs_tok, hyps_tok)
```

## Agentic evaluation

The agentic measures evaluate systems that read *n*-best hypotheses and
decide *when to edit and when to abstain*, such as LLM error correctors and
dictation agents:

| measure | question it answers |
|---|---|
| **RIR**: Recoverable Information Ratio (ρ) | *How much of the gap between the 1-best and the oracle did the correction close?* ρ>1 beats the n-best oracle; ρ<0 is the damage regime. |
| **HER**: Harmful Edit Rate | *Of the edits actually made, what fraction broke a correct token?* Isolates over-correction. |
| **o_nb / o_cp** | The two HyPoradise oracles: best single hypothesis (the reranking bound) and best token recombination (the correction bound). |

`evaluate()` takes references, the corrector's outputs, and the n-best lists
(`nbest[i][0]` must be the ASR 1-best):

```python
out = agwer.evaluate(references, corrected, nbest=nbest)
```

| field | meaning |
|---|---|
| `wer_1best`, `wer_corrected` | corpus WER before / after correction |
| `wer_oracle` | **o_nb** — best single hypothesis (reranking bound) |
| `wer_compositional` | **o_cp** — best token recombination (correction bound, ≤ o_nb) |
| `rir` | ρ = (WER₁ᵦₑₛₜ − WERᵧ) / (WER₁ᵦₑₛₜ − WERₒᵣₐ𝒸ₗₑ); `None` if no headroom |
| `her` | harmful / (harmful + helpful) edits; `None` if the corrector never edited |
| `edits` | full helpful / harmful / neutral / missed / no_edit decomposition |

`her_granularity="utterance"` (default) labels each edited utterance by its
net WER effect — the accounting behind the Voice Memory paper's reported
values. `"token"` is the formal per-edit account (each fixed token is
helpful, each broken token harmful, spurious insertions harmful).

`return_items=True` attaches per-utterance rows for error analysis.

## Normalizers

- `agwer.default_normalize` — conservative (lowercase, keep apostrophes,
  strip other punctuation); the agentic entry points' default.
- `agwer.BasicTextNormalizer()` — language-agnostic Whisper basic normalizer.
- `agwer.EnglishTextNormalizer()` — the Whisper English normalizer
  (numbers→digits, currency, contractions, British→American), vendored MIT
  with behavior golden-pinned byte-identical to the original.
  `cached=True` adds an LRU for agent/dictation loops with recurring strings.

!!! note "Report your normalizer"
    Normalization moves WER by whole points on the same data. It is part of
    the metric — always report which normalizer you used.

## Multi-speaker ASR: cpWER

`cpwer(reference, hypothesis)` scores speaker-attributed transcripts with
the concatenated minimum-permutation WER (the MeetEval definition,
arXiv:2307.11394): utterances are concatenated per speaker, and the speaker
permutation with the fewest word errors is solved exactly. Inputs are
`{speaker: text}` dicts (text may be a list of utterances) or plain lists.
No timestamps are needed; order utterances by start time upstream if
needed. `cp_statistics()` adds the full accounting: errors, reference
words, the speaker `assignment`, and missed / false-alarm / scored speaker
counts. Validated against meeteval on an 87-case golden fixture and 13 to
16 times faster on meeting-sized inputs; for time-constrained variants
(tcpWER) and ORC / MIMO, use [MeetEval](https://github.com/fgnt/meeteval).

## Entity F1 and Word Hallucination Rate

`entity_f1(reference, hypothesis, entities=... | predicate=...)` scores only
the information-carrying tokens (amounts, codes, names) with the same
alignment engine as WER, returning recall, precision, f1, and entity_wer.
`agwer.numeric_tokens` is a ready-made predicate for digits and spelled
numbers.

`word_hallucination_rate(references, outputs, hypotheses)` measures made-up
text with occurrence-bounded source attribution. It accepts a 1-best string
or an n-best list per utterance and decomposes into novel hallucinations
(invented words), repetition hallucinations (autoregressive loops over heard
words, the failure documented in arXiv:2408.16180), passed-through ASR
errors (not hallucination), and generative recoveries (correct words no
hypothesis contained).
