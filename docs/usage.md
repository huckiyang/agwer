# Usage

## Classic measures

`wer`, `mer`, `wil`, `wip`, `cer` accept a string or a list of strings for
both arguments. Lists are pooled corpus-level (total errors / total reference
words), matching jiwer's semantics (validated bit-identical, pinned in tests).

```python
import agwer

agwer.wer("hello world", "hello duck")          # 0.5
agwer.wer(["a b", "c d"], ["a b", "c x"])       # 0.25 (pooled)
agwer.cer("hello", "hallo")                     # 0.2
```

By default strings are scored exactly as given. Pass any
`Callable[[str], str]` as `normalize=`:

```python
agwer.wer("Hello, World!", "hello world", normalize=agwer.default_normalize)  # 0.0
```

## Agentic evaluation

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
