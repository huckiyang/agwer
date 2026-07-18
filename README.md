# AgWER: Agent oriented Word Erorr Rate 

**agwer** is a simple and fast Python package to evaluate speech recognition
and voice agents. It is **self-contained**: one dependency
([RapidFuzz](https://github.com/rapidfuzz/RapidFuzz), C++ edit distance), a
40 KB wheel, sub-20 ms import.

It supports the classic ASR similarity measures:

1. word error rate (WER)
2. match error rate (MER)
3. word information lost (WIL)
4. word information preserved (WIP)
5. character error rate (CER)

plus the **agentic** measures for systems that read $n$-best hypotheses and
decide *when to edit and when to abstain* (LLM error correctors, dictation
agents), introduced in the *Voice Memory* paper:

| measure | question it answers |
|---|---|
| **RIR** (ρ) | *How much of the 1-best→oracle gap did the correction close?* ρ>1 beats the n-best oracle; ρ<0 is the damage regime. |
| **HER** | *Of the edits actually made, what fraction broke a correct token?* Isolates over-correction. |
| **o_nb / o_cp** | The two HyPoradise oracles: best single hypothesis (reranking bound) / best token recombination (correction bound). |

## Installation

```bash
pip install agwer        # or: uv add agwer
```

## Usage

The simplest use-case is computing the word error rate between two strings:

```python
from agwer import wer

error = wer("hello world", "hello duck")   # 0.5
```

All measures accept a single string or a list of strings; lists are pooled
corpus-level (total errors / total reference words), and `mer`, `wil`, `wip`,
`cer` work the same way:

```python
import agwer

refs = ["the cat sat", "hello world"]
hyps = ["the cat sad", "hello world"]
agwer.wer(refs, hyps)     # corpus WER
agwer.cer(refs, hyps)     # corpus CER
```

### Evaluating a corrector / voice agent

With $n$-best input, one call computes everything:

```python
nbest = [                       # nbest[i][0] is the ASR 1-best
    ["the cat sad", "the cat sat"],
    ["hello world", "hello word"],
]
corrected = ["the cat sat", "hello world"]   # your corrector's output

out = agwer.evaluate(refs, corrected, nbest=nbest)
out.wer_corrected        # 0.0
out.rir                  # 1.0  -> closed the n-best gap exactly
out.her                  # None -> no harmful edits (always-abstain has no HER)
out.wer_oracle           # o_nb: best single hypothesis
out.wer_compositional    # o_cp: best token recombination (o_cp <= o_nb)
```

HER comes in two granularities — `her_granularity="utterance"` (default;
the accounting behind the paper's reported values) and `"token"` (the formal
per-edit definition). A sentence where the corrector fixes one token and
breaks another is *neutral* at utterance granularity but *helpful=1,
harmful=1* at token granularity; report which one you used.

### Normalization

Normalization is the main reason WER numbers are incomparable across papers.
Every entry point takes `normalize=` (any `Callable[[str], str]`); agwer
ships the standards:

```python
agwer.wer("it costs $22.50", "it costs twenty two dollars and fifty cents",
          normalize=agwer.EnglishTextNormalizer())   # 0.0
```

| normalizer | what |
|---|---|
| `None` (general measures' default) | score strings exactly as given |
| `agwer.default_normalize` (agentic default) | conservative: lowercase, keep apostrophes, strip other punctuation |
| `BasicTextNormalizer()` | language-agnostic: symbols, brackets, optional diacritic folding |
| `EnglishTextNormalizer()` | the Whisper English normalizer (numbers→digits, currency, contractions, British→American); `cached=True` adds an LRU for agent loops |

The Whisper normalizers are vendored (MIT, © 2022 OpenAI, attribution
included) with behavior pinned **byte-identical** to the original by golden
tests. Report which normalizer you used; it is part of the metric.

### CLI

```bash
agwer results.jsonl                 # {"reference","corrected","nbest"} per line
agwer results.jsonl --json --her-granularity token
```

## Performance

Batched RapidFuzz hot path; benchmarks ship in the package:

```bash
python -m agwer.bench
```

~0.2 s for a full agentic evaluation (WER×3 + both oracles + RIR + HER) of a
10,000-utterance × 5-best corpus on a laptop; classic corpus WER in ~20 ms.

## Compatibility & reproducibility

Measure semantics match [jiwer](https://github.com/jitsi/jiwer) (validated
bit-identical on 600-corpus goldens, pinned in `tests/`), and the default
agentic settings reproduce the Voice Memory paper's published evaluation
(golden-pinned in `tests/test_paper_reference.py`).

## Citation

If you use RIR/ρ or HER, please cite the Voice Memory paper
(*Exploring Voice Memory for Agentic Speech Recognition*, under review, 2026 —
citation entry will be updated at camera-ready) and this package.

## License

Apache-2.0. Vendored Whisper normalizers: MIT (see
`src/agwer/normalizers/LICENSE_WHISPER`).
