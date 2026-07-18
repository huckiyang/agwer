# AgWER: Agent-oriented Word Error Rate

[![PyPI](https://img.shields.io/pypi/v/agwer)](https://pypi.org/project/agwer/)
[![ci](https://img.shields.io/github/actions/workflow/status/huckiyang/agwer/ci.yml?branch=main&label=ci)](https://github.com/huckiyang/agwer/actions/workflows/ci.yml)
[![Platforms](https://img.shields.io/badge/platforms-CPU%20%7C%20Apple%20Silicon%20native-blue)](#apple-silicon)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-huckiyang.github.io%2Fagwer-8A2BE2)](https://huckiyang.github.io/agwer/)

**agwer** is a simple and fast Python package to evaluate speech recognition
and voice agents. It is **self-contained**: one dependency
([RapidFuzz](https://github.com/rapidfuzz/RapidFuzz), C++ edit distance), a
40 KB wheel, and it imports in under 20 ms.

It supports the classic ASR similarity measures and the agentic ones:

1. word error rate (WER), plus match error rate (MER) and word information
   lost/preserved (WIL/WIP)
2. character error rate (CER)
3. Recoverable Information Ratio (RIR, the paper's ρ)
4. Harmful Edit Rate (HER)

The agentic measures (3 and 4) evaluate systems that read $n$-best hypotheses
and decide *when to edit and when to abstain*, such as LLM error correctors
and dictation agents:

| measure | question it answers |
|---|---|
| **RIR**: Recoverable Information Ratio (ρ) | *How much of the gap between the 1-best and the oracle did the correction close?* ρ>1 beats the n-best oracle; ρ<0 is the damage regime. |
| **HER**: Harmful Edit Rate | *Of the edits actually made, what fraction broke a correct token?* Isolates over-correction. |
| **o_nb / o_cp** | The two HyPoradise oracles: best single hypothesis (the reranking bound) and best token recombination (the correction bound). |

## Installation

With [uv](https://docs.astral.sh/uv/):

```bash
uv add agwer             # as a project dependency
uv pip install agwer     # into the active environment
```

Or with pip (Python >= 3.9):

```bash
pip install agwer
```

## Usage

The simplest use case is computing the word error rate of a dictated
utterance:

```python
from agwer import wer

ref  = "please schedule the quarterly budget review for tuesday march twenty first at nine thirty and invite the design team"
asr_decoded = "please schedule the quarterly budget review for tuesday march twenty first at nine thirty and invite the desire team"

wer(ref, asr_decoded)   # 0.0526, one broken word in nineteen
```

All measures accept a single string or a list of strings. Lists are pooled
corpus-level (total errors over total reference words), and `mer`, `wil`,
`wip`, `cer` work the same way:

```python
import agwer

refs = ["send the revised contract to the legal team before the board meeting on friday afternoon",
        "remind me to pick up the prescription from the pharmacy after the dentist appointment"]
hyps = ["send the revised contract to the legal team before the bored meeting on friday afternoon",
        "remind me to pick up the prescription from the pharmacy after the dentist appointment"]

agwer.wer(refs, hyps)     # 0.0345, corpus WER
agwer.cer(refs, hyps)     # 0.0116, corpus CER
```

### Evaluating a corrector or voice agent

With $n$-best input, one call computes everything. This is a **real Whisper
5-best decode** from the MIT-licensed HyPoradise benchmark (WSJ): a 23-word
dictated stock quote where the ASR merges the spelled ticker and garbles the
fractions. The spelled form *"i b m"* survives in **no** hypothesis:

```python
ref = "i b m fell one and seven eighths to one hundred twenty and three eighths on more than two point five million shares"

nbest = [[   # real Whisper 5-best; nbest[i][0] is the 1-best
    "ibm fell one seven eight to one hundred and twenty three eight on more than two point five million shares",
    "ibm fell one point seven eight to one hundred and twenty point three eight on more than two point five million shares",
    "ibm fell one and seven eighths to one hundred and twenty and three eighths on more than two point five million shares",
    "ibm fell one point seven eights to one hundred and twenty point three eights on more than two point five million shares",
    "ibm fell one seven eighths to one hundred and twenty three eighths on more than two point five million shares",
]]
corrected = [ref]   # a corrector that resolves the ticker and fractions from context

out = agwer.evaluate([ref], corrected, nbest=nbest)
out.wer_1best            # 0.3478  the raw ASR broke a third of the quote
out.wer_oracle           # 0.1739  o_nb: the best single hypothesis still has 4 errors
out.wer_compositional    # 0.1304  o_cp: no token recombination can spell "i b m"
out.wer_corrected        # 0.0
out.rir                  # 2.0     twice the n-best headroom: generative correction
out.her                  # 0.0     and nothing broken
```

Real decodes also show why reranking alone cannot save a voice agent. In
another HyPoradise utterance, the command *"leaving **after noon**"* comes
back as *"leaving **afternoon**"* in **all five** hypotheses. The query's
meaning flips, every reranker is helpless, and only a corrector (and agwer's
oracles) can see it.

### Vibe-coding dictation: when the agent beats every hypothesis (ρ > 1)

Dictating to a coding agent is the hardest case, because package names and
code terms are exactly what ASR mangles. Here the 1-best hears *"you v pip
install ag where"* and *"pie test"*, and the package name `agwer` appears in
**no** hypothesis. No reranker, and not even the compositional oracle, can
fully recover the command. The coding agent can, because it knows the package
from context:

```python
ref = ("open a terminal run uv pip install agwer then write a pytest that checks "
       "the word error rate of the two transcripts stays below five percent")

nbest = [[  # 26-word dictation; ASR breaks the technical terms
    "open a terminal run you v pip install ag where then write a pie test that checks "
    "the word error rate of the two transcripts stays below five percent",
    "open a terminal run uv pip install a g wear then write a pytest that checks "
    "the word error rate of the two transcripts stays below five percent",
    "open a terminal run you've pip installed ag where then write a pie test that checks "
    "the word error rate of the two transcript stays below five percent",
]]
corrected = [ref]   # the agent reconstructs 'uv pip install agwer' and 'pytest'

out = agwer.evaluate([ref], corrected, nbest=nbest)
out.wer_1best            # 0.2308  the raw ASR broke almost a quarter of the command
out.wer_oracle           # 0.1154  the best single hypothesis still has 3 errors
out.wer_compositional    # 0.0385  even recombining all tokens cannot spell 'agwer'
out.wer_corrected        # 0.0
out.rir                  # 2.0     recovered twice the n-best headroom
out.her                  # 0.0     and broke nothing
```

ρ > 1 is the signature of *generative* correction: the agent supplied truth
that exists nowhere in the hypothesis list. Plain WER, reranking metrics, and
even oracle bounds cannot see this. RIR was built to measure it.

HER comes in two granularities. `her_granularity="utterance"` (the default)
is the accounting behind the paper's reported values, and `"token"` follows
the formal per-edit definition. A sentence where the corrector fixes one
token and breaks another is *neutral* at utterance granularity but
*helpful=1, harmful=1* at token granularity, so report which one you used.

### Normalization

Normalization is the main reason WER numbers are incomparable across papers.
Every entry point takes `normalize=` (any `Callable[[str], str]`), and agwer
ships the standards.

A dictation agent that says amounts out loud looks 87% wrong against the
written form, until you normalize:

```python
ref = "the invoice total came to $1,250.75 after the 15% discount was applied on march 3rd"
hyp = ("the invoice total came to one thousand two hundred fifty dollars "
       "and seventy five cents after the fifteen percent discount was applied on march third")

agwer.wer(ref, hyp)                                          # 0.867 (!)
agwer.wer(ref, hyp, normalize=agwer.EnglishTextNormalizer())  # 0.0
```

| normalizer | what it does |
|---|---|
| `None` (the general measures' default) | scores strings exactly as given |
| `agwer.default_normalize` (the agentic default) | conservative: lowercase, keep apostrophes, strip other punctuation |
| `BasicTextNormalizer()` | language-agnostic: symbols, brackets, optional diacritic folding |
| `EnglishTextNormalizer()` | the Whisper English normalizer: spelled numbers to digits, currency, contractions, British to American spelling; `cached=True` adds an LRU for agent loops |

The Whisper normalizers are vendored (MIT, © 2022 OpenAI, attribution
included) with behavior pinned **byte-identical** to the original by golden
tests. Report which normalizer you used; it is part of the metric.

### CLI

```bash
agwer results.jsonl                 # {"reference","corrected","nbest"} per line
agwer results.jsonl --json --her-granularity token
```

## Performance

The hot path is batched RapidFuzz, and every aggregate agwer computes is
**count-additive**, so large corpora parallelize *exactly*. Results are
identical for any worker count: `evaluate(..., workers=8)`.

Typical performance on Apple Silicon (M-series, 14 cores) for the full
agentic evaluation (three WERs, both oracles, RIR, and HER) of 5-best
corpora:

| scenario | time |
|---|---|
| 10k utterances (single-threaded) | ~0.2 s |
| 100k utterances (single-threaded) | 2.6 s |
| 100k utterances (8 workers) | **0.48 s** (5.5×) |
| 1M utterances (single-threaded) | 27.9 s |
| 1M utterances (8 workers) | **5.2 s** (5.4×) |

Workers pay process startup, so they win from roughly 100k utterances up. On
macOS and Windows, call from a `if __name__ == "__main__"` guard, as with any
multiprocessing. Reproduce on your machine:

```bash
python -m agwer.bench --workers 8
```

### Apple Silicon

[![Speed](https://img.shields.io/badge/1M%20utterances-5.2s%20on%20M--series-brightgreen)](#performance)

agwer is **native on Apple Silicon out of the box**, with no separate
install: pip and uv select the arm64 wheel automatically, and RapidFuzz
ships compiled `arm64-darwin` extensions, so the C++ edit-distance core runs
natively on M-series machines. `workers=` then scales the whole pipeline
across performance cores (see the table above). Planned next is an optional
`agwer[mlx]` extra for embedding-based *semantic* metrics on the Apple GPU
via [MLX](https://github.com/ml-explore/mlx). Semantic inference is the one
place extra hardware genuinely helps; edit distance does not need it.

## Compatibility & reproducibility

Measure semantics match [jiwer](https://github.com/jitsi/jiwer), validated
bit-identical on a 600-corpus golden set pinned in `tests/`.

agwer gratefully builds on three projects:

* [jiwer](https://github.com/jitsi/jiwer) defined the measure semantics and
  the easy, friendly API style this package follows.
* [OpenAI Whisper](https://github.com/openai/whisper) created the English
  text normalizer that became the community standard for ASR evaluation. We
  vendor it faithfully under its MIT license, with attribution.
* [NVIDIA NeMo text processing](https://github.com/NVIDIA/NeMo-text-processing)
  sets the bar for full text normalization across languages. Its semiotic
  class taxonomy guides our normalization roadmap.

## License

Apache-2.0. Vendored Whisper normalizers: MIT (see
`src/agwer/normalizers/LICENSE_WHISPER`).
