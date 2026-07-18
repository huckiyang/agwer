# AgWER: Agent-oriented Word Error Rate

**agwer** is a simple and fast Python package to evaluate speech recognition
and voice agents. It is **self-contained**: one dependency
([RapidFuzz](https://github.com/rapidfuzz/RapidFuzz), C++ edit distance), a
40 KB wheel, sub-20 ms import.

It supports the classic ASR similarity measures and the agentic ones:

1. word error rate (WER) — also match error rate (MER), word information
   lost/preserved (WIL/WIP)
2. character error rate (CER)
3. Recoverable Information Ratio (RIR, the paper's ρ)
4. Harmful Edit Rate (HER)

The agentic measures (3–4) evaluate systems that read $n$-best hypotheses and
decide *when to edit and when to abstain* (LLM error correctors, dictation
agents):

| measure | question it answers |
|---|---|
| **RIR**: Recoverable Information Ratio (ρ) | *How much of the 1-best→oracle gap did the correction close?* ρ>1 beats the n-best oracle; ρ<0 is the damage regime. |
| **HER**: Harmful Edit Rate | *Of the edits actually made, what fraction broke a correct token?* Isolates over-correction. |
| **o_nb / o_cp** | The two HyPoradise oracles: best single hypothesis (reranking bound) / best token recombination (correction bound). |

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

The simplest use-case is computing the word error rate of a dictated
utterance:

```python
from agwer import wer

ref  = "please schedule the quarterly budget review for tuesday march twenty first at nine thirty and invite the design team"
asr_decoded = "please schedule the quarterly budget review for tuesday march twenty first at nine thirty and invite the desire team"

wer(ref, asr_decoded)   # 0.0526 one broken word in nineteen
```

All measures accept a single string or a list of strings; lists are pooled
corpus-level (total errors / total reference words), and `mer`, `wil`, `wip`,
`cer` work the same way:

```python
import agwer

refs = ["send the revised contract to the legal team before the board meeting on friday afternoon",
        "remind me to pick up the prescription from the pharmacy after the dentist appointment"]
hyps = ["send the revised contract to the legal team before the bored meeting on friday afternoon",
        "remind me to pick up the prescription from the pharmacy after the dentist appointment"]

agwer.wer(refs, hyps)     # 0.0345: corpus WER
agwer.cer(refs, hyps)     # 0.0116: corpus CER
```

### Evaluating a corrector / voice agent

With $n$-best input, one call computes everything. This is a **real Whisper
5-best decode** from the HyPoradise benchmark (WSJ, MIT-licensed): a 23-word
dictated stock quote where the ASR merges the spelled ticker and garbles the
fractions — *"i b m"* survives in **no** hypothesis:

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
out.wer_1best            # 0.3478 -> the raw ASR broke a third of the quote
out.wer_oracle           # 0.1739 -> o_nb: the best single hypothesis still has 4 errors
out.wer_compositional    # 0.1304 -> o_cp: no token recombination can spell "i b m"
out.wer_corrected        # 0.0
out.rir                  # 2.0    -> twice the n-best headroom: generative correction
out.her                  # 0.0    -> and nothing broken
```

Real decodes also show why reranking alone cannot save a voice agent: in
another HyPoradise utterance the command *"leaving **after noon**"* comes back
as *"leaving **afternoon**"* in **all five** hypotheses — the query's meaning
flips, every reranker is helpless, and only a corrector (and agwer's oracles)
can see it.

### Vibe-coding dictation: when the agent beats every hypothesis (ρ > 1)

Dictating to a coding agent is the hardest case: package names and code terms
are exactly what ASR mangles. Here the 1-best hears *"you v pip install ag
where"* and *"pie test"* — and the package name `agwer` appears in **no**
hypothesis, so no reranker and not even the compositional oracle can fully
recover the command. The coding agent can, because it knows the package from
context:

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
out.wer_1best            # 0.2308 -> the raw ASR broke almost a quarter of the command
out.wer_oracle           # 0.1154 -> the best single hypothesis still has 3 errors
out.wer_compositional    # 0.0385 -> even recombining all tokens cannot spell 'agwer'
out.wer_corrected        # 0.0
out.rir                  # 2.0    -> recovered TWICE the n-best headroom: generative
out.her                  # 0.0    -> and broke nothing
```

ρ > 1 is the signature of *generative* correction — the agent supplied truth
that exists nowhere in the hypothesis list. This is what plain WER, reranking
metrics, and even oracle bounds cannot see, and what RIR was built to measure.

HER comes in two granularities — `her_granularity="utterance"` (default;
the accounting behind the paper's reported values) and `"token"` (the formal
per-edit definition). A sentence where the corrector fixes one token and
breaks another is *neutral* at utterance granularity but *helpful=1,
harmful=1* at token granularity; report which one you used.

### Normalization

Normalization is the main reason WER numbers are incomparable across papers.
Every entry point takes `normalize=` (any `Callable[[str], str]`); agwer
ships the standards:

A dictation agent that says amounts out loud looks 87% wrong against the
written form — until you normalize:

```python
ref = "the invoice total came to $1,250.75 after the 15% discount was applied on march 3rd"
hyp = ("the invoice total came to one thousand two hundred fifty dollars "
       "and seventy five cents after the fifteen percent discount was applied on march third")

agwer.wer(ref, hyp)                                          # 0.867 (!)
agwer.wer(ref, hyp, normalize=agwer.EnglishTextNormalizer())  # 0.0
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
