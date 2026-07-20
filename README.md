# AgWER: Agent-oriented Word Error Rate

[![PyPI](https://img.shields.io/pypi/v/agwer)](https://pypi.org/project/agwer/)
[![ci](https://img.shields.io/github/actions/workflow/status/huckiyang/agwer/ci.yml?branch=main&label=ci)](https://github.com/huckiyang/agwer/actions/workflows/ci.yml)
[![Platforms](https://img.shields.io/badge/platforms-CPU%20%7C%20Apple%20Silicon%20native-blue)](#apple-silicon)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-huckiyang.github.io%2Fagwer-8A2BE2)](https://huckiyang.github.io/agwer/)

**agwer** is a simple and efficient Python package to evaluate speech recognition
and voice agents. A 40 KB wheel that imports in under 20 ms, fast on CPU and
native on Apple Silicon M-series. 

- note: this PyPI package was developed organically by a human programmer since 2025. We work together with the great Fable. 🙂
  - this sentence is 100% human progammer. We welcome contributions from the community.

agwer supports the classic ASR similarity measures and the agentic ones:

1. word error rate (WER) and sentence error rate (SER)
2. character error rate (CER)
3. recoverable information ratio (RIR)
4. harmful edit rate (HER)
5. named entity F1 score (NF1)
6. word hallucination rate (WHR)
7. concatenated minimum-permutation word error rate (cpWER) for multi-speaker ASR

The agentic measures (3 and 4) evaluate systems that read *n*-best hypotheses
and decide *when to edit and when to abstain*, such as LLM error correctors
and dictation agents. The [metric guide](https://huckiyang.github.io/agwer/usage/)
explains what question each measure answers, and the
[example session](https://huckiyang.github.io/agwer/examples/) walks through a
complete evaluation from install to CLI.

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

The test suite ships inside the package, so any install can verify itself:

```bash
pip install "agwer[test]"
python -m pytest --pyargs agwer     # 116 tests, a few seconds
```

## Usage

The simplest use case is computing the word error rate of a dictated
utterance:

```python
import agwer

ref  = "please schedule the quarterly budget review for tuesday march twenty first at nine thirty and invite the design team"
asr_decoded = "please schedule the quarterly budget review for tuesday march twenty first at nine thirty and invite the desire team"

agwer.wer(ref, asr_decoded)   # 0.0526, one broken word in nineteen
```

All measures accept a single string or a list of strings. Lists are pooled
corpus-level (total errors over total reference words), and `mer`, `wil`,
`wip`, `cer`, `ser` work the same way:

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

With *n*-best input, one call computes everything. Real data end to end:
three **real Whisper 5-best decodes** from the MIT-licensed HyPoradise
benchmark (WSJ), each with the **verbatim output of a real LLM corrector**.
The three utterances show the three things a corrector can do: restore a
word from another hypothesis, fix a reading that every hypothesis got wrong,
and break a name that was already right.

```python
refs = [
    "quote obviously we were disappointed we did not get a larger award",
    "he retired as a partner in nineteen eighty three and as counsel in nineteen eighty six",
    "a senior painewebber official said the firm hopes the job can be cut through attrition since the turnover in such positions tends to be high",
]

year = ("he retired as a partner in one thousand, nine hundred and eighty-three "
        "and as counsel in one thousand, nine hundred and eighty-six")
firm = ("official said the firm hopes the job can be cut through attrition "
        "since the turnover in such positions tends to be high")
nbest = [   # real Whisper 5-best lists; nbest[i][0] is the 1-best
    ["obviously we were disappointed if we did not get a larger award",
     "quote obviously we were disappointed we did not get a larger award",
     "obviously we were disappointed if we did not get a larger award",
     "quote obviously we were disappointed we did not get a larger award",
     "quote obviously we were disappointed we did not get a larger award"],
    [year] * 5,   # all five hypotheses agree on the wrong year reading
    [f"a senior {x} {firm}" for x in
     ["painewebber", "payne weber", "payne weber", "paine webber", "pain weber"]],
]

lm_corrected = [   # the LLM corrector's verbatim outputs
    "quote obviously we were disappointed we did not get a larger award",
    "he retired as a partner in nineteen eighty three and as counsel in nineteen eighty six",
    "a senior paine webber official said the firm hopes the job can be cut through attrition since the turnover in such positions tends to be high",
]

out = agwer.evaluate(refs, lm_corrected, nbest=nbest)
out.wer_1best            # 0.2264  the raw ASR
out.wer_oracle           # 0.1887  o_nb: the best any reranker could reach
out.wer_compositional    # 0.0377  o_cp: recombining n-best tokens cannot write "nineteen"
out.wer_corrected        # 0.0377
out.rir                  # 5.0     five times the reranking headroom: generative correction
out.her                  # 0.3333  one of the three consequential edits broke a correct name
```

Real decodes also show why reranking alone cannot save a voice agent. In
another HyPoradise utterance, the command *"leaving **after noon**"* comes
back as *"leaving **afternoon**"* in **all five** hypotheses. The query's
meaning flips, every reranker is helpless, and only a corrector (and agwer's
oracles) can see it.

### When the corrector beats every hypothesis (ρ > 1)

Look at the second utterance above. All five hypotheses read the year the
same wrong way, so the reranking oracle equals the 1-best and offers zero
headroom. Recombining *n*-best tokens cannot help either, because the word
*"nineteen"* appears nowhere in the list. The corrector still writes
*"nineteen eighty three"*, the dictation convention it knows from context.
Supplying truth that exists in no hypothesis is *generative* correction. It
is why ρ can exceed 1, and it is exactly what plain WER, reranking metrics,
and oracle bounds cannot see. RIR was built to measure it.

Dictating to a coding agent has the same structure: package names and code
terms (`uv`, `pytest`, `agwer`) are exactly the tokens ASR breaks and
exactly the ones the agent can restore from context.

On the full 30-utterance WSJ set these corrections come from, the corrector
scores ρ = 2.0 with HER = 0.29: it recovers twice the *n*-best headroom, and
roughly two of every seven consequential edits still break something. That
pair of numbers, not WER alone, is the act/abstain profile agwer exists to
report.

### Entity F1 and Word Hallucination Rate

Two failure modes that plain WER cannot name. First, entity errors hide
inside acceptable WER: one substituted amount in a 21-word transfer request
is a comfortable 4.8% WER and a broken transaction. `entity_f1` scores an
explicit token subset with the same alignment engine as WER:

```python
ref = "okay so please transfer five hundred dollars to the savings account before friday and send me a confirmation text right away"
hyp = "okay so please transfer nine hundred dollars to the savings account before friday and send me a confirmation text right away"

agwer.wer(ref, hyp)                                   # 0.0476, looks fine
m = agwer.entity_f1(ref, hyp, entities={"five", "hundred"})
m["recall"], m["entity_wer"]                          # 0.5, 0.5: it is not fine
agwer.entity_f1(ref, hyp, predicate=agwer.numeric_tokens)  # ready-made subset: digits and spelled numbers
```

Second, LLM correctors make text up, and the two mechanisms differ. A
corrector can invent a word the ASR never produced, or it can loop
autoregressively and repeat words it really heard (one system in
[arXiv:2408.16180](https://arxiv.org/abs/2408.16180) repeats a real segment
eleven times, so a vocabulary check sees nothing wrong).
`word_hallucination_rate` uses occurrence-bounded attribution to catch both,
works with a single 1-best or a full n-best list, and never penalizes
copying an ASR error or a correct generative recovery:

```python
m = agwer.word_hallucination_rate(
    ["cancel my nine a m meeting"],                   # references
    ["cancel my nine a m meeting meeting meeting"],   # corrector outputs
    ["cancel my nine a m meeting"])                   # hypotheses (1-best or n-best)

m["whr"]                          # 0.25: two of eight output tokens hallucinated
m["repetition_hallucinations"]    # 2, an autoregressive loop over a heard word
m["novel_hallucinations"]         # 0, nothing invented outright
```

The decomposition also reports `passed_through_errors` (copied ASR errors,
not hallucination) and `generative_tokens` (correct words no hypothesis
contained, the ρ > 1 mechanism).

HER comes in two granularities. `her_granularity="utterance"` (the default)
is the accounting behind the paper's reported values, and `"token"` follows
the formal per-edit definition. A sentence where the corrector fixes one
token and breaks another is *neutral* at utterance granularity but
*helpful=1, harmful=1* at token granularity, so report which one you used.

### Multi-speaker ASR: cpWER

Meeting and conversation transcripts come speaker-attributed, and the
speaker labels a system assigns are arbitrary. `cpwer` concatenates each
speaker's utterances and finds the label permutation with the fewest word
errors (the MeetEval definition, validated against it):

```python
ref = {"alice": "let us start with the quarterly numbers",
       "bob": "sounds good i will share my screen"}
hyp = {"spk0": "sounds good i will share my screen",
       "spk1": "let us start with the quality numbers"}

agwer.cpwer(ref, hyp)                      # 0.0714, labels permuted, one word wrong
agwer.cp_statistics(ref, hyp)["assignment"]  # [('alice', 'spk1'), ('bob', 'spk0')]
```

Values may be single strings or lists of utterances (concatenated in the
given order). A missed speaker counts all its words as deletions, an extra
hypothesis speaker all its words as insertions; `cp_statistics` reports the
missed and false-alarm speaker counts.

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

The Whisper normalizers have their behavior pinned **byte-identical** to the original by golden
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
| 10k utterances (single-threaded) | 0.15 s |
| 100k utterances (single-threaded) | 1.79 s |
| 100k utterances (8 workers) | **0.38 s** (4.7×) |
| 1M utterances (single-threaded) | 18.8 s |
| 1M utterances (8 workers) | **4.0 s** (4.7×) |

Workers pay process startup, so they win from roughly 100k utterances up. On
macOS and Windows, call from a `if __name__ == "__main__"` guard, as with any
multiprocessing. Reproduce on your machine:

```bash
python -m agwer.bench --workers 8
```

### Apple Silicon

[![Speed](https://img.shields.io/badge/1M%20utterances-4.0s%20on%20M--series-brightgreen)](#performance)

agwer is **native on Apple Silicon**, with no separate
install: pip and uv select the arm64 wheel automatically, and RapidFuzz
ships compiled `arm64-darwin` extensions, so the C++ edit-distance core runs
natively on M-series machines. `workers=` then scales the whole pipeline
across performance cores (see the table above).

- Planned next is an optional `agwer[mlx]` extra for embedding-based *semantic* metrics on the Apple GPU
via [MLX](https://github.com/ml-explore/mlx): semantic inference is the one place extra hardware genuinely helps; edit distance does not need it.

## Compatibility & reproducibility

Measure semantics match [jiwer](https://github.com/jitsi/jiwer), validated
bit-identical on a 600-corpus golden set pinned in the package test suite.

agwer gratefully builds on and learns from these projects:

* [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) provides the C++
  bit-parallel edit-distance engine, agwer's only dependency;
  [FuzzyMatch](https://github.com/ordo-one/FuzzyMatch) shows the same
  design done right in Swift.
* [jiwer](https://github.com/jitsi/jiwer) defined the measure semantics and
  the easy, friendly API style this package follows.
* [OpenAI Whisper](https://github.com/openai/whisper) created the English
  text normalizer that became the community standard for ASR evaluation. We
  vendor it faithfully under its MIT license, with attribution.
* [NVIDIA NeMo text processing](https://github.com/NVIDIA/NeMo-text-processing)
  sets the bar for full text normalization across languages. Its semiotic
  class taxonomy guides our normalization roadmap.
* [MeetEval](https://github.com/fgnt/meeteval) defined the cpWER reference
  implementation for multi-speaker ASR. agwer's `cpwer` matches it on a
  golden set of 87 pinned cases; for time-constrained metrics (tcpWER) and
  ORC/MIMO variants, use MeetEval itself.

## License

Apache-2.0. Vendored Whisper normalizers: MIT (see
`src/agwer/normalizers/LICENSE_WHISPER`).
