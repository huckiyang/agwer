# Example session

A complete walkthrough, from install to CLI. Every value on this page is the
real output of the released package.

## 1. Install and self-test

The test suite ships inside the package, so the first thing any install can
do is prove itself correct:

```console
$ pip install "agwer[test]"
$ python -m pytest --pyargs agwer -q
101 passed in 0.38s
```

## 2. First measurements

A dictation agent hears a banking command and breaks one word — the amount:

```pycon
>>> import agwer
>>> ref = "transfer twelve hundred fifty dollars to the joint account on monday"
>>> hyp = "transfer twelve hundred fifteen dollars to the joint account on monday"
>>> agwer.wer(ref, hyp)
0.09090909090909091
>>> agwer.ser([ref, ref], [hyp, ref])
0.5
```

One word in eleven: a comfortable 9% word error rate, and half the
utterances touched (SER). But WER treats every word equally, and the broken
word is the *amount*.

## 3. Catch the entity error

`entity_f1` scores only the information-carrying tokens, here selected with
the built-in numeric predicate:

```pycon
>>> agwer.entity_f1(ref, hyp, predicate=agwer.numeric_tokens)
{'recall': 0.666..., 'precision': 0.666..., 'f1': 0.666...,
 'entity_wer': 0.333..., 'ref_entities': 3, 'hyp_entities': 3}
```

The same transcript that looked 91% correct lost a third of its numeric
entities. This is the "entity errors hide inside acceptable WER" failure
mode, made measurable.

## 4. Catch hallucination

A corrector that loops autoregressively repeats words it really heard, so
vocabulary checks pass. `word_hallucination_rate` bounds each word by how
often the source actually said it:

```pycon
>>> agwer.word_hallucination_rate(
...     ["cancel my nine a m meeting"],                      # reference
...     ["cancel my nine a m meeting meeting meeting"],      # corrector output
...     ["cancel my nine a m meeting"])                      # what ASR gave it
{'output_tokens': 8, 'hallucinated_tokens': 2,
 'novel_hallucinations': 0, 'repetition_hallucinations': 2,
 'passed_through_errors': 0, 'generative_tokens': 0, 'whr': 0.25}
```

Two of the eight output tokens exceed the source budget: repetition
hallucinations, not invented words.

## 5. Score a multi-speaker meeting

Speaker-attributed transcripts carry arbitrary speaker labels. `cpwer`
finds the label permutation with the fewest word errors (the MeetEval
definition, validated against it on a golden fixture):

```pycon
>>> ref = {"alice": "let us start with the quarterly numbers",
...        "bob": "sounds good i will share my screen"}
>>> hyp = {"spk0": "sounds good i will share my screen",
...        "spk1": "let us start with the quality numbers"}
>>> agwer.cpwer(ref, hyp)
0.07142857142857142
>>> agwer.cp_statistics(ref, hyp)["assignment"]
[('alice', 'spk1'), ('bob', 'spk0')]
```

One word wrong out of fourteen once the labels are matched; the
permuted labels themselves cost nothing.

## 6. Evaluate a corrector end to end

Real data: three real Whisper 5-best decodes from the HyPoradise benchmark
(WSJ), each with the verbatim output of a real LLM corrector. The three
utterances show the three things a corrector can do: restore a word from
another hypothesis, fix a reading that every hypothesis got wrong, and break
a name that was already right:

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

The second utterance is the generative case: all five hypotheses read the
year the same wrong way, so reranking is powerless (o_nb equals the 1-best
there), and no token recombination can produce *"nineteen"* — yet the
corrector writes it. The third utterance is the price of acting: the 1-best
was already perfect, and the corrector split *"painewebber"* into two words.
ρ and HER report both sides of that act/abstain trade at once. On the full
30-utterance WSJ set these corrections come from, this corrector scores
ρ = 2.0 with HER = 0.29.

## 7. The same evaluation from the shell

Put one utterance per line in a JSONL file with `reference`, `corrected`,
and `nbest` (or `onebest`) fields, and the `agwer` command prints the full
report. With the three utterances above:

```console
$ agwer results.jsonl
n utterances     : 3
WER 1-best       : 22.6%
WER corrected    : 3.8%
WER oracle (o_nb): 18.9%
WER compos.(o_cp): 3.8%
RIR (rho)        : 5.000   [>1 beats the n-best oracle]
HER (utterance): 0.333
edits            : helpful=2 harmful=1 neutral=0 missed=0 no_edit=0
```

Error rates in the report follow the standard ASR convention, one decimal
(`xx.x%`). `--her-granularity token` switches to the formal per-edit
accounting, `--raw` skips normalization, and `--json` emits full-precision
machine-readable output (the Python API never rounds either).
