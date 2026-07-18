# Example session

A complete walkthrough, from install to CLI. Every value on this page is the
real output of the released package.

## 1. Install and self-test

The test suite ships inside the package, so the first thing any install can
do is prove itself correct:

```console
$ pip install "agwer[test]"
$ python -m pytest --pyargs agwer -q
68 passed in 0.27s
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

## 5. Evaluate a corrector end to end

A real Whisper 5-best decode from the HyPoradise benchmark (WSJ): the ASR
merges the spelled ticker *"i b m"* and garbles the fractions, and the
spelled form survives in **no** hypothesis. The corrector resolves both from
context:

```pycon
>>> ref = "i b m fell one and seven eighths to one hundred twenty and three eighths on more than two point five million shares"
>>> nbest = [[
...     "ibm fell one seven eight to one hundred and twenty three eight on more than two point five million shares",
...     "ibm fell one point seven eight to one hundred and twenty point three eight on more than two point five million shares",
...     "ibm fell one and seven eighths to one hundred and twenty and three eighths on more than two point five million shares",
...     "ibm fell one point seven eights to one hundred and twenty point three eights on more than two point five million shares",
...     "ibm fell one seven eighths to one hundred and twenty three eighths on more than two point five million shares",
... ]]
>>> out = agwer.evaluate([ref], [ref], nbest=nbest)
>>> out.wer_1best, out.wer_oracle, out.wer_compositional
(0.34782608695652173, 0.17391304347826086, 0.13043478260869565)
>>> out.rir, out.her
(2.0, 0.0)
```

ρ = 2.0: the correction recovered twice the *n*-best headroom, because it
supplied truth ("i b m") that exists nowhere in the hypothesis list. That
generative signature is exactly what RIR was built to measure. And HER = 0.0
says the corrector's one consequential edit broke nothing.

## 6. The same evaluation from the shell

Put one utterance per line in a JSONL file with `reference`, `corrected`,
and `nbest` (or `onebest`) fields, and the `agwer` command prints the full
report:

```console
$ agwer results.jsonl
n utterances     : 1
WER 1-best       : 34.78%
WER corrected    : 0.00%
WER oracle (o_nb): 17.39%
WER compos.(o_cp): 13.04%
RIR (rho)        : 2.000   [>1 beats the n-best oracle]
HER (utterance): 0.000
edits            : helpful=1 harmful=0 neutral=0 missed=0 no_edit=0
```

`--her-granularity token` switches to the formal per-edit accounting,
`--raw` skips normalization, and `--json` emits machine-readable output.
