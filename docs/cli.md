# CLI

```bash
agwer results.jsonl
agwer results.jsonl --json --her-granularity token
agwer results.jsonl --raw          # skip default normalization
```

Each JSONL line is one utterance:

```json
{"reference": "the cat sat", "corrected": "the cat sat",
 "nbest": ["the cat sad", "the cat sat"]}
```

`nbest[0]` must be the 1-best. `"onebest": "..."` may replace `nbest`
(RIR and the oracles then unavailable; HER + WERs still reported).

The human-readable report rounds error rates to the standard ASR
convention, one decimal (`xx.x%`). `--json` always emits full-precision
floats, and the Python API never rounds; round only at the reporting
step.

## LLM output formats

Corrector outputs usually live inside chat logs. `--format` accepts them
directly (`auto`, the default, detects each shape); the corrected
transcript is taken from the last assistant turn, and structured outputs
with a `"corrected"` key are unwrapped. `reference` and `nbest` (or
`onebest`) always ride along as top-level keys.

OpenAI chat sessions (`--format openai`):

```json
{"reference": "a b c", "nbest": ["a x c", "a b c"],
 "messages": [{"role": "system", "content": "correct the asr"},
              {"role": "user", "content": "hypothesis 1: a x c ..."},
              {"role": "assistant", "content": "a b c"}]}
```

Anthropic Messages with structured output (`--format anthropic`):

```json
{"reference": "a b c", "nbest": ["a x c", "a b c"],
 "response": {"role": "assistant", "stop_reason": "end_turn",
              "content": [{"type": "text",
                           "text": "{\"corrected\": \"a b c\"}"}]}}
```

ShareGPT conversations (`--format sharegpt`):

```json
{"reference": "a b c", "nbest": ["a x c", "a b c"],
 "conversations": [{"from": "human", "value": "hypothesis 1: a x c ..."},
                   {"from": "gpt", "value": "a b c"}]}
```

Parquet batches (`--format parquet`, columns `reference`, `corrected`,
`nbest`) for efficient many-entry processing; needs
`pip install "agwer[parquet]"`.

A ready-made test set with the same 30 real correction sessions in all
five formats lives at
[huckiyang/agwer_asr_batch_test_v0](https://huggingface.co/datasets/huckiyang/agwer_asr_batch_test_v0):

```bash
hf download huckiyang/agwer_asr_batch_test_v0 --repo-type dataset --local-dir batch
agwer batch/input.openai.jsonl      # identical report from every file
```

Benchmark your install:

```bash
python -m agwer.bench
```
