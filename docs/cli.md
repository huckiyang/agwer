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

Benchmark your install:

```bash
python -m agwer.bench
```
