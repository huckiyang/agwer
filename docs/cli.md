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

Benchmark your install:

```bash
python -m agwer.bench
```
