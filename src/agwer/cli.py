"""Command-line interface: ``agwer data.jsonl``.

Each JSONL line is one utterance with fields:

    {"reference": "...", "corrected": "...", "nbest": ["...", ...]}

``nbest[0]`` must be the 1-best. ``"onebest": "..."`` may replace ``nbest``
(then RIR is unavailable and only HER + WERs are reported).

LLM output logs are accepted directly via ``--format`` (auto-detected by
default): OpenAI chat sessions, Anthropic Messages/structured outputs,
ShareGPT conversations, and parquet batches — see :mod:`agwer.formats`.

Example::

    agwer results.jsonl
    agwer sessions.openai.jsonl        # format sniffed from the records
    agwer batch.parquet --format parquet
    agwer results.jsonl --her-granularity token --raw --json
"""

from __future__ import annotations

import argparse
import json
import sys

from agwer.agentic import evaluate
from agwer.formats import FORMATS, load_records
from agwer.text import default_normalize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="agwer",
        description="Agentic WER metrics (RIR / HER) for ASR error correction.",
    )
    ap.add_argument("jsonl", help="input file: {reference, corrected, nbest|onebest}")
    ap.add_argument(
        "--format", choices=FORMATS, default="auto",
        help="input format: native jsonl, openai/anthropic/sharegpt chat "
             "logs, or parquet (default: auto-detect)",
    )
    ap.add_argument(
        "--her-granularity", choices=["utterance", "token"], default="utterance",
        help="HER accounting granularity (default: utterance, as reported in the paper)",
    )
    ap.add_argument(
        "--raw", action="store_true",
        help="score raw strings (skip the default paper normalization)",
    )
    ap.add_argument("--json", action="store_true", help="print metrics as JSON")
    args = ap.parse_args(argv)

    rows = load_records(args.jsonl, args.format)
    refs = [r["reference"] for r in rows]
    corr = [r["corrected"] for r in rows]
    nbest = [r["nbest"] for r in rows] if "nbest" in rows[0] else None
    onebest = [r["onebest"] for r in rows] if nbest is None else None

    out = evaluate(
        refs, corr, nbest=nbest, onebest=onebest,
        normalize=None if args.raw else default_normalize,
        her_granularity=args.her_granularity,
    )

    if args.json:
        print(json.dumps(out.as_dict(), indent=2))
        return 0

    def pct(x):
        # standard ASR reporting convention: one decimal (xx.x%).
        # --json keeps full-precision floats for machines.
        return f"{x * 100:.1f}%" if x is not None else "n/a"

    def num(x):
        return f"{x:.3f}" if x is not None else "n/a"

    print(f"n utterances     : {out.n}")
    print(f"WER 1-best       : {pct(out.wer_1best)}")
    print(f"WER corrected    : {pct(out.wer_corrected)}")
    print(f"WER oracle (o_nb): {pct(out.wer_oracle)}")
    print(f"WER compos.(o_cp): {pct(out.wer_compositional)}")
    print(f"RIR (rho)        : {num(out.rir)}"
          + ("   [>1 beats the n-best oracle]" if out.rir is not None and out.rir > 1 else ""))
    print(f"HER ({out.edits.granularity:9s}): {num(out.her)}")
    e = out.edits
    print(f"edits            : helpful={e.helpful} harmful={e.harmful} "
          f"neutral={e.neutral} missed={e.missed} no_edit={e.no_edit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
