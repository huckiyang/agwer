"""Command-line interface: ``agwer data.jsonl``.

Each JSONL line is one utterance with fields:

    {"reference": "...", "corrected": "...", "nbest": ["...", ...]}

``nbest[0]`` must be the 1-best. ``"onebest": "..."`` may replace ``nbest``
(then RIR is unavailable and only HER + WERs are reported).

Example::

    agwer results.jsonl
    agwer results.jsonl --her-granularity token --raw --json
"""

from __future__ import annotations

import argparse
import json
import sys

from agwer.agentic import evaluate
from agwer.text import default_normalize


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON ({e})") from e
    if not rows:
        raise SystemExit(f"{path}: no records")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="agwer",
        description="Agentic WER metrics (RIR / HER) for ASR error correction.",
    )
    ap.add_argument("jsonl", help="JSONL file: {reference, corrected, nbest|onebest}")
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

    rows = _load_jsonl(args.jsonl)
    missing = [k for k in ("reference", "corrected") if k not in rows[0]]
    if missing:
        raise SystemExit(f"records must contain fields: {missing}")

    refs = [r["reference"] for r in rows]
    corr = [r["corrected"] for r in rows]
    nbest = [r["nbest"] for r in rows] if "nbest" in rows[0] else None
    onebest = [r["onebest"] for r in rows] if nbest is None else None
    if nbest is None and onebest is None:
        raise SystemExit("records must contain 'nbest' (list) or 'onebest' (string)")

    out = evaluate(
        refs, corr, nbest=nbest, onebest=onebest,
        normalize=None if args.raw else default_normalize,
        her_granularity=args.her_granularity,
    )

    if args.json:
        print(json.dumps(out.as_dict(), indent=2))
        return 0

    def pct(x):
        return f"{x * 100:.2f}%" if x is not None else "n/a"

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
