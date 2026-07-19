"""Input format adapters: LLM output logs to agwer evaluation records.

The CLI evaluates records of ``{reference, corrected, nbest | onebest}``.
Real corrector outputs usually live inside chat logs, so ``--format``
accepts the common shapes and extracts the corrected text; ``reference``
and ``nbest`` (or ``onebest``) always ride along as top-level keys:

* ``jsonl`` — native records, one JSON object per line.
* ``openai`` — OpenAI chat format: one session per line with ``messages``;
  the corrected text is the last assistant message (string content or a
  list of content parts).
* ``anthropic`` — Anthropic Messages format: the line's ``response`` (or
  the line itself) is an assistant message whose ``content`` text blocks
  are joined. Structured outputs are unwrapped: when the text is a JSON
  object with a ``corrected`` key, that value is used.
* ``sharegpt`` — ShareGPT format: ``conversations`` with from/value turns;
  the corrected text is the last ``gpt``/``assistant`` turn.
* ``parquet`` — columnar batch file with ``reference``, ``corrected``,
  ``nbest``/``onebest`` columns (requires ``pyarrow``:
  ``pip install "agwer[parquet]"``).

``--format auto`` (the default) picks ``parquet`` for ``.parquet`` files
and sniffs the first JSONL record otherwise. The structured-output unwrap
applies to every chat format. Example batch files in all formats (the real
WSJ 30-session corrector run):
https://huggingface.co/datasets/huckiyang/agwer_asr_batch_test_v0
"""

from __future__ import annotations

import json
from typing import Optional

__all__ = ["FORMATS", "load_records"]

FORMATS = ("auto", "jsonl", "openai", "anthropic", "sharegpt", "parquet")


def _fail(path: str, lineno: int, why: str) -> SystemExit:
    return SystemExit(f"{path}:{lineno}: {why}")


def _maybe_structured(text: str) -> str:
    """Unwrap a structured output: JSON object with a 'corrected' key."""
    t = text.strip()
    if t.startswith("{"):
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            return text
        if isinstance(obj, dict) and isinstance(obj.get("corrected"), str):
            return obj["corrected"]
    return text


def _content_text(content) -> str:
    """Message content: a plain string or a list of typed parts/blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)


def _from_openai(row: dict) -> Optional[str]:
    assistant = [m for m in row.get("messages", ())
                 if m.get("role") == "assistant"]
    if not assistant:
        return None
    return _maybe_structured(_content_text(assistant[-1].get("content", "")))


def _from_anthropic(row: dict) -> Optional[str]:
    msg = row.get("response", row)
    if not isinstance(msg, dict) or "content" not in msg:
        return None
    return _maybe_structured(_content_text(msg["content"]))


def _from_sharegpt(row: dict) -> Optional[str]:
    turns = [t for t in row.get("conversations", ())
             if t.get("from") in ("gpt", "assistant")]
    if not turns:
        return None
    return _maybe_structured(str(turns[-1].get("value", "")))


_EXTRACT = {"openai": _from_openai, "anthropic": _from_anthropic,
            "sharegpt": _from_sharegpt}


def _sniff(row: dict) -> str:
    if "messages" in row:
        return "openai"
    if "response" in row or ("content" in row and "role" in row):
        return "anthropic"
    if "conversations" in row:
        return "sharegpt"
    return "jsonl"


def _record(row: dict, fmt: str, path: str, lineno: int) -> dict:
    rec = {k: row[k] for k in ("reference", "corrected", "nbest", "onebest")
           if k in row}
    if fmt != "jsonl":
        corrected = _EXTRACT[fmt](row)
        if corrected is None:
            raise _fail(path, lineno, f"no assistant output found ({fmt} format)")
        rec["corrected"] = corrected
    if "reference" not in rec:
        raise _fail(path, lineno, "missing 'reference'")
    if "corrected" not in rec:
        raise _fail(path, lineno, "missing 'corrected'")
    if "nbest" not in rec and "onebest" not in rec:
        raise _fail(path, lineno, "missing 'nbest' (list) or 'onebest' (string)")
    return rec


def _load_parquet(path: str) -> list:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit(
            "parquet input requires pyarrow: pip install \"agwer[parquet]\""
        ) from None
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise SystemExit(f"{path}: no records")
    return [_record(row, "jsonl", path, i + 1) for i, row in enumerate(rows)]


def load_records(path: str, fmt: str = "auto") -> list:
    """Load evaluation records from ``path`` in the given format."""
    if fmt not in FORMATS:
        raise SystemExit(f"unknown format {fmt!r}; choose from {FORMATS}")
    if fmt == "parquet" or (fmt == "auto" and path.endswith(".parquet")):
        return _load_parquet(path)
    records, sniffed = [], None
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise _fail(path, lineno, f"invalid JSON ({e})") from e
            if sniffed is None:
                sniffed = _sniff(row) if fmt == "auto" else fmt
            records.append(_record(row, sniffed, path, lineno))
    if not records:
        raise SystemExit(f"{path}: no records")
    return records
