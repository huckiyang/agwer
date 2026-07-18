"""Text normalization for agentic WER metrics.

`default_normalize` reproduces the normalization used in the Voice Memory paper
(and standard HyPoradise/GER evaluation): lowercase, conservative punctuation
stripping (apostrophes are kept, so contractions and possessives survive), and
whitespace collapsing.

All agwer entry points accept a ``normalize`` argument:
  * ``default_normalize`` (the default) - paper-compatible scoring.
  * ``None`` - strings are scored exactly as given (jiwer-raw semantics).
  * any ``Callable[[str], str]`` - your own normalizer.
"""

from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

__all__ = ["default_normalize", "apply_normalize"]

_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")


def default_normalize(text: str) -> str:
    """Lowercase, strip punctuation except apostrophes, collapse whitespace."""
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def apply_normalize(
    texts: Sequence[str], normalize: Optional[Callable[[str], str]]
) -> list[str]:
    """Apply a normalizer to a sequence of strings (identity when ``None``)."""
    if normalize is None:
        return [str(t) for t in texts]
    return [normalize(str(t)) for t in texts]
