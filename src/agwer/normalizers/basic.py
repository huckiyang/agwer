"""Basic (language-agnostic) text normalizer.

Vendored from OpenAI Whisper (MIT License, Copyright (c) 2022 OpenAI; see
LICENSE_WHISPER in this directory), with two agwer adaptations that do not
change behavior:
* the third-party ``regex`` module is imported lazily and only needed for the
  rare ``split_letters=True`` path (grapheme clusters),
* the fixed patterns are precompiled at import time.
"""

from __future__ import annotations

import re
import unicodedata

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}

_BRACKETS = re.compile(r"[<\[][^>\]]*[>\]]")
_PARENS = re.compile(r"\(([^)]+?)\)")
_WS = re.compile(r"\s+")


def remove_symbols_and_diacritics(s: str, keep: str = "") -> str:
    """Replace markers/symbols/punctuation with a space and drop diacritics
    (category 'Mn' plus the manual mappings above)."""
    return "".join(
        (
            c
            if c in keep
            else (
                ADDITIONAL_DIACRITICS[c]
                if c in ADDITIONAL_DIACRITICS
                else (
                    ""
                    if unicodedata.category(c) == "Mn"
                    else " " if unicodedata.category(c)[0] in "MSP" else c
                )
            )
        )
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str) -> str:
    """Replace markers/symbols/punctuation with a space, keeping diacritics."""
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    )


class BasicTextNormalizer:
    """Lowercase; strip bracketed/parenthesized asides, symbols, punctuation.

    Language-agnostic. ``remove_diacritics=True`` also folds accents;
    ``split_letters=True`` splits into grapheme clusters (requires the
    third-party ``regex`` package).
    """

    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str) -> str:
        s = s.lower()
        s = _BRACKETS.sub("", s)
        s = _PARENS.sub("", s)
        s = self.clean(s).lower()

        if self.split_letters:
            try:
                import regex
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "split_letters=True requires the 'regex' package: "
                    "pip install regex"
                ) from e
            s = " ".join(regex.findall(r"\X", s, regex.U))

        return _WS.sub(" ", s)
