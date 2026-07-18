"""Text normalizers for ASR evaluation.

Normalization is the main source of incomparable WER numbers across papers;
agwer ships the standard ones so results are reproducible by construction.
Every normalizer is a plain ``Callable[[str], str]`` and plugs into any agwer
entry point via ``normalize=``.

* :func:`agwer.default_normalize` - conservative (lowercase, keep apostrophes,
  strip other punctuation); the Voice Memory paper's convention. Fastest.
* :class:`BasicTextNormalizer` - language-agnostic Whisper basic normalizer
  (symbols, brackets, optional diacritic folding).
* :class:`EnglishTextNormalizer` - the Whisper English normalizer, the
  de-facto ASR-eval standard (numbers, currency, contractions,
  British->American spelling). ``cached=True`` adds an LRU for agent loops.

The Whisper normalizers are vendored (MIT, Copyright (c) 2022 OpenAI - see
LICENSE_WHISPER) with behavior pinned against the original by golden tests;
agwer only removed third-party dependencies and precompiled the patterns.
For multilingual WFST normalization (NeMo), see ENGINEERING_PLAN.md.
"""

from agwer.normalizers.basic import (
    BasicTextNormalizer,
    remove_symbols,
    remove_symbols_and_diacritics,
)
from agwer.normalizers.english import (
    EnglishNumberNormalizer,
    EnglishSpellingNormalizer,
    EnglishTextNormalizer,
)

__all__ = [
    "BasicTextNormalizer",
    "EnglishTextNormalizer",
    "EnglishNumberNormalizer",
    "EnglishSpellingNormalizer",
    "remove_symbols",
    "remove_symbols_and_diacritics",
]
