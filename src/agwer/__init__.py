"""agwer — agentic WER metrics for ASR error correction.

Evaluates the act/abstain behavior of an LLM corrector over n-best ASR
hypotheses with two measures from the Voice Memory paper:

* RIR (rho): Recoverable Information Ratio — how much of the 1-best-to-oracle
  gap a correction closes (rho > 1 beats the n-best oracle; rho < 0 is the
  damage regime).
* HER: Harmful Edit Rate — of the corrector's edits, the fraction that broke a
  correct token (over-correction).

Self-contained on RapidFuzz; one alignment core (:mod:`agwer.align`) feeds
every metric, so tokenization and alignment semantics cannot drift between
them. Quickstart::

    import agwer

    out = agwer.evaluate(references, corrected, nbest=nbest)
    print(out.rir, out.her, out.wer_corrected)
"""

from agwer.agentic import (
    AgenticOutput,
    compositional_oracle_wer,
    evaluate,
    her,
    oracle_hypotheses,
    oracle_select,
    oracle_wer,
    process_agentic,
    rho,
    rir,
)
from agwer.classic import cer, mer, ser, wer, wil, wip
from agwer.edits import EditCounts, classify_tokens, classify_utterance
from agwer.entity import entity_f1, numeric_tokens
from agwer.hallucination import word_hallucination_rate
from agwer.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from agwer.text import default_normalize

__version__ = "0.4.1"

__all__ = [
    "wer",
    "mer",
    "wil",
    "wip",
    "cer",
    "ser",
    "entity_f1",
    "numeric_tokens",
    "word_hallucination_rate",
    "evaluate",
    "rir",
    "rho",
    "her",
    "oracle_wer",
    "oracle_hypotheses",
    "compositional_oracle_wer",
    "oracle_select",
    "process_agentic",
    "AgenticOutput",
    "EditCounts",
    "classify_utterance",
    "classify_tokens",
    "default_normalize",
    "BasicTextNormalizer",
    "EnglishTextNormalizer",
    "__version__",
]
