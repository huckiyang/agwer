"""agwer — agentic WER metrics for ASR error correction.

Evaluates the act/abstain behavior of an LLM corrector over n-best ASR
hypotheses with two measures from the Voice Memory paper:

* RIR (rho): Recoverable Information Ratio — how much of the 1-best-to-oracle
  gap a correction closes (rho > 1 beats the n-best oracle; rho < 0 is the
  damage regime).
* HER: Harmful Edit Rate — of the corrector's edits, the fraction that broke a
  correct token (over-correction).

Built on jiwer/RapidFuzz. Quickstart::

    import agwer

    out = agwer.evaluate(references, corrected, nbest=nbest)
    print(out.rir, out.her, out.wer_corrected)
"""

from agwer.edits import EditCounts, classify_tokens, classify_utterance
from agwer.measures import (
    cer,
    compositional_oracle_wer,
    evaluate,
    her,
    mer,
    oracle_hypotheses,
    oracle_wer,
    rho,
    rir,
    wer,
    wil,
    wip,
)
from agwer.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from agwer.process import AgenticOutput, oracle_select, process_agentic
from agwer.transforms import default_normalize

__version__ = "0.1.0"

__all__ = [
    "wer",
    "mer",
    "wil",
    "wip",
    "cer",
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
