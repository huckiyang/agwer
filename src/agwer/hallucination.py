"""Word Hallucination Rate (WHR) for generative ASR correction.

LLM correctors fail in a way plain WER cannot name: they make text up. The
two mechanisms differ (see arXiv:2408.16180, Table 7, for striking Japanese
GER examples):

* **novel** hallucination: the output contains a word the ASR never produced
  in any hypothesis, and it is wrong (an invented word);
* **repetition** hallucination: autoregressive loops copy words the ASR
  *did* produce, more times than they were heard (one system in the paper
  repeats a real segment eleven times). Vocabulary checks cannot see this;
  attribution must be occurrence-bounded.

Definition. For each output token, two questions are asked:

1. Is it correct? A token is correct iff it lies in an ``equal`` span of the
   minimal word-level alignment between output and reference.
2. Is it supported by the source? The support budget of word ``w`` is the
   maximum number of times ``w`` occurs in any single source hypothesis
   (works with a 1-best string or an n-best list). Output occurrences of
   ``w`` beyond the budget are unsupported.

A token is **hallucinated** when it is wrong and unsupported (the excess is
attributed to wrong tokens first, so correct repetitions of a heard word are
never penalized). Wrong but supported tokens are **passed-through ASR
errors** (copying the input is not hallucinating). Correct tokens whose word
appears in no hypothesis are **generative recoveries**, the mechanism behind
rho > 1.

    WHR = hallucinated output tokens / output tokens
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional, Sequence, Union

from rapidfuzz.distance import Levenshtein

from agwer.transforms import default_normalize

__all__ = ["word_hallucination_rate"]

Hypotheses = Union[Sequence[str], Sequence[Sequence[str]]]


def _toks(s: str) -> list:
    return [t for t in s.split(" ") if t]


def _out_flags(ref_tok: list, out_tok: list) -> list:
    """Per-output-token correctness from the minimal word alignment."""
    ok = [False] * len(out_tok)
    for op in Levenshtein.opcodes(ref_tok, out_tok):
        if op.tag == "equal":
            for j in range(op.dest_start, op.dest_end):
                ok[j] = True
    return ok


def word_hallucination_rate(
    references: Sequence[str],
    outputs: Sequence[str],
    hypotheses: Hypotheses,
    normalize: Optional[Callable[[str], str]] = default_normalize,
) -> dict:
    """Compute WHR and its mechanism decomposition for a corpus.

    Args:
        references: ground-truth transcripts.
        outputs: the corrector's outputs (the text under test).
        hypotheses: what the corrector was given; either one 1-best string
            per utterance or an n-best list per utterance.
        normalize: applied to every string (default: paper-compatible
            :func:`agwer.default_normalize`; ``None`` scores raw strings).

    Returns a dict with ``whr`` plus the decomposition:
    ``novel_hallucinations``, ``repetition_hallucinations``,
    ``passed_through_errors`` (wrong but source-supported),
    ``generative_tokens`` (correct and absent from every hypothesis), and
    the token totals.
    """
    n = len(references)
    if not (len(outputs) == n and len(hypotheses) == n):
        raise ValueError(
            "references, outputs, and hypotheses must have the same length"
        )
    norm = (lambda s: s) if normalize is None else normalize

    out_total = halluc_novel = halluc_rep = passed = generative = 0
    for ref, out, hyp in zip(references, outputs, hypotheses):
        hyps = [hyp] if isinstance(hyp, str) else list(hyp)
        if not hyps:
            raise ValueError("every utterance needs at least one hypothesis")
        ref_tok = _toks(norm(ref))
        out_tok = _toks(norm(out))
        # support budget: max occurrences of each word in any one hypothesis
        budget: Counter = Counter()
        for h in hyps:
            for w, c in Counter(_toks(norm(h))).items():
                if c > budget[w]:
                    budget[w] = c

        ok = _out_flags(ref_tok, out_tok)
        out_total += len(out_tok)
        c_out = Counter(out_tok)
        wrong = Counter(t for t, k in zip(out_tok, ok) if not k)
        correct = Counter(t for t, k in zip(out_tok, ok) if k)
        for w, c in c_out.items():
            excess = max(0, c - budget[w])
            hallucinated = min(wrong[w], excess)
            if budget[w] == 0:
                halluc_novel += hallucinated
                generative += correct[w]
            else:
                halluc_rep += hallucinated
            passed += wrong[w] - hallucinated

    hallucinated = halluc_novel + halluc_rep
    return {
        "output_tokens": out_total,
        "hallucinated_tokens": hallucinated,
        "novel_hallucinations": halluc_novel,
        "repetition_hallucinations": halluc_rep,
        "passed_through_errors": passed,
        "generative_tokens": generative,
        "whr": hallucinated / out_total if out_total else 0.0,
    }
