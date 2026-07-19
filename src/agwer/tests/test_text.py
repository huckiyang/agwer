"""default_normalize: the ASCII fast path must be byte-identical to the
regex path it replaced, on every kind of input."""

import random
import re
import string

from agwer.text import default_normalize

_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")


def _regex_reference(text):
    text = text.lower().strip()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def test_fast_path_equals_regex_reference():
    rng = random.Random(2)
    alphabet = (string.ascii_letters + string.digits + string.punctuation
                + " \t\n'" + "\x00\x1c\x7f")
    cases = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
             for _ in range(2000)]
    cases += [
        "", " ", "   ", "a\tb\nc", "it's O'Brien's...", "don't -- stop",
        "Hello, World!", "a  b   c", "$1,250.75 (15%)", "'''", "_under_score_",
        # unicode falls back to the regex path; both must agree with reference
        "café — naïve!", "über–maß",
        "日本語、テスト。",
        "٣ عربى!", "mixed café and ascii, too.",
    ]
    for t in cases:
        assert default_normalize(t) == _regex_reference(t), repr(t)
