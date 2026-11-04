"""Answer-string normalisation for CRT scoring v2 (JZ-RESCORE.md §1).

This is the SINGLE SOURCE OF TRUTH for normalisation. Both the deterministic
key layer and the LLM adjudicator see the string this function returns, and the
adjudication cache is keyed on it. The rule is published verbatim in
`CHANGELOG.md` / JZ-RESCORE.md §1 and must not drift.

The rule, applied in exactly this order and NOTHING else:

    lowercase
    replace curly apostrophes ’ and ‘ with '
    strip leading/trailing whitespace
    collapse internal whitespace runs to one space
    strip a leading "$"
    strip trailing "." "!" "?"

Do not stem, do not strip units, do not canonicalise numbers, do not spell-correct.
"""
from __future__ import annotations

import re

_WS_RUN = re.compile(r"\s+")

# U+2019 RIGHT SINGLE QUOTATION MARK, U+2018 LEFT SINGLE QUOTATION MARK
_CURLY_APOSTROPHES = ("’", "‘")


def normalise(raw: object) -> str:
    """Return the normalised form of a raw answer string (JZ-RESCORE.md §1).

    ``None``/NaN and non-string inputs are coerced to the empty string first
    (an empty answer normalises to "" and matches no key, i.e. residual/other).
    """
    if raw is None:
        return ""
    text = str(raw)
    # pandas NaN coerces to the literal "nan"; treat as empty.
    if text == "nan":
        return ""

    # 1. lowercase
    text = text.lower()
    # 2. replace curly apostrophes with a straight apostrophe
    for ch in _CURLY_APOSTROPHES:
        text = text.replace(ch, "'")
    # 3. strip leading/trailing whitespace
    text = text.strip()
    # 4. collapse internal whitespace runs to one space
    text = _WS_RUN.sub(" ", text)
    # 5. strip a leading "$"
    if text.startswith("$"):
        text = text[1:]
    # 6. strip trailing "." "!" "?"
    text = text.rstrip(".!?")
    return text
