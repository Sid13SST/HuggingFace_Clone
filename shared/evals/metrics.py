"""Metric primitives.

Kept dependency-light and unit-tested so that a number moving in a report can
be trusted to mean the model changed, not the metric.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def recall_at_k(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    gold = set(relevant)
    if not gold:
        return 0.0
    hits = len(gold & set(ranked[:k]))
    return hits / len(gold)


def precision_at_k(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold = set(relevant)
    return len(gold & set(ranked[:k])) / k


def reciprocal_rank(relevant: Iterable[str], ranked: Sequence[str]) -> float:
    gold = set(relevant)
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    relevant: Iterable[str],
    ranked: Sequence[str],
    k: int,
    gains: dict[str, float] | None = None,
) -> float:
    """Binary-gain nDCG unless a per-document gain map is supplied."""
    gold = set(relevant)
    if not gold:
        return 0.0

    def gain(doc_id: str) -> float:
        if gains is not None:
            return gains.get(doc_id, 0.0)
        return 1.0 if doc_id in gold else 0.0

    dcg = sum(gain(d) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((gain(d) for d in (gains or dict.fromkeys(gold, 1.0))), reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------
# numeric answers
# --------------------------------------------------------------------------

#: Suffixes that name the *unit* a figure is stated in rather than a scale.
#: They behave like "%": the writer has already said what the number measures,
#: so a table header reading "in millions" is talking about the dollar rows and
#: not about this one. Without this, "7 years" in a table of millions parses as
#: seven million years and is then refused for implausibility -- the figure is
#: lost to a confidence floor rather than to a parse error, which is worse
#: because nothing reports it.
_UNIT_WORDS = frozenset(
    {
        "year", "years", "month", "months", "day", "days", "week", "weeks",
        "share", "shares", "employee", "employees", "x",
    }
)

_SCALE_WORDS = {
    "hundred": 1e2,
    "k": 1e3,
    "thousand": 1e3,
    "thousands": 1e3,
    "m": 1e6,
    "mm": 1e6,
    "million": 1e6,
    "millions": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "billions": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}

_NUMBER_RE = re.compile(
    r"""
    (?P<open_paren>\()?          # accounting negative
    \s*[$€£]?\s*
    (?P<sign>[-+])?
    # The comma-grouped form must contain at least one group, otherwise it
    # matches the leading three digits of an ungrouped run and silently turns
    # 1842600000 into 184.
    (?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<suffix>%|[a-zA-Z]+)?
    \s*(?P<close_paren>\))?
    """,
    re.VERBOSE,
)


class UnparseableNumber(ValueError):
    pass


def parse_number(text: str | float | int, *, scale_hint: float = 1.0) -> float:
    """Parse a financial figure into a plain float.

    Handles the four things that make filing figures hard to compare:
    currency symbols, thousands separators, accounting parentheses for
    negatives, and a scale that may live in the string ("$1.2 billion") or in
    the table header ("in thousands", passed as `scale_hint`).

    A scale in the string always wins over the hint -- "$4.1 billion" inside a
    table stated in thousands is still 4.1e9, because the writer overrode the
    header on purpose.
    """
    if isinstance(text, (int, float)):
        return float(text) * scale_hint

    raw = text.strip()
    if not raw:
        raise UnparseableNumber("empty string")

    match = _NUMBER_RE.search(raw)
    if not match:
        raise UnparseableNumber(f"no number in {text!r}")
    return _value_from_match(match, scale_hint)


def _value_from_match(match: re.Match[str], scale_hint: float) -> float:
    value = float(match.group("digits").replace(",", ""))

    suffix = (match.group("suffix") or "").lower()
    if suffix == "%" or suffix in _UNIT_WORDS:
        # The figure states its own unit, so it is already written in full.
        # Percentages are additionally compared as written, not as fractions.
        scale = 1.0
    elif suffix in _SCALE_WORDS:
        scale = _SCALE_WORDS[suffix]
    else:
        scale = scale_hint
    value *= scale

    negative = match.group("sign") == "-" or (
        match.group("open_paren") and match.group("close_paren")
    )
    return -value if negative else value


def extract_numbers(text: str, *, scale_hint: float = 1.0) -> list[float]:
    """Every figure stated in a passage, in order.

    Used by the contradiction checker, which needs *all* the numbers a
    narrative answer asserts rather than just the first -- an answer that
    quotes the right figure alongside a wrong one is still wrong, and
    `parse_number` would only ever see the first.
    """
    return [_value_from_match(m, scale_hint) for m in _NUMBER_RE.finditer(text)]


def numeric_match(
    predicted: str | float | None,
    gold: str | float,
    *,
    rel_tol: float = 0.005,
    abs_tol: float = 0.0,
    scale_hint: float = 1.0,
) -> bool:
    """Tolerant equality for a reported figure.

    Default tolerance is 0.5% relative, which absorbs legitimate rounding
    ("$1.23bn" vs "1,234.5") without absorbing a wrong number.
    """
    if predicted is None:
        return False
    try:
        p = parse_number(predicted, scale_hint=scale_hint)
        g = parse_number(gold, scale_hint=scale_hint)
    except UnparseableNumber:
        return False
    return abs(p - g) <= max(abs_tol, rel_tol * abs(g))


# --------------------------------------------------------------------------
# classification / agreement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    support: int

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}precision": self.precision,
            f"{prefix}recall": self.recall,
            f"{prefix}f1": self.f1,
        }


def binary_prf(y_true: Sequence[bool], y_pred: Sequence[bool]) -> PRF:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, support=sum(1 for t in y_true if t))


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Agreement between two raters, corrected for chance.

    Used to decide whether an LLM judge is trustworthy enough to gate on. An
    unvalidated judge is just a second opinion with a decimal point attached.
    """
    if len(a) != len(b):
        raise ValueError("rater sequences must be the same length")
    n = len(a)
    if n == 0:
        return 0.0

    labels = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = sum(
        (a.count(label) / n) * (b.count(label) / n) for label in labels
    )
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float:
    """ECE with equal-width bins.

    Matters wherever a downstream decision is routed by a confidence score --
    an overconfident detector silently misroutes work.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    n = len(confidences)
    if n == 0:
        return 0.0

    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            i
            for i, c in enumerate(confidences)
            if (c > lo or (b == 0 and c >= lo)) and c <= hi
        ]
        if not members:
            continue
        avg_conf = sum(confidences[i] for i in members) / len(members)
        accuracy = sum(1 for i in members if correct[i]) / len(members)
        total += (len(members) / n) * abs(avg_conf - accuracy)
    return total


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
