"""Resolve a question to a table cell, or decline.

This is the deterministic baseline for Table QA -- no model, no code
execution. It exists to prove the *architecture* is what moves numeric
accuracy, not the size of the language model bolted onto it. A real TAPAS or
LLM row-resolver drops in behind `answer_numeric` and gets measured on the
same golden set.

Declining is a first-class outcome. Two rules produce it, and both matter more
than the row matcher:

  1. A question naming a period the table does not have is refused outright.
     Answering "fiscal 2023 segment revenue" with the fiscal 2024 column is
     the single most dangerous failure mode in filing analysis.
  2. Question content words matched by nothing in the table erode confidence.
     "What share of net revenue came from the largest customer?" matches the
     row "Net revenue" perfectly and is still not a question about that cell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ledgerline.tables.model import Table, TableStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_YEAR_RE = re.compile(r"\b(?:fiscal\s+|fy\s*)?((?:19|20)\d{2})\b", re.IGNORECASE)

# Interrogatives and filler. Kept separate from the retrieval stoplist: here
# the job is to isolate the *content* of the question so unmatched content can
# be counted honestly.
_FILLER = frozenset(
    (
        "what was were is are the a an in of for how much many did do does on "
        "at to and we our us company total this that from by with be been had "
        "has have there it its their any which when during as into over"
    ).split()
)

#: Row-label agreement below this refuses. Tuned on the numeric golden set,
#: and recalibrated when `score_row` became symmetric: an F1 is bounded by the
#: weaker of its two sides, so scores that used to clear 0.60 on label coverage
#: alone now land lower without the match being any worse. Measured over
#: 0.30-0.60 on both golden sets, 0.40 is the middle of a flat region rather
#: than an edge -- 0.35 and 0.30 score identically.
CONFIDENCE_FLOOR = 0.40
#: Each unmatched content word in the question costs this much confidence.
UNMATCHED_PENALTY = 0.10


@dataclass(frozen=True)
class Answer:
    value: float
    raw: str
    unit: str
    table_id: str
    row_label: str
    column: str
    confidence: float

    def citation(self) -> str:
        """Where the number came from, in the form a reviewer can check."""
        return f"{self.table_id}[{self.row_label!r}, {self.column!r}]"


@dataclass(frozen=True)
class Declined:
    reason: str
    best_row: str | None = None
    confidence: float = 0.0


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _FILLER]


def requested_year(question: str) -> str | None:
    match = _YEAR_RE.search(question)
    return match.group(1) if match else None


def _column_for(table: Table, year: str | None) -> str | None:
    if year is None:
        # No period named: the current period is the first column, which is
        # how every filing orders them.
        return table.columns[0] if table.columns else None
    for column in table.columns:
        if year in column:
            return column
    return None


def score_row(question_tokens: set[str], row_label: str) -> float:
    """How well a row label and a question account for each other.

    This used to be label coverage alone -- the fraction of the label's tokens
    the question supplies -- on the reasoning that a longer, more specific
    label would win. It does the opposite whenever the short label is a subset
    of the long one, because a two-word label the question happens to contain
    scores a perfect 1.000 and nothing longer can beat it.

    That is not hypothetical. Three tables carry a bare row labelled
    "Construction Industries", and for "What were Construction Industries total
    sales in 2025?" the bare label scored 1.000 while the row that actually
    holds the figure, "Construction Industries -- Total Sales and Revenues",
    scored 0.750. The question was answered 5,442 -- the segment's *assets* --
    with a citation attached.

    So the label has to account for the question as well: an F1 over the two
    token sets, which only rewards a label for being specific if the
    specificity is what was asked for. Measured on both golden sets it does not
    merely re-rank, it removes the wrong answers outright -- 32 right with 1
    wrong becomes 33 right with 0 wrong, and the synthetic set is unchanged.
    """
    label_tokens = set(tokenize(row_label))
    if not label_tokens or not question_tokens:
        return 0.0
    shared = len(label_tokens & question_tokens)
    if not shared:
        return 0.0
    precision = shared / len(label_tokens)
    recall = shared / len(question_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_numeric(question: str, store: TableStore) -> Answer | Declined:
    tokens = set(tokenize(question))
    year = requested_year(question)

    best: tuple[float, Table, str, str] | None = None
    period_miss: str | None = None

    for table in store.tables:
        # Anything named in the caption or the column headers counts as
        # "explained by this table" when charging the unmatched penalty.
        context = set(tokenize(table.caption))
        for column in table.columns:
            context |= set(tokenize(column))

        for row_label in table.row_labels:
            coverage = score_row(tokens, row_label)
            if coverage <= 0:
                continue

            column = _column_for(table, year)
            if column is None:
                # The row exists but the period does not. Remember it so the
                # refusal can say *why* rather than "no match".
                if coverage >= CONFIDENCE_FLOOR:
                    period_miss = row_label
                continue

            explained = context | set(tokenize(row_label))
            unmatched = len(tokens - explained)
            confidence = coverage - UNMATCHED_PENALTY * unmatched

            if best is None or confidence > best[0]:
                best = (confidence, table, row_label, column)

    if period_miss and (best is None or best[0] < CONFIDENCE_FLOOR):
        return Declined(
            reason=f"no column for the requested period ({year})", best_row=period_miss
        )

    if best is None:
        return Declined(reason="no row matched the question")

    confidence, table, row_label, column = best
    if confidence < CONFIDENCE_FLOOR:
        return Declined(
            reason="row match below confidence floor",
            best_row=row_label,
            confidence=confidence,
        )

    cell = table.cell(row_label, column)
    if cell is None or cell.value is None:
        return Declined(
            reason="matched cell is empty", best_row=row_label, confidence=confidence
        )

    return Answer(
        value=cell.value,
        raw=cell.raw,
        unit=table.unit_for(row_label),
        table_id=table.id,
        row_label=row_label,
        column=column,
        confidence=confidence,
    )
