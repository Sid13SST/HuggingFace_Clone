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

#: Row-label coverage below this refuses. Tuned on the numeric golden set.
CONFIDENCE_FLOOR = 0.60
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
    """Fraction of the row label's own tokens the question supplies.

    Coverage of the *label* rather than of the question: "Industrial Systems
    operating income" should beat "Industrial Systems revenue" for a question
    about operating income, and label-coverage is what makes the longer, more
    specific label win.
    """
    label_tokens = set(tokenize(row_label))
    if not label_tokens:
        return 0.0
    return len(label_tokens & question_tokens) / len(label_tokens)


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
