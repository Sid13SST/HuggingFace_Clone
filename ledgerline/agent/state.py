"""The state a run carries, and the three ways it can end.

Terminal outcomes are the load-bearing design decision here, so they are named
before any node is written:

  * `answered`  -- the graph produced a figure or a claim, with citations.
  * `refused`   -- the graph declined *on the merits*. The evidence was
                   retrieved, read, and judged insufficient. This is a correct
                   answer to an unanswerable question and is scored as one.
  * `degraded`  -- the graph could not do its job. Retrieval returned nothing,
                   a model was unavailable, a step raised. Nothing is claimed.

Collapsing `refused` and `degraded` into one "no answer" state is the mistake
that makes an agent impossible to operate. They have opposite fixes: a refusal
rate that rises means the questions got harder or the corpus got thinner, and a
degradation rate that rises means something is broken. A single number cannot
tell you which, and the on-call engineer needs to know within seconds.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict


class Route(StrEnum):
    """What kind of question this is, and therefore who should answer it."""

    NUMERIC = "numeric"
    NARRATIVE = "narrative"
    CROSS_MODAL = "cross_modal"


class Outcome(StrEnum):
    ANSWERED = "answered"
    REFUSED = "refused"
    DEGRADED = "degraded"


class Citation(TypedDict, total=False):
    """A claim's receipt. Always resolvable back to a span in a source."""

    chunk_id: str
    kind: str
    section: str | None
    quote: str


class AgentState(TypedDict, total=False):
    """LangGraph state. Every field optional because nodes fill in their own.

    Deliberately flat and JSON-serialisable: it gets checkpointed after every
    node and written to `ledgerline.runs` at the end, and a state that cannot
    round-trip through JSON is a run that cannot be replayed.
    """

    # --- inputs ---
    question: str
    cik: str | None

    # --- plan ---
    route: str
    plan: list[str]

    # --- retrieval ---
    chunk_ids: list[str]
    chunks: list[dict[str, Any]]

    # --- analysts ---
    numeric_value: float | None
    numeric_declined_reason: str | None
    narrative_text: str | None

    # --- reconciliation ---
    contradictions: list[str]

    # --- terminal ---
    outcome: str
    answer: str | None
    citations: list[Citation]
    #: Why the run degraded. Empty on every other outcome. Kept as a list
    #: because more than one thing can be broken at once, and reporting only
    #: the first sends the on-call engineer down one of two rabbit holes.
    degraded_reasons: list[str]
    steps: list[str]


def initial_state(question: str, cik: str | None = None) -> AgentState:
    return AgentState(
        question=question,
        cik=cik,
        plan=[],
        chunk_ids=[],
        chunks=[],
        contradictions=[],
        citations=[],
        degraded_reasons=[],
        steps=[],
    )
