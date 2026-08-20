"""Writing a finished run to `ledgerline.runs`.

The schema comment says "a run you cannot replay is a bug you cannot fix", and
this is the function that makes that true. Three things are worth noting about
what gets stored:

`state` is the full final state as JSON, not a summary. Summaries are written
by someone who already knows which field mattered, and the whole problem with a
production incident is that nobody does yet.

Degraded runs are stored too. The instinct is to persist successes and log
failures, which leaves the failures in a text stream that rotates away in a
fortnight -- exactly the runs you most want to query later.

Cost and latency live here rather than in a metrics backend because they are
attributes of a run. Aggregate dashboards are downstream of this table; the
table is the record.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ledgerline.agent.state import AgentState
from shared.logging import get_logger

log = get_logger(__name__)


def save_run(
    conn: Any,
    state: AgentState,
    *,
    run_id: str | None = None,
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
    prompt_version: str | None = None,
    trace_id: str | None = None,
) -> str:
    """Persist one run. Returns its id. Caller owns the transaction."""
    identifier = run_id or str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO ledgerline.runs
            (id, question, cik, answer, citations, state, outcome, prompt_version,
             model, input_tokens, output_tokens, cost_usd, latency_ms, trace_id)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """,
        (
            identifier,
            state.get("question", ""),
            state.get("cik"),
            state.get("answer"),
            json.dumps(state.get("citations", [])),
            json.dumps(_jsonable(state)),
            state.get("outcome"),
            prompt_version,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
            latency_ms,
            trace_id,
        ),
    )
    log.info("agent.run.saved", run_id=identifier, outcome=state.get("outcome"))
    return identifier


def _jsonable(state: AgentState) -> dict:
    """Drop anything that will not survive a round trip through JSON.

    A state that cannot be serialised is a run that cannot be replayed, so this
    is checked rather than hoped for: an unserialisable value is replaced by
    its repr instead of raising, because losing one field is better than losing
    the entire record of a run that already went wrong.
    """
    out: dict[str, Any] = {}
    for key, value in state.items():
        try:
            json.dumps(value)
            out[key] = value
        except (TypeError, ValueError):
            out[key] = repr(value)
    return out


def load_run(conn: Any, run_id: str) -> dict | None:
    row = conn.execute(
        """
        SELECT id::text, question, answer, outcome, citations, state, cost_usd, latency_ms
        FROM ledgerline.runs WHERE id = %s
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    keys = (
        "id", "question", "answer", "outcome", "citations", "state", "cost_usd", "latency_ms",
    )
    return dict(zip(keys, row, strict=True))


def outcome_counts(conn: Any) -> dict[str, int]:
    """Terminal outcomes across all stored runs.

    The first thing to look at during an incident: a spike in `degraded` and a
    flat `refused` means infrastructure, the reverse means the corpus or the
    questions changed.
    """
    rows = conn.execute(
        "SELECT outcome, count(*) FROM ledgerline.runs GROUP BY outcome"
    ).fetchall()
    return {str(outcome): int(count) for outcome, count in rows}
