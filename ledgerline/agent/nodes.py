"""The graph's nodes.

Each one takes state and returns only the keys it changed, which is what
LangGraph merges. Three rules hold across all of them, and they are the
difference between a graph that can be operated and one that cannot:

1. **A node never raises.** Anything unexpected becomes a `degraded_reason`.
   A graph that can throw has no terminal state for the thing that went wrong,
   and the caller is left holding a traceback instead of an outcome.
2. **A node never invents.** If the evidence is not there, it says so and lets
   `finalize` decide between refusing and degrading.
3. **Every node appends to `steps`.** That list is the replay trace, and it is
   written to `ledgerline.runs` whether the run succeeded or not.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ledgerline.agent.llm import LanguageModel, ModelUnavailable
from ledgerline.agent.router import classify, plan_for
from ledgerline.agent.state import AgentState, Citation, Outcome, Route
from ledgerline.tables import Answer, TableStore, answer_numeric
from shared.logging import get_logger

log = get_logger(__name__)

#: Retriever protocol is structural: anything with .rank(query, k) works, which
#: is how the same graph runs against the offline mirror and against Postgres.
Retriever = Any

NARRATIVE_SYSTEM = (
    "You answer questions about SEC filings and earnings calls using only the "
    "excerpts provided. Quote the sentence that supports your answer. If the "
    "excerpts do not contain the answer, reply exactly INSUFFICIENT and nothing "
    "else. Never estimate a figure that is not stated."
)

#: The literal the narrative analyst must return when the excerpts do not
#: answer the question. A sentinel rather than a judgement call downstream --
#: parsing "I'm not sure" out of prose is how over-claiming gets in.
INSUFFICIENT = "INSUFFICIENT"


def format_figure(value: float) -> str:
    """Render a resolved figure so it can be read back as the same number.

    `f"{value:g}"` turns 1842600000.0 into "1.8426e+09", which this project's
    own `parse_number` reads as 1.8426 -- the exponent is parsed as a unit
    suffix. Every consumer of the answer string was therefore off by nine
    orders of magnitude: the eval scored it wrong, and the contradiction
    checker would have compared against it happily.

    Plain decimal, trailing zeros trimmed. An answer a machine cannot re-read
    is not an answer.
    """
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return str(value)
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    return rendered or "0"


def plan_node(state: AgentState) -> AgentState:
    route = classify(state["question"])
    return AgentState(
        route=route.value,
        plan=plan_for(route),
        steps=[*state.get("steps", []), "plan"],
    )


def make_retrieve_node(
    retriever: Retriever, corpus: dict[str, dict], k: int = 5
) -> Callable[[AgentState], AgentState]:
    """Retrieval node bound to a retriever and the corpus it ranks over."""

    def retrieve_node(state: AgentState) -> AgentState:
        steps = [*state.get("steps", []), "retrieve"]
        try:
            ranked = retriever.rank(state["question"], k=k)
        except Exception as exc:  # noqa: BLE001 - a broken index degrades, never crashes
            log.warning("agent.retrieve.failed", error=str(exc))
            return AgentState(
                chunk_ids=[],
                chunks=[],
                degraded_reasons=[
                    *state.get("degraded_reasons", []),
                    f"retrieval failed: {type(exc).__name__}",
                ],
                steps=steps,
            )

        chunks = [
            {
                "chunk_id": chunk_id,
                "kind": corpus[chunk_id].get("kind", "filing"),
                "section": corpus[chunk_id].get("section"),
                "text": corpus[chunk_id]["text"],
            }
            for chunk_id in ranked
            if chunk_id in corpus
        ]
        reasons = list(state.get("degraded_reasons", []))
        if not chunks:
            reasons.append("retrieval returned nothing")
        return AgentState(
            chunk_ids=[c["chunk_id"] for c in chunks],
            chunks=chunks,
            degraded_reasons=reasons,
            steps=steps,
        )

    return retrieve_node


def make_table_node(store: TableStore) -> Callable[[AgentState], AgentState]:
    """The numeric analyst: resolve a figure to a cell, or decline."""

    def table_node(state: AgentState) -> AgentState:
        steps = [*state.get("steps", []), "table_analyst"]
        try:
            result = answer_numeric(state["question"], store)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.table.failed", error=str(exc))
            return AgentState(
                numeric_value=None,
                numeric_declined_reason=None,
                degraded_reasons=[
                    *state.get("degraded_reasons", []),
                    f"table analyst failed: {type(exc).__name__}",
                ],
                steps=steps,
            )

        if isinstance(result, Answer):
            return AgentState(
                numeric_value=result.value,
                numeric_declined_reason=None,
                citations=[
                    *state.get("citations", []),
                    Citation(
                        chunk_id=result.table_id,
                        kind="table",
                        section=result.row_label,
                        quote=result.citation(),
                    ),
                ],
                steps=steps,
            )
        return AgentState(
            numeric_value=None,
            numeric_declined_reason=result.reason,
            steps=steps,
        )

    return table_node


def make_narrative_node(
    model: LanguageModel | None,
) -> Callable[[AgentState], AgentState]:
    """The narrative analyst. Degrades rather than guessing when unavailable.

    `model=None` is a supported configuration, not an error: the deterministic
    half of this system is useful on its own, and a graph that refuses to start
    without an API key cannot be run in CI.
    """

    def narrative_node(state: AgentState) -> AgentState:
        steps = [*state.get("steps", []), "narrative_analyst"]
        chunks = state.get("chunks", [])

        if model is None:
            return AgentState(
                narrative_text=None,
                degraded_reasons=[
                    *state.get("degraded_reasons", []),
                    "no language model configured",
                ],
                steps=steps,
            )
        if not chunks:
            # Nothing to read. Calling the model anyway would invite it to
            # answer from parametric memory, which is the failure this whole
            # architecture exists to prevent.
            return AgentState(narrative_text=None, steps=steps)

        excerpts = "\n\n".join(
            f"[{c['chunk_id']}] ({c['kind']}, {c.get('section') or 'n/a'})\n{c['text']}"
            for c in chunks
        )
        prompt = f"Excerpts:\n\n{excerpts}\n\nQuestion: {state['question']}"

        try:
            completion = model.complete(prompt, system=NARRATIVE_SYSTEM)
        except ModelUnavailable as exc:
            return AgentState(
                narrative_text=None,
                degraded_reasons=[
                    *state.get("degraded_reasons", []),
                    f"model unavailable: {exc}",
                ],
                steps=steps,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.narrative.failed", error=str(exc))
            return AgentState(
                narrative_text=None,
                degraded_reasons=[
                    *state.get("degraded_reasons", []),
                    f"narrative analyst failed: {type(exc).__name__}",
                ],
                steps=steps,
            )

        if completion.refusal_category:
            return AgentState(
                narrative_text=None,
                degraded_reasons=[
                    *state.get("degraded_reasons", []),
                    f"provider refused: {completion.refusal_category}",
                ],
                steps=steps,
            )

        text = completion.text.strip()
        if not text or text.upper().startswith(INSUFFICIENT):
            return AgentState(narrative_text=None, steps=steps)

        cited = [
            Citation(
                chunk_id=c["chunk_id"],
                kind=c["kind"],
                section=c.get("section"),
                quote=c["text"][:200],
            )
            for c in chunks
            if c["chunk_id"] in text
        ]
        return AgentState(
            narrative_text=text,
            citations=[*state.get("citations", []), *cited],
            steps=steps,
        )

    return narrative_node


def reconcile_node(state: AgentState) -> AgentState:
    """Look for a table figure the narrative text disagrees with.

    Only runs with both analysts' output, which is why cross-modal questions
    fan out to both. The check is deliberately narrow -- it flags a figure
    stated in prose that does not match the resolved cell -- because a
    contradiction detector that fires on paraphrase is one nobody reads.
    """
    steps = [*state.get("steps", []), "reconcile"]
    value, text = state.get("numeric_value"), state.get("narrative_text")
    if value is None or not text:
        return AgentState(steps=steps)

    from shared.evals.metrics import extract_numbers, numeric_match

    stated = extract_numbers(text)
    if stated and not any(numeric_match(candidate, value) for candidate in stated):
        return AgentState(
            contradictions=[
                *state.get("contradictions", []),
                f"table cell resolves to {format_figure(value)}; narrative states "
                f"{', '.join(format_figure(n) for n in stated[:3])}",
            ],
            steps=steps,
        )
    return AgentState(steps=steps)


def finalize_node(state: AgentState) -> AgentState:
    """Choose the terminal outcome. The only node that may set `outcome`.

    Order matters and encodes a policy:

    * Degradation beats everything. If a required analyst never ran, whatever
      the other one produced is a partial answer being passed off as a whole
      one.
    * An unresolved contradiction degrades rather than answers. Two sources
      disagree and this system has no basis for picking a winner; saying so is
      the honest output.
    * Having nothing to say is a refusal, not a degradation -- the machinery
      worked and the evidence was insufficient.
    """
    steps = [*state.get("steps", []), "finalize"]
    reasons = list(state.get("degraded_reasons", []))
    contradictions = state.get("contradictions", [])

    if reasons:
        return AgentState(
            outcome=Outcome.DEGRADED.value, answer=None, degraded_reasons=reasons, steps=steps
        )
    if contradictions:
        return AgentState(
            outcome=Outcome.DEGRADED.value,
            answer=None,
            degraded_reasons=[*reasons, *contradictions],
            steps=steps,
        )

    value, text = state.get("numeric_value"), state.get("narrative_text")
    if value is None and not text:
        return AgentState(outcome=Outcome.REFUSED.value, answer=None, steps=steps)

    route = state.get("route")
    if route == Route.NUMERIC.value and value is not None:
        answer = format_figure(value)
    elif value is not None and text:
        answer = f"{text}\n\nResolved figure: {format_figure(value)}"
    else:
        answer = text if text else format_figure(value)

    return AgentState(outcome=Outcome.ANSWERED.value, answer=answer, steps=steps)
