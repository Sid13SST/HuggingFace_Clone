"""The LangGraph wiring.

    plan ──▶ retrieve ──┬─▶ table_analyst ─────┐
                        ├─▶ narrative_analyst ─┼─▶ reconcile ──▶ finalize ──▶ END
                        └─▶ (both, cross-modal)┘

Why a graph rather than a loop: the routing decision changes which analysts
run, the analysts are independent, and the reconciliation step only means
anything when both produced output. That is a dependency structure, and writing
it as one is what makes the run replayable and the failure attributable to a
node instead of to "the agent".

Checkpointing is on from the first commit rather than added later, because the
`degraded` outcome is only useful if you can see the state at the step that
degraded. A checkpointer bolted on afterwards always ends up recording the
final state, which is precisely the one that tells you nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ledgerline.agent.llm import LanguageModel
from ledgerline.agent.nodes import (
    finalize_node,
    make_narrative_node,
    make_retrieve_node,
    make_table_node,
    plan_node,
    reconcile_node,
)
from ledgerline.agent.state import AgentState, Outcome, Route, initial_state
from ledgerline.tables import TableStore
from shared.logging import get_logger

log = get_logger(__name__)


def _next_after_retrieve(state: AgentState) -> list[str]:
    """Fan out to the analysts this route's plan calls for.

    Returning a list makes LangGraph run them as parallel branches, which is
    the honest encoding: the table analyst and the narrative analyst do not
    depend on each other, and pretending they are sequential would hide that
    the cross-modal path could halve its latency.
    """
    return list(state.get("plan") or ["narrative_analyst"])


@dataclass
class LedgerlineAgent:
    """A compiled graph plus the pieces it was built from.

    Holds its dependencies rather than reaching for globals so that a test can
    build one with a fake retriever and a scripted model, and production can
    build one with Postgres and Claude, without the graph knowing which.
    """

    graph: Any
    checkpointer: Any

    @classmethod
    def build(
        cls,
        retriever: Any,
        corpus: dict[str, dict],
        table_store: TableStore,
        model: LanguageModel | None = None,
        *,
        k: int = 5,
        checkpointer: Any | None = None,
    ) -> LedgerlineAgent:
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - needs the extra
            raise ImportError(
                'langgraph is not installed. `pip install -e ".[ledgerline]"`.'
            ) from exc

        builder = StateGraph(AgentState)
        builder.add_node("plan", plan_node)
        builder.add_node("retrieve", make_retrieve_node(retriever, corpus, k=k))
        builder.add_node("table_analyst", make_table_node(table_store))
        builder.add_node("narrative_analyst", make_narrative_node(model))
        builder.add_node("reconcile", reconcile_node)
        builder.add_node("finalize", finalize_node)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "retrieve")
        builder.add_conditional_edges(
            "retrieve", _next_after_retrieve, ["table_analyst", "narrative_analyst"]
        )
        builder.add_edge("table_analyst", "reconcile")
        builder.add_edge("narrative_analyst", "reconcile")
        builder.add_edge("reconcile", "finalize")
        builder.add_edge("finalize", END)

        saver = checkpointer or InMemorySaver()
        return cls(graph=builder.compile(checkpointer=saver), checkpointer=saver)

    def run(
        self, question: str, cik: str | None = None, thread_id: str | None = None
    ) -> AgentState:
        """Run one question to a terminal state. Never raises.

        A caller of this system is answering a user; handing them an exception
        instead of an outcome moves the problem rather than solving it. Even a
        bug in the graph itself comes back as `degraded` with the reason
        attached, and the checkpoint holds the state it died at.
        """
        import uuid

        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
        try:
            final = self.graph.invoke(initial_state(question, cik), config)
        except Exception as exc:  # noqa: BLE001 - the whole point of this method
            log.error("agent.run.failed", question=question[:80], error=str(exc))
            state = initial_state(question, cik)
            state["outcome"] = Outcome.DEGRADED.value
            state["degraded_reasons"] = [f"graph raised: {type(exc).__name__}: {exc}"]
            return state

        log.info(
            "agent.run",
            route=final.get("route"),
            outcome=final.get("outcome"),
            chunks=len(final.get("chunk_ids", [])),
        )
        return final

    def history(self, thread_id: str) -> list[AgentState]:
        """Every checkpointed state for a run, oldest first.

        This is the replay surface. `runs.state` in Postgres holds the final
        state for auditing; this holds the intermediate ones for debugging, and
        the two answer different questions.
        """
        config = {"configurable": {"thread_id": thread_id}}
        snapshots = list(self.graph.get_state_history(config))
        return [snapshot.values for snapshot in reversed(snapshots)]


def route_of(state: AgentState) -> Route:
    return Route(state["route"])
