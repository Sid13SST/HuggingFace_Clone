"""Storing agent runs, against a real database.

Skips without one and runs in CI's schema job, same arrangement as the parity
tests. What is checked here is narrow but load-bearing: that a run written to
`ledgerline.runs` can be read back and is the same run. A replay surface that
loses a field is worse than none, because it invites you to trust it.
"""

from __future__ import annotations

import uuid

import pytest

from ledgerline.agent.persistence import load_run, outcome_counts, save_run
from ledgerline.agent.state import Outcome, initial_state


@pytest.fixture
def clean(db):
    """A database with no runs in it, restored afterwards.

    Scoped per-test rather than per-module: `outcome_counts` reads the whole
    table, so a leftover row from a neighbouring test would make its assertion
    depend on execution order.
    """
    db.execute("DELETE FROM ledgerline.runs")
    db.commit()
    yield db
    db.execute("DELETE FROM ledgerline.runs")
    db.commit()


def _answered_state() -> dict:
    state = initial_state("What was net revenue in fiscal 2025?", cik="0000000000")
    state["route"] = "numeric"
    state["outcome"] = Outcome.ANSWERED.value
    state["answer"] = "1842600000"
    state["citations"] = [{"chunk_id": "t-income", "kind": "table", "quote": "cell"}]
    state["steps"] = ["plan", "retrieve", "table_analyst", "reconcile", "finalize"]
    return state


class TestSaveRun:
    def test_round_trips_a_run(self, clean):
        run_id = save_run(clean, _answered_state(), cost_usd=0.0123, latency_ms=42)
        clean.commit()

        stored = load_run(clean, run_id)
        assert stored is not None
        assert stored["outcome"] == Outcome.ANSWERED.value
        assert stored["answer"] == "1842600000"
        assert stored["latency_ms"] == 42
        assert float(stored["cost_usd"]) == pytest.approx(0.0123)

    def test_stores_the_whole_state_not_a_summary(self, clean):
        """The point of the column. During an incident nobody yet knows which
        field mattered, so the summary is always written by the wrong person."""
        run_id = save_run(clean, _answered_state())
        clean.commit()

        state = load_run(clean, run_id)["state"]
        assert state["steps"][-1] == "finalize"
        assert state["route"] == "numeric"

    def test_degraded_runs_are_stored_too(self, clean):
        """The instinct is to persist successes and log failures, which leaves
        the failures in a stream that rotates away -- exactly the runs worth
        querying later."""
        state = initial_state("Why did gross margin decline?")
        state["outcome"] = Outcome.DEGRADED.value
        state["degraded_reasons"] = ["no language model configured"]
        run_id = save_run(clean, state)
        clean.commit()

        stored = load_run(clean, run_id)
        assert stored["outcome"] == Outcome.DEGRADED.value
        assert stored["state"]["degraded_reasons"] == ["no language model configured"]

    def test_unserialisable_values_lose_a_field_not_the_run(self, clean):
        """A record of a run that already went wrong is worth more than strict
        typing of one field."""
        state = _answered_state()
        state["chunks"] = [{"obj": object()}]
        run_id = save_run(clean, state)
        clean.commit()
        assert load_run(clean, run_id)["outcome"] == Outcome.ANSWERED.value

    def test_saving_twice_does_not_duplicate(self, clean):
        run_id = str(uuid.uuid4())
        save_run(clean, _answered_state(), run_id=run_id)
        save_run(clean, _answered_state(), run_id=run_id)
        clean.commit()
        assert sum(outcome_counts(clean).values()) == 1

    def test_missing_run_is_none_not_an_error(self, clean):
        assert load_run(clean, str(uuid.uuid4())) is None


class TestOutcomeCounts:
    def test_separates_refused_from_degraded(self, clean):
        """The first query during an incident. A spike in degraded with flat
        refused means infrastructure; the reverse means the corpus or the
        questions changed. One combined number answers neither."""
        for outcome in (Outcome.ANSWERED, Outcome.REFUSED, Outcome.REFUSED, Outcome.DEGRADED):
            state = initial_state("q")
            state["outcome"] = outcome.value
            save_run(clean, state)
        clean.commit()

        counts = outcome_counts(clean)
        assert counts[Outcome.REFUSED.value] == 2
        assert counts[Outcome.DEGRADED.value] == 1
        assert counts[Outcome.ANSWERED.value] == 1


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("langgraph") is None,
    reason="langgraph not installed",
)
class TestEndToEnd:
    def test_a_real_run_persists_and_reloads(self, clean):
        from ledgerline.agent.graph import LedgerlineAgent
        from ledgerline.evals import corpus_by_id, reranking_retriever, table_store

        agent = LedgerlineAgent.build(
            reranking_retriever(), corpus_by_id(), table_store(), model=None
        )
        state = agent.run("What was net revenue in fiscal 2025?")
        run_id = save_run(clean, state)
        clean.commit()

        stored = load_run(clean, run_id)
        assert stored["outcome"] == state["outcome"]
        assert stored["state"]["steps"] == state["steps"]
