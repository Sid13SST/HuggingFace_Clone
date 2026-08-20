"""The agent graph: routing, nodes, terminal states, and the LLM seam."""

from __future__ import annotations

import json

import pytest

from ledgerline.agent.llm import (
    CachedModel,
    Completion,
    CompletionCacheMiss,
    ModelUnavailable,
    prompt_key,
    save_completion_cache,
)
from ledgerline.agent.nodes import (
    INSUFFICIENT,
    finalize_node,
    format_figure,
    make_narrative_node,
    make_retrieve_node,
    reconcile_node,
)
from ledgerline.agent.router import classify, plan_for
from ledgerline.agent.state import Outcome, Route, initial_state
from shared.evals.metrics import numeric_match, parse_number


class TestFormatFigure:
    def test_large_figures_survive_a_round_trip(self):
        """The bug this function exists for.

        `f"{value:g}"` renders 1842600000.0 as "1.8426e+09", and this project's
        own parser reads the exponent as a unit suffix and returns 1.8426 --
        nine orders of magnitude out, silently, in the string the user sees.
        """
        rendered = format_figure(1842600000.0)
        assert "e" not in rendered
        assert parse_number(rendered) == 1842600000.0

    @pytest.mark.parametrize("value", [0.0, -1234.5, 34.2, 1e12, 0.001])
    def test_round_trips_across_magnitudes(self, value):
        assert numeric_match(format_figure(value), value)

    def test_trims_noise_without_losing_precision(self):
        assert format_figure(34.20) == "34.2"
        assert format_figure(1000.0) == "1000"


class TestRouter:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("How much did net revenue grow in fiscal 2025?", Route.NUMERIC),
            ("How many people does the company employ?", Route.NUMERIC),
            ("What proportion of steel is under supply agreements?", Route.NUMERIC),
            ("What accrual was recorded for the class action?", Route.NUMERIC),
            ("Why did gross margin decline?", Route.NARRATIVE),
            ("What drove the change in operating income?", Route.NARRATIVE),
            ("What did the CFO say about margin pressure?", Route.NARRATIVE),
            ("Does management believe input cost pressure is temporary?", Route.CROSS_MODAL),
            ("Is the guidance consistent with the filing?", Route.CROSS_MODAL),
        ],
    )
    def test_routes_by_question_form(self, question, expected):
        assert classify(question) == expected

    def test_subordinate_why_does_not_hijack_a_figure_question(self):
        """"What was the effective tax rate and why did it change?" is a figure
        question with an explanation attached. Routing it to prose loses the
        number, which is why the why-rule is anchored to the opener."""
        assert classify("What was the effective tax rate and why did it change?") == (
            Route.NUMERIC
        )

    def test_naming_a_source_is_not_comparing_two(self):
        """An early version treated "on the call" as a cross-modal cue, which
        sent every ordinary transcript question down the reconciliation path."""
        assert classify("What did the CFO say about pricing on the call?") == (
            Route.NARRATIVE
        )

    def test_unrecognisable_questions_default_to_narrative(self):
        """Narrative can retrieve, read and decline. The numeric path resolves
        a cell, and a confidently wrong figure is this system's worst output."""
        assert classify("Tell me about the company.") == Route.NARRATIVE

    def test_never_raises_on_junk(self):
        for junk in ("", "   ", "???", "\n\n", "1"):
            assert isinstance(classify(junk), Route)

    def test_cross_modal_runs_both_analysts(self):
        """You cannot detect a table disagreeing with a transcript using one
        of them."""
        assert plan_for(Route.CROSS_MODAL) == ["table_analyst", "narrative_analyst"]
        assert plan_for(Route.NUMERIC) == ["table_analyst"]


class TestPromptKey:
    def test_model_is_part_of_the_key(self):
        """Otherwise switching models silently reuses the old model's answers
        and the ablation measures nothing."""
        assert prompt_key("claude-opus-5", None, "q") != prompt_key("other", None, "q")

    def test_system_prompt_is_part_of_the_key(self):
        assert prompt_key("m", "be terse", "q") != prompt_key("m", "be verbose", "q")

    def test_whitespace_insensitive_on_the_prompt(self):
        assert prompt_key("m", None, "a  b\nc") == prompt_key("m", None, "a b c")


class TestCompletionCost:
    def test_prices_input_and_output_separately(self):
        completion = Completion(text="x", input_tokens=1_000_000, output_tokens=1_000_000)
        assert completion.cost_usd == pytest.approx(30.0)  # $5 in + $25 out

    def test_cache_reads_are_charged_at_a_tenth(self):
        cached = Completion(text="x", cached_input_tokens=1_000_000)
        assert cached.cost_usd == pytest.approx(0.5)

    def test_unknown_model_falls_back_to_default_pricing(self):
        """A cost of zero for an unrecognised model would under-report spend,
        which is the direction that goes unnoticed."""
        assert Completion(text="x", model="mystery", input_tokens=1_000_000).cost_usd > 0


class ScriptedModel:
    """Returns a fixed reply. Enough to exercise the node's plumbing."""

    model_name = "scripted"

    def __init__(self, reply: str = "Margin fell on steel costs [c-mda-margin]."):
        self.reply = reply
        self.calls: list[str] = []

    def complete(self, prompt, *, system=None):
        self.calls.append(prompt)
        return Completion(text=self.reply, model=self.model_name)


class UnavailableModel:
    model_name = "down"

    def complete(self, prompt, *, system=None):
        raise ModelUnavailable("provider 503")


class TestCachedModel:
    @pytest.fixture
    def cache(self, tmp_path):
        return save_completion_cache(
            tmp_path / "c.json", [("hello", "sys")], ScriptedModel("hi")
        )

    def test_round_trips(self, cache):
        assert CachedModel.from_json(cache).complete("hello", system="sys").text == "hi"

    def test_miss_is_fatal_not_silent(self, cache):
        with pytest.raises(CompletionCacheMiss, match="warm-cache"):
            CachedModel.from_json(cache).complete("unseen", system="sys")

    def test_a_different_system_prompt_is_a_miss(self, cache):
        """Same question, different instructions, different answer. Treating
        them as one entry would serve the wrong cached response."""
        with pytest.raises(CompletionCacheMiss):
            CachedModel.from_json(cache).complete("hello", system="other")

    def test_missing_file_names_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="warm-cache"):
            CachedModel.from_json(tmp_path / "absent.json")

    def test_cache_is_readable_json(self, cache):
        """It gets committed and reviewed, so it has to be diffable."""
        assert "responses" in json.loads(cache.read_text(encoding="utf-8"))


class TestNarrativeNode:
    def test_missing_model_degrades_rather_than_refusing(self):
        """The distinction the whole outcome vocabulary exists for: no model is
        something being broken, not evidence being insufficient."""
        state = initial_state("why?")
        state["chunks"] = [{"chunk_id": "c1", "kind": "filing", "text": "x"}]
        result = make_narrative_node(None)(state)
        assert result["degraded_reasons"] == ["no language model configured"]

    def test_unavailable_model_degrades(self):
        state = initial_state("why?")
        state["chunks"] = [{"chunk_id": "c1", "kind": "filing", "text": "x"}]
        result = make_narrative_node(UnavailableModel())(state)
        assert any("model unavailable" in r for r in result["degraded_reasons"])

    def test_no_chunks_means_no_model_call(self):
        """With nothing retrieved, calling the model invites it to answer from
        memory -- the exact failure this architecture exists to prevent."""
        model = ScriptedModel()
        result = make_narrative_node(model)(initial_state("why?"))
        assert model.calls == []
        assert result["narrative_text"] is None

    def test_insufficient_sentinel_is_not_an_answer(self):
        state = initial_state("why?")
        state["chunks"] = [{"chunk_id": "c1", "kind": "filing", "text": "x"}]
        result = make_narrative_node(ScriptedModel(INSUFFICIENT))(state)
        assert result["narrative_text"] is None

    def test_provider_refusal_degrades(self):
        class Refusing:
            model_name = "r"

            def complete(self, prompt, *, system=None):
                return Completion(text="", refusal_category="cyber")

        state = initial_state("why?")
        state["chunks"] = [{"chunk_id": "c1", "kind": "filing", "text": "x"}]
        result = make_narrative_node(Refusing())(state)
        assert any("provider refused" in r for r in result["degraded_reasons"])


class TestRetrieveNode:
    def test_a_broken_index_degrades_instead_of_raising(self):
        class Broken:
            def rank(self, query, k=10):
                raise RuntimeError("index corrupt")

        result = make_retrieve_node(Broken(), {})(initial_state("q"))
        assert any("retrieval failed" in r for r in result["degraded_reasons"])

    def test_empty_results_are_recorded_as_a_reason(self):
        class Empty:
            def rank(self, query, k=10):
                return []

        result = make_retrieve_node(Empty(), {})(initial_state("q"))
        assert result["degraded_reasons"] == ["retrieval returned nothing"]

    def test_ids_missing_from_the_corpus_are_dropped_not_fatal(self):
        class Stale:
            def rank(self, query, k=10):
                return ["gone", "here"]

        corpus = {"here": {"kind": "filing", "text": "t"}}
        result = make_retrieve_node(Stale(), corpus)(initial_state("q"))
        assert result["chunk_ids"] == ["here"]


class TestReconcile:
    def test_flags_a_narrative_figure_that_contradicts_the_cell(self):
        state = initial_state("q")
        state["numeric_value"] = 1842600000.0
        state["narrative_text"] = "Revenue was $1,500.0 million in fiscal 2025."
        assert reconcile_node(state)["contradictions"]

    def test_agreement_produces_nothing(self):
        state = initial_state("q")
        state["numeric_value"] = 1842600000.0
        state["narrative_text"] = "Revenue was $1,842.6 million."
        assert not reconcile_node(state).get("contradictions")

    def test_needs_both_analysts_to_say_anything(self):
        state = initial_state("q")
        state["numeric_value"] = 5.0
        assert not reconcile_node(state).get("contradictions")


class TestFinalize:
    def test_degradation_beats_a_partial_answer(self):
        """Half the plan ran. Presenting the half that worked as the whole
        answer is how a silent wrong answer reaches a user."""
        state = initial_state("q")
        state["numeric_value"] = 42.0
        state["degraded_reasons"] = ["no language model configured"]
        assert finalize_node(state)["outcome"] == Outcome.DEGRADED.value

    def test_unresolved_contradiction_degrades(self):
        state = initial_state("q")
        state["numeric_value"] = 42.0
        state["narrative_text"] = "It was 99."
        state["contradictions"] = ["table says 42; narrative says 99"]
        result = finalize_node(state)
        assert result["outcome"] == Outcome.DEGRADED.value
        assert result["answer"] is None

    def test_nothing_to_say_is_a_refusal_not_a_degradation(self):
        """The machinery worked; the evidence was insufficient. That is a
        correct answer to an unanswerable question."""
        assert finalize_node(initial_state("q"))["outcome"] == Outcome.REFUSED.value

    def test_numeric_answer_is_parseable(self):
        state = initial_state("q")
        state["route"] = Route.NUMERIC.value
        state["numeric_value"] = 1842600000.0
        assert numeric_match(finalize_node(state)["answer"], 1842600000.0)


def _langgraph_installed() -> bool:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def graph():
    from ledgerline.agent.graph import LedgerlineAgent
    from ledgerline.evals import corpus_by_id, reranking_retriever, table_store

    return LedgerlineAgent.build(
        reranking_retriever(), corpus_by_id(), table_store(), model=None
    )


@pytest.mark.skipif(not _langgraph_installed(), reason="langgraph not installed")
class TestGraph:

    def test_numeric_question_is_answered_end_to_end(self, graph):
        state = graph.run("What was net revenue in fiscal 2025?")
        assert state["outcome"] == Outcome.ANSWERED.value
        assert numeric_match(state["answer"], "$1,842.6 million")

    def test_narrative_question_degrades_without_a_model(self, graph):
        state = graph.run("Why did gross margin decline?")
        assert state["outcome"] == Outcome.DEGRADED.value
        assert state["answer"] is None

    def test_every_run_reaches_a_terminal_state(self, graph):
        """A caller is answering a user. Handing them a traceback instead of an
        outcome moves the problem rather than solving it."""
        for question in ("", "?????", "What was net revenue?", "Why?"):
            assert graph.run(question)["outcome"] in {o.value for o in Outcome}

    def test_answered_runs_carry_citations(self, graph):
        state = graph.run("What was net revenue in fiscal 2025?")
        assert state["citations"]

    def test_steps_record_the_path_taken(self, graph):
        state = graph.run("What was net revenue in fiscal 2025?")
        assert state["steps"][0] == "plan"
        assert state["steps"][-1] == "finalize"
        assert "table_analyst" in state["steps"]

    def test_checkpoints_expose_intermediate_states(self, graph):
        """A `degraded` outcome is only useful if you can see the state at the
        step that degraded."""
        graph.run("What was net revenue in fiscal 2025?", thread_id="t-history")
        history = graph.history("t-history")
        assert len(history) > 1

    def test_state_survives_a_json_round_trip(self, graph):
        """It gets written to `ledgerline.runs`. A state that cannot serialise
        is a run that cannot be replayed."""
        state = graph.run("What was net revenue in fiscal 2025?")
        assert json.loads(json.dumps(state))["outcome"] == state["outcome"]
