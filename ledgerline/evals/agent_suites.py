"""Suites for the agent graph: routing, and end-to-end outcomes.

Routing is measured against labels nobody wrote for this purpose. The retrieval
golden set has carried `numeric` / `narrative` / `transcript` / `cross-modal`
tags since before the router existed, assigned to slice retrieval metrics. That
makes them an unusually honest label set: they cannot have been tuned to
flatter the classifier, because the classifier did not exist when they were
written.

The end-to-end suite measures something most agent demos never report -- what
happens when the agent *cannot* answer. All three terminal outcomes are counted
separately, so a system that answers one question perfectly and falls over on
the rest cannot post a good number.

No language model is configured here, which is not a limitation of the suite
but the thing it currently measures: how far the deterministic half gets on its
own. On the numeric golden set that is 73% answered at perfect accuracy and 27%
refused, with nothing degraded and nothing fabricated. When the narrative
analyst lands, the refusals are what should move -- and if `answered_correct`
or `fabrication_rate` moves instead, the model made the system worse and this
suite is where that shows up.
"""

from __future__ import annotations

from functools import lru_cache

from ledgerline.agent.router import classify
from ledgerline.agent.state import Outcome, Route
from shared.evals.dataset import Example
from shared.evals.metrics import cohens_kappa, mean, numeric_match
from shared.evals.registry import Gate, Suite, register_suite

from ledgerline.evals import HERE  # isort: skip


def expected_route(tags: list[str]) -> Route:
    """The route a question's tags imply.

    Precedence, not first-match: a question tagged both `narrative` and
    `cross-modal` is cross-modal, because comparing two sources is the harder
    job and the one that decides which analysts run.
    """
    if "cross-modal" in tags:
        return Route.CROSS_MODAL
    if "numeric" in tags:
        return Route.NUMERIC
    return Route.NARRATIVE


def run_routing(examples: list[Example]) -> dict[str, float]:
    predicted = [classify(e.inputs["question"]) for e in examples]
    expected = [expected_route(list(e.tags)) for e in examples]

    metrics = {
        "accuracy": mean(1.0 if p == g else 0.0 for p, g in zip(predicted, expected, strict=True)),
        # Chance-corrected, because three classes with an uneven prior make raw
        # accuracy flattering: always predicting `numeric` scores 0.6 here.
        "kappa": cohens_kappa([p.value for p in predicted], [g.value for g in expected]),
    }
    for route in Route:
        subset = [
            (p, g) for p, g in zip(predicted, expected, strict=True) if g == route
        ]
        if subset:
            metrics[f"recall.{route.value}"] = mean(
                1.0 if p == g else 0.0 for p, g in subset
            )
    return metrics


def run_routing_heldout(examples: list[Example]) -> dict[str, float]:
    """Routing measured on a set the rules were never revised against.

    Every question in the numeric golden set expects a numeric answer -- that
    is what makes it that golden set -- so the correct route is `numeric` for
    all of them, with no labelling work required. That makes it a free held-out
    slice, and it has already earned its place: a partitive rule tuned to fix
    one question on the retrieval set scored 1.000 there and immediately broke
    "How much of the revolving facility remains available?" here.

    When these two numbers diverge, believe this one.
    """
    predicted = [classify(e.inputs["question"]) for e in examples]
    return {
        "accuracy": mean(1.0 if p == Route.NUMERIC else 0.0 for p in predicted),
        # Which way the errors go matters. Routing a figure question to the
        # narrative analyst wastes a model call and usually degrades; the
        # reverse returns a confident wrong number.
        "leaked_to_narrative": mean(
            1.0 if p == Route.NARRATIVE else 0.0 for p in predicted
        ),
    }


@lru_cache(maxsize=1)
def agent():
    """The graph under test, with no language model.

    Deliberately `model=None` rather than a scripted stand-in. A scripted model
    would make the narrative path *look* exercised while measuring nothing
    about it, and the honest number here is how much of the golden set the
    deterministic half can answer on its own.
    """
    from ledgerline.agent.graph import LedgerlineAgent
    from ledgerline.evals import corpus_by_id, reranking_retriever, table_store

    return LedgerlineAgent.build(
        reranking_retriever(), corpus_by_id(), table_store(), model=None
    )


def run_agent(examples: list[Example]) -> dict[str, float]:
    """End-to-end outcomes on the numeric golden set.

    `answered_correct` is scored only over runs that answered, and
    `answered_rate` is reported next to it, because a system that answers one
    question perfectly and degrades on the rest would otherwise post a perfect
    accuracy.
    """
    graph = agent()
    states = [graph.run(e.inputs["question"]) for e in examples]

    outcomes = [s.get("outcome") for s in states]
    counts = {
        f"{outcome.value}_rate": mean(1.0 if o == outcome.value else 0.0 for o in outcomes)
        for outcome in Outcome
    }

    answered = [
        (e, s)
        for e, s in zip(examples, states, strict=True)
        if s.get("outcome") == Outcome.ANSWERED.value
    ]
    correct = [
        numeric_match(
            s.get("answer"),
            e.expected["value"],
            scale_hint=float(e.expected.get("scale_hint", 1)),
        )
        for e, s in answered
        if not e.expected.get("not_disclosed")
    ]

    # An undisclosed figure must not come back as an answer. This is the
    # over-claiming check, and it is separate from accuracy on purpose: a
    # confident answer to a question the filing never addresses is a worse
    # failure than a wrong figure, because nothing downstream can catch it.
    fabricated = [
        1.0 if s.get("outcome") == Outcome.ANSWERED.value else 0.0
        for e, s in zip(examples, states, strict=True)
        if e.expected.get("not_disclosed")
    ]

    return {
        **counts,
        "answered_correct": mean(1.0 if c else 0.0 for c in correct) if correct else 0.0,
        "fabrication_rate": mean(fabricated) if fabricated else 0.0,
        # Every run must reach a terminal state. A graph that can return
        # without one is a graph whose caller has to guess.
        "terminal_rate": mean(1.0 if o in {o2.value for o2 in Outcome} else 0.0 for o in outcomes),
    }


register_suite(
    Suite(
        name="ledgerline.routing",
        project="ledgerline",
        dataset=HERE / "datasets" / "retrieval.jsonl",
        run=run_routing,
        description="Keyword routing against the golden set's own topic tags.",
        gates=[
            # Set below the achieved 0.933, not at it. This set is the one
            # the rules were revised against, so its number is an upper bound
            # on quality rather than an estimate of it -- gating at the
            # measured value would lock in the overfitting.
            Gate("accuracy", min_value=0.85, max_regression=0.05),
            Gate("kappa", min_value=0.75, max_regression=0.10),
        ],
    )
)

register_suite(
    Suite(
        name="ledgerline.routing_heldout",
        project="ledgerline",
        dataset=HERE / "datasets" / "numeric.jsonl",
        run=run_routing_heldout,
        description="Routing on a set the rules were never revised against.",
        gates=[
            Gate("accuracy", min_value=0.85, max_regression=0.05),
            Gate("leaked_to_narrative", max_value=0.15, higher_is_better=False),
        ],
    )
)

register_suite(
    Suite(
        name="ledgerline.agent",
        project="ledgerline",
        dataset=HERE / "datasets" / "numeric.jsonl",
        run=run_agent,
        description="End-to-end graph outcomes: answered, refused, degraded.",
        gates=[
            # The two that matter regardless of how good the analysts get.
            Gate("terminal_rate", min_value=1.0),
            # Lower is better, so this is a ceiling rather than a floor --
            # and the ceiling is zero. Answering a question the filing
            # never addressed is the one failure nothing downstream can
            # catch, because it arrives with citations attached.
            Gate("fabrication_rate", max_value=0.0, higher_is_better=False),
            Gate("answered_correct", min_value=0.75, max_regression=0.05),
        ],
    )
)
