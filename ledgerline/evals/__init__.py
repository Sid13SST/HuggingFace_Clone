"""Ledgerline eval suites.

Imported for its side effect: registering suites with the shared registry.

What is committed here is the *baseline* system -- BM25 retrieval and a naive
"first number in the top chunk" extractor. That is the pipeline a plain RAG
tutorial produces, and it is deliberately the thing the real system has to
beat. Every later improvement (rerank, table analyst, verifier loop) gets
justified as a delta against these numbers.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

from ledgerline.retrieval.bm25 import BM25Index
from ledgerline.retrieval.hybrid import HybridRetriever
from ledgerline.retrieval.rerank import CachedReranker, RerankingRetriever
from ledgerline.tables import Answer, TableStore, answer_numeric
from shared.config import REPO_ROOT
from shared.embeddings import CachedEmbedder
from shared.evals.dataset import Example
from shared.evals.metrics import (
    binary_prf,
    mean,
    ndcg_at_k,
    numeric_match,
    recall_at_k,
    reciprocal_rank,
)
from shared.evals.registry import Gate, Suite, register_suite

HERE = REPO_ROOT / "ledgerline" / "evals"
CORPUS_PATH = HERE / "fixtures" / "corpus.jsonl"

# Slices reported alongside the headline number. Chosen because each one has a
# different failure mode: cross-modal breaks when routing breaks, transcript
# breaks when diarization breaks, numeric breaks when chunking splits a figure
# from its label.
REPORTED_TAGS = ("numeric", "narrative", "transcript", "cross-modal")


@lru_cache(maxsize=1)
def load_corpus() -> list[dict]:
    import json

    records: list[dict] = []
    with CORPUS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                records.append(json.loads(stripped))
    return records


@lru_cache(maxsize=1)
def build_index() -> BM25Index:
    return BM25Index.build((r["id"], r["text"]) for r in load_corpus())


@lru_cache(maxsize=1)
def corpus_by_id() -> dict[str, dict]:
    return {r["id"]: r for r in load_corpus()}


EMBEDDING_CACHE_PATH = HERE / "fixtures" / "embeddings.npz"


def texts_to_embed() -> list[str]:
    """Everything the offline suite will ever ask the embedder to encode.

    Corpus chunks plus every golden-set question. Anything outside this set is
    a cache miss, which is deliberately fatal rather than silently re-encoded
    with a different model.
    """
    import json

    texts = [r["text"] for r in load_corpus()]
    for name in ("retrieval.jsonl", "numeric.jsonl"):
        with (HERE / "datasets" / name).open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    question = json.loads(stripped)["inputs"].get("question")
                    if question:
                        texts.append(question)
    return texts


@lru_cache(maxsize=1)
def embedder() -> CachedEmbedder:
    return CachedEmbedder.from_npz(EMBEDDING_CACHE_PATH)


@lru_cache(maxsize=1)
def hybrid_retriever() -> HybridRetriever:
    return HybridRetriever.build(
        [(r["id"], r["text"]) for r in load_corpus()], embedder()
    )


RERANK_CACHE_PATH = HERE / "fixtures" / "rerank.npz"


def rerank_pairs() -> list[tuple[str, str]]:
    """Every (question, document) pair the offline suite can score.

    The full cross product rather than only the pairs the current retriever
    happens to shortlist -- otherwise tuning candidate_k would produce a cache
    miss instead of a measurement, and the knob would be untunable for exactly
    the reason DedupeConfig had to be refactored.
    """
    documents = [r["text"] for r in load_corpus()]
    questions = [q for q in texts_to_embed() if q not in set(documents)]
    return [(q, d) for q in questions for d in documents]


@lru_cache(maxsize=1)
def reranker() -> CachedReranker:
    return CachedReranker.from_npz(RERANK_CACHE_PATH)


@lru_cache(maxsize=1)
def reranking_retriever() -> RerankingRetriever:
    return RerankingRetriever.build(
        [(r["id"], r["text"]) for r in load_corpus()], hybrid_retriever(), reranker()
    )


# --------------------------------------------------------------------------
# suite: retrieval
# --------------------------------------------------------------------------


def _score_retrieval(
    examples: list[Example], rank: Callable[[str], list[str]]
) -> dict[str, float]:
    """Shared scoring, so BM25 and hybrid are directly comparable.

    Same golden set, same metrics, same slices -- only the ranker varies.
    That is what makes the difference an ablation rather than an anecdote.
    """
    per_example: list[tuple[Example, list[str]]] = [
        (e, rank(e.inputs["question"])) for e in examples
    ]

    metrics: dict[str, float] = {
        "recall@5": mean(
            recall_at_k(e.expected["relevant"], ranked, 5) for e, ranked in per_example
        ),
        "recall@10": mean(
            recall_at_k(e.expected["relevant"], ranked, 10) for e, ranked in per_example
        ),
        "ndcg@10": mean(
            ndcg_at_k(e.expected["relevant"], ranked, 10) for e, ranked in per_example
        ),
        "mrr": mean(
            reciprocal_rank(e.expected["relevant"], ranked) for e, ranked in per_example
        ),
    }

    for tag in REPORTED_TAGS:
        subset = [(e, r) for e, r in per_example if tag in e.tags]
        if subset:
            metrics[f"ndcg@10.{tag}"] = mean(
                ndcg_at_k(e.expected["relevant"], ranked, 10) for e, ranked in subset
            )
    return metrics


def run_retrieval(examples: list[Example]) -> dict[str, float]:
    """The system under test: hybrid retrieval, cross-encoder reranked."""
    retriever = reranking_retriever()
    return _score_retrieval(examples, lambda q: retriever.rank(q, k=10))


def run_retrieval_hybrid(examples: list[Example]) -> dict[str, float]:
    """Hybrid without reranking. The middle rung of the ablation."""
    retriever = hybrid_retriever()
    return _score_retrieval(examples, lambda q: retriever.rank(q, k=10))


def run_retrieval_bm25(examples: list[Example]) -> dict[str, float]:
    """The lexical-only control, kept in CI forever."""
    index = build_index()
    return _score_retrieval(examples, lambda q: index.rank(q, k=10))


# --------------------------------------------------------------------------
# suite: numeric answers
# --------------------------------------------------------------------------

_FIRST_NUMBER = re.compile(r"[$€£]?\s*\d[\d,]*(?:\.\d+)?\s*(?:%|million|billion|thousand)?")


def naive_numeric_answer(question: str) -> str | None:
    """The straw man, stated honestly.

    Retrieve one chunk, return the first number in it, never refuse. This is
    what "just do RAG over the 10-K" amounts to once the tables have been
    flattened into prose, and its failure profile -- confident, plausible,
    wrong, and never abstaining -- is the thing worth measuring against.
    """
    index = build_index()
    ranked = index.rank(question, k=1)
    if not ranked:
        return None
    match = _FIRST_NUMBER.search(corpus_by_id()[ranked[0]]["text"])
    return match.group(0).strip() if match else None


@lru_cache(maxsize=1)
def table_store() -> TableStore:
    return TableStore.from_jsonl(HERE / "fixtures" / "tables.jsonl")


def table_numeric_answer(question: str) -> float | None:
    """Resolve a figure to a cell, or return None to decline."""
    result = answer_numeric(question, table_store())
    return result.value if isinstance(result, Answer) else None


def _score_numeric(
    examples: list[Example], predict: Callable[[str], float | str | None]
) -> dict[str, float]:
    """Shared scoring so the baseline and the table analyst are directly comparable.

    Same golden set, same metrics, same refusal accounting -- the only thing
    that varies between the two suites is the function being graded, which is
    what makes the delta an ablation rather than an anecdote.
    """
    answerable = [e for e in examples if not e.expected.get("not_disclosed")]
    unanswerable = [e for e in examples if e.expected.get("not_disclosed")]

    predictions = {e.id: predict(e.inputs["question"]) for e in examples}

    hits = [
        numeric_match(
            predictions[e.id],
            e.expected["value"],
            scale_hint=float(e.expected.get("scale_hint", 1)),
        )
        for e in answerable
    ]

    # Refusal is a first-class outcome: predicting nothing on an undisclosed
    # figure is correct behaviour, and a system that never abstains should be
    # penalised for it rather than quietly scored on the answerable subset.
    y_true = [True] * len(unanswerable) + [False] * len(answerable)
    y_pred = [predictions[e.id] is None for e in (*unanswerable, *answerable)]
    refusal = binary_prf(y_true, y_pred)

    metrics = {
        "exact_match": mean(1.0 if h else 0.0 for h in hits),
        "answerable_n": float(len(answerable)),
        **refusal.as_dict(prefix="refusal_"),
    }

    distractors = [
        (e, h)
        for e, h in zip(answerable, hits, strict=True)
        if "distractor-heavy" in e.tags
    ]
    if distractors:
        metrics["exact_match.distractor_heavy"] = mean(
            1.0 if h else 0.0 for _, h in distractors
        )
    return metrics


def run_numeric(examples: list[Example]) -> dict[str, float]:
    """The system under test: figures resolved to table cells."""
    return _score_numeric(examples, table_numeric_answer)


def run_numeric_baseline(examples: list[Example]) -> dict[str, float]:
    """The straw man, kept in CI forever.

    A "before" number that only exists in a README decays the moment someone
    edits the golden set. Re-measuring it every run means the ablation in the
    README is always true.
    """
    return _score_numeric(examples, naive_numeric_answer)


register_suite(
    Suite(
        name="ledgerline.retrieval",
        project="ledgerline",
        dataset=HERE / "datasets" / "retrieval.jsonl",
        run=run_retrieval,
        description="Hybrid RRF over BM25 + dense, rescored by a cross-encoder.",
        gates=[
            Gate("ndcg@10", min_value=0.90, max_regression=0.02),
            Gate("recall@10", min_value=0.90, max_regression=0.02),
            # The slice dense retrieval failed to fix and reranking did. Gating
            # the headline alone would let it regress behind a healthy average.
            Gate("ndcg@10.narrative", min_value=0.80, max_regression=0.03),
        ],
    )
)

register_suite(
    Suite(
        name="ledgerline.retrieval_hybrid",
        project="ledgerline",
        dataset=HERE / "datasets" / "retrieval.jsonl",
        run=run_retrieval_hybrid,
        description="Hybrid without reranking. Middle rung of the retrieval ablation.",
        gates=[],
    )
)

register_suite(
    Suite(
        name="ledgerline.retrieval_bm25",
        project="ledgerline",
        dataset=HERE / "datasets" / "retrieval.jsonl",
        run=run_retrieval_bm25,
        description="Lexical-only control. The permanent 'before' for the hybrid ablation.",
        gates=[],
    )
)

register_suite(
    Suite(
        name="ledgerline.numeric",
        project="ledgerline",
        dataset=HERE / "datasets" / "numeric.jsonl",
        run=run_numeric,
        description="Figures resolved to table cells, with declining as a first-class outcome.",
        gates=[
            Gate("exact_match", min_value=0.75, max_regression=0.01),
            # The floor the naive extractor could never clear: it never
            # abstained, so it scored 0 here by construction.
            Gate("refusal_recall", min_value=0.90, max_regression=0.05),
        ],
    )
)

register_suite(
    Suite(
        name="ledgerline.numeric_baseline",
        project="ledgerline",
        dataset=HERE / "datasets" / "numeric.jsonl",
        run=run_numeric_baseline,
        description="Naive first-number-in-top-chunk extractor. The permanent 'before'.",
        # No gates. This suite is not supposed to improve -- it is the control,
        # and gating it would create pressure to quietly make the straw man
        # stronger so the ablation looks better.
        gates=[],
    )
)


# Registered last, and imported for the side effect rather than for a name: the
# agent suites depend on the retrievers and table store defined above, so this
# import cannot sit at the top of the file without a cycle.
from ledgerline.evals import agent_suites  # noqa: E402,F401  isort: skip
