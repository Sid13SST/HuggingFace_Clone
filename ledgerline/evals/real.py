"""The real-corpus eval surface: Caterpillar's FY2025 10-K.

Everything else under `ledgerline/evals/` scores a 17-chunk synthetic fixture
written to be answerable. This module scores 484 chunks of an actual filing,
and the two are kept side by side on purpose: the synthetic suite says whether
the machinery works, and this one says whether it works on the thing it claims
to be for. When the two disagree, this one is right.

Labels here are anchored by quoted text rather than by chunk id. A golden set
keyed on `cat-2025-12-31-c17` measures the chunker: change the window size and
every label silently points somewhere else, with the suite still reporting a
number. An anchor that no longer resolves raises instead, which is the same
argument the embedding cache makes by treating a miss as fatal.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from shared.config import REPO_ROOT

HERE = REPO_ROOT / "ledgerline" / "evals"
CORPUS_PATH = HERE / "fixtures" / "cat_corpus.jsonl"
TABLES_PATH = HERE / "fixtures" / "cat_tables.jsonl"
RETRIEVAL_PATH = HERE / "datasets" / "retrieval_cat.jsonl"
NUMERIC_PATH = HERE / "datasets" / "numeric_cat.jsonl"
EMBEDDING_CACHE_PATH = HERE / "fixtures" / "cat_embeddings.npz"
RERANK_CACHE_PATH = HERE / "fixtures" / "cat_rerank.npz"

#: The filing these fixtures were exported from. Committed so a re-export
#: against a different filing is a visible diff rather than a quiet swap.
DOCUMENT_ID = "cat-2025-12-31"


class AnchorNotFound(LookupError):
    """A label points at text the corpus no longer contains.

    Raised rather than returned. The alternative -- dropping the label and
    scoring what is left -- reports a number that looks fine while measuring a
    smaller golden set than the one on disk.
    """


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                records.append(json.loads(stripped))
    return records


@lru_cache(maxsize=1)
def load_corpus() -> list[dict]:
    return read_jsonl(CORPUS_PATH)


@lru_cache(maxsize=1)
def corpus_by_id() -> dict[str, dict]:
    return {r["id"]: r for r in load_corpus()}


#: An anchor matching more chunks than this is boilerplate, not a locator.
#: Three rather than one because the chunker overlaps its windows: a sentence
#: near a boundary is genuinely present in two chunks, and retrieving either of
#: them has served the reader. Rejecting that would force every anchor into the
#: interior of a window, which is a labelling rule about the chunker's stride --
#: exactly the coupling anchoring exists to avoid.
MAX_ANCHOR_MATCHES = 3


def resolve_anchor(anchor: str) -> list[str]:
    """Return the ids of the chunks containing `anchor`."""
    hits = [r["id"] for r in load_corpus() if anchor in r["text"]]
    if not hits:
        raise AnchorNotFound(f"no chunk contains {anchor!r}")
    if len(hits) > MAX_ANCHOR_MATCHES:
        raise AnchorNotFound(
            f"{anchor!r} matches {len(hits)} chunks; it locates a topic, not a passage"
        )
    return hits


def relevant_ids(expected: dict) -> list[str]:
    """Resolve a golden-set record's anchors to chunk ids, order preserved."""
    seen: dict[str, None] = {}
    for anchor in expected["relevant"]:
        for chunk_id in resolve_anchor(anchor):
            seen.setdefault(chunk_id, None)
    return list(seen)


# --------------------------------------------------------------------------
# retrievers over the real corpus
# --------------------------------------------------------------------------


def texts_to_embed() -> list[str]:
    """Every text the offline real-corpus suites will ask the embedder for.

    Same contract as the synthetic suite's: anything outside this set is a
    cache miss, and a miss is fatal rather than quietly re-encoded.
    """
    texts = [r["text"] for r in load_corpus()]
    for path in (RETRIEVAL_PATH, NUMERIC_PATH):
        texts.extend(r["inputs"]["question"] for r in read_jsonl(path))
    return texts


@lru_cache(maxsize=1)
def embedder():
    from shared.embeddings import CachedEmbedder

    return CachedEmbedder.from_npz(EMBEDDING_CACHE_PATH)


@lru_cache(maxsize=1)
def hybrid_retriever():
    from ledgerline.retrieval.hybrid import HybridRetriever

    return HybridRetriever.build(
        [(r["id"], r["text"]) for r in load_corpus()], embedder()
    )


@lru_cache(maxsize=1)
def bm25_index():
    from ledgerline.retrieval.bm25 import BM25Index

    return BM25Index.build((r["id"], r["text"]) for r in load_corpus())


def rerank_pairs() -> list[tuple[str, str]]:
    """Question-document pairs the cross-encoder may be asked to score.

    The synthetic suite caches the full cross product so that tuning
    candidate_k stays a measurement rather than becoming a cache miss. Here the
    cross product is 45 x 484 = 21,780 pairs, which is affordable, and keeping
    the same rule means candidate_k is tunable on the real corpus too.
    """
    documents = [r["text"] for r in load_corpus()]
    questions = [r["inputs"]["question"] for r in read_jsonl(RETRIEVAL_PATH)]
    return [(q, d) for q in questions for d in documents]


@lru_cache(maxsize=1)
def reranker():
    from ledgerline.retrieval.rerank import CachedReranker

    return CachedReranker.from_npz(RERANK_CACHE_PATH)


@lru_cache(maxsize=1)
def reranking_retriever():
    from ledgerline.retrieval.rerank import RerankingRetriever

    return RerankingRetriever.build(
        [(r["id"], r["text"]) for r in load_corpus()], hybrid_retriever(), reranker()
    )


@lru_cache(maxsize=1)
def table_store():
    from ledgerline.tables import TableStore

    return TableStore.from_jsonl(TABLES_PATH)


# --------------------------------------------------------------------------
# suite: retrieval on the real corpus
# --------------------------------------------------------------------------

#: Slices reported next to the headline. `glossary-trap` is the one that
#: justifies the whole file: Item 7 closes with 24 defined terms, and those
#: definitions are the densest lexical match in the filing for exactly the
#: vocabulary an analyst question uses. The synthetic corpus contains no such
#: passage, so no amount of tuning against it could have surfaced this.
REPORTED_TAGS = ("numeric", "narrative", "risk", "outlook", "segment", "glossary-trap")


def _score_retrieval(examples, rank) -> dict[str, float]:
    from shared.evals.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank

    scored = [
        (e, relevant_ids(e.expected), rank(e.inputs["question"])) for e in examples
    ]

    metrics = {
        "recall@5": mean(recall_at_k(rel, ranked, 5) for _, rel, ranked in scored),
        "recall@10": mean(recall_at_k(rel, ranked, 10) for _, rel, ranked in scored),
        "ndcg@10": mean(ndcg_at_k(rel, ranked, 10) for _, rel, ranked in scored),
        "mrr": mean(reciprocal_rank(rel, ranked) for _, rel, ranked in scored),
    }
    for tag in REPORTED_TAGS:
        subset = [(rel, ranked) for e, rel, ranked in scored if tag in e.tags]
        if subset:
            metrics[f"ndcg@10.{tag}"] = mean(
                ndcg_at_k(rel, ranked, 10) for rel, ranked in subset
            )
    return metrics


def run_retrieval(examples) -> dict[str, float]:
    """The system under test: hybrid RRF, cross-encoder reranked."""
    retriever = reranking_retriever()
    return _score_retrieval(examples, lambda q: retriever.rank(q, k=10))


def run_retrieval_hybrid(examples) -> dict[str, float]:
    return _score_retrieval(examples, lambda q: hybrid_retriever().rank(q, k=10))


def run_retrieval_bm25(examples) -> dict[str, float]:
    return _score_retrieval(examples, lambda q: bm25_index().rank(q, k=10))


# --------------------------------------------------------------------------
# suite: numeric answers on the real corpus
# --------------------------------------------------------------------------


def table_numeric_answer(question: str):
    from ledgerline.tables import Answer, answer_numeric

    result = answer_numeric(question, table_store())
    return result.value if isinstance(result, Answer) else None


def naive_numeric_answer(question: str):
    """The same straw man as the synthetic suite, on a real filing.

    Retrieve one chunk, return the first number in it, never decline. On 484
    chunks of boilerplate this is not a weaker version of the real system; it
    is a different failure mode, and worth measuring for that reason alone.
    """
    import re

    ranked = bm25_index().rank(question, k=1)
    if not ranked:
        return None
    pattern = re.compile(r"[$]?\s*\d[\d,]*(?:\.\d+)?\s*(?:%|million|billion|thousand)?")
    match = pattern.search(corpus_by_id()[ranked[0]]["text"])
    return match.group(0).strip() if match else None


def _score_numeric(examples, predict) -> dict[str, float]:
    from shared.evals.metrics import binary_prf, mean, numeric_match

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

    y_true = [True] * len(unanswerable) + [False] * len(answerable)
    y_pred = [predictions[e.id] is None for e in (*unanswerable, *answerable)]
    refusal = binary_prf(y_true, y_pred)

    metrics = {
        "exact_match": mean(1.0 if h else 0.0 for h in hits),
        "answerable_n": float(len(answerable)),
        **refusal.as_dict(prefix="refusal_"),
    }

    # The figure exists in the filing and the parser could not reach it.
    # Declining is right, so this never shows up as a refusal error -- which is
    # precisely why it needs its own number. Without it, raising the extraction
    # rate looks like work with no metric attached to it.
    gap = [e for e in unanswerable if e.expected.get("reason") == "not-extracted"]
    if examples:
        metrics["coverage_gap"] = len(gap) / len(examples)

    for tag in ("unit-trap", "distractor-heavy"):
        subset = [h for e, h in zip(answerable, hits, strict=True) if tag in e.tags]
        if subset:
            metrics[f"exact_match.{tag.replace('-', '_')}"] = mean(
                1.0 if h else 0.0 for h in subset
            )
    return metrics


def run_numeric(examples) -> dict[str, float]:
    return _score_numeric(examples, table_numeric_answer)


def run_numeric_baseline(examples) -> dict[str, float]:
    return _score_numeric(examples, naive_numeric_answer)


def register() -> None:
    """Register the real-corpus suites.

    All ungated on purpose, and staying that way until the numbers have been
    published once. A gate written at the same time as its first measurement
    is a gate set wherever the code happened to land, which is a rubber stamp
    with a threshold in it.
    """
    from shared.evals.registry import Suite, register_suite

    register_suite(
        Suite(
            name="ledgerline.retrieval_cat",
            project="ledgerline",
            dataset=RETRIEVAL_PATH,
            run=run_retrieval,
            description="Reranked hybrid retrieval over a real 10-K.",
            gates=[],
        )
    )
    register_suite(
        Suite(
            name="ledgerline.retrieval_cat_hybrid",
            project="ledgerline",
            dataset=RETRIEVAL_PATH,
            run=run_retrieval_hybrid,
            description="Hybrid without reranking, real corpus.",
            gates=[],
        )
    )
    register_suite(
        Suite(
            name="ledgerline.retrieval_cat_bm25",
            project="ledgerline",
            dataset=RETRIEVAL_PATH,
            run=run_retrieval_bm25,
            description="Lexical-only control, real corpus.",
            gates=[],
        )
    )
    register_suite(
        Suite(
            name="ledgerline.numeric_cat",
            project="ledgerline",
            dataset=NUMERIC_PATH,
            run=run_numeric,
            description="Figures resolved to cells of a real 10-K's tables.",
            gates=[],
        )
    )
    register_suite(
        Suite(
            name="ledgerline.numeric_cat_baseline",
            project="ledgerline",
            dataset=NUMERIC_PATH,
            run=run_numeric_baseline,
            description="Naive first-number extractor, real corpus. The permanent 'before'.",
            gates=[],
        )
    )
