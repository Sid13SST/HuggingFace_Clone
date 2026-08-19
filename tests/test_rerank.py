import numpy as np
import pytest

from ledgerline.retrieval.hybrid import HybridRetriever
from ledgerline.retrieval.rerank import (
    CachedReranker,
    RerankCacheMiss,
    RerankingRetriever,
    pair_key,
    save_rerank_cache,
)
from tests.test_embeddings import FakeEmbedder


class KeywordReranker:
    """Deterministic stand-in for a cross-encoder.

    Rewards the query appearing as a contiguous phrase and penalises token
    repetition. Both halves matter: a real cross-encoder reads query and
    passage jointly, so it rewards a passage that states the proposition and
    is unimpressed by one that merely repeats the query's words. That is the
    behaviour under test, and it needs no ONNX session to model.
    """

    model_name = "fake-cross-encoder"

    def score(self, query, documents):
        needle = query.lower().strip("?")
        scores = []
        for document in documents:
            tokens = document.lower().split()
            repetition = 1 - (len(set(tokens)) / len(tokens)) if tokens else 0.0
            scores.append((1.0 if needle in document.lower() else 0.0) - repetition)
        return scores


DOCS = [
    ("d-cause", "gross margin declined because steel and freight costs rose"),
    ("d-outlook", "gross margin pressure is transitory and should recover"),
    ("d-revenue", "net revenue grew twelve percent on higher volumes"),
    # Term-frequency bait: every query token, repeated, in an order that says
    # nothing. BM25 ranks it first; joint scoring does not. This is the shape
    # of the failure reranking exists to fix.
    ("d-bait", "gross gross margin margin declined declined margin gross"),
]


class TestPairKey:
    def test_is_whitespace_insensitive_on_both_sides(self):
        assert pair_key("why  did\nit", "the  doc") == pair_key("why did it", "the doc")

    def test_cannot_collide_across_the_boundary(self):
        """("ab","c") and ("a","bc") must not hash the same."""
        assert pair_key("ab", "c") != pair_key("a", "bc")

    def test_differs_on_a_real_edit(self):
        assert pair_key("q", "doc one") != pair_key("q", "doc two")


class TestCachedReranker:
    @pytest.fixture
    def cache_path(self, tmp_path):
        pairs = [("gross margin declined", text) for _, text in DOCS]
        return save_rerank_cache(tmp_path / "r.npz", pairs, KeywordReranker())

    def test_round_trips_scores(self, cache_path):
        reranker = CachedReranker.from_npz(cache_path)
        scores = reranker.score("gross margin declined", [t for _, t in DOCS])
        assert scores[0] == 1.0
        assert scores[2] == 0.0

    def test_miss_is_fatal_not_silent(self, cache_path):
        reranker = CachedReranker.from_npz(cache_path)
        with pytest.raises(RerankCacheMiss, match="rerank-cache"):
            reranker.score("a question nobody scored", ["some document"])

    def test_miss_falls_back_when_one_is_supplied(self, cache_path):
        reranker = CachedReranker.from_npz(cache_path, fallback=KeywordReranker())
        assert reranker.score("net revenue grew", ["net revenue grew twelve"]) == [1.0]

    def test_missing_file_names_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="rerank-cache"):
            CachedReranker.from_npz(tmp_path / "absent.npz")

    def test_cache_deduplicates_repeated_pairs(self, tmp_path):
        pairs = [("q", "doc"), ("q", "doc"), ("q", "other")]
        path = save_rerank_cache(tmp_path / "d.npz", pairs, KeywordReranker())
        with np.load(path) as payload:
            assert len(payload["keys"]) == 2

    def test_records_the_model_it_was_built_with(self, cache_path):
        assert CachedReranker.from_npz(cache_path).model_name == "fake-cross-encoder"


class TestRerankingRetriever:
    @pytest.fixture
    def retriever(self):
        base = HybridRetriever.build(DOCS, FakeEmbedder())
        return RerankingRetriever.build(DOCS, base, KeywordReranker(), candidate_k=4)

    def test_demotes_the_keyword_bait(self, retriever):
        query = "gross margin declined"
        base = retriever.base.rank(query, k=4)
        reranked = retriever.rank(query, k=4)
        assert base.index("d-bait") < base.index("d-revenue"), (
            "fixture no longer exercises the failure: bait must start ahead"
        )
        assert reranked[0] == "d-cause"
        assert reranked.index("d-bait") > base.index("d-bait")

    def test_explain_reports_movement(self, retriever):
        """A reranker that moves nothing is paying a forward pass per candidate
        for no benefit, and this is where that shows up first."""
        rows = retriever.explain("gross margin declined", k=4)
        assert all("rank_before" in row and "rank_after" in row for row in rows)
        assert any(row["moved"] != 0 for row in rows)

    def test_cannot_invent_recall(self):
        """Reranking only reorders what retrieval found.

        With candidate_k=1 the shortlist holds one document, so a perfect
        scorer still cannot surface the right one. This is why the retrieval
        stages matter and why candidate_k is a real knob.
        """
        base = HybridRetriever.build(DOCS, FakeEmbedder())
        narrow = RerankingRetriever.build(DOCS, base, KeywordReranker(), candidate_k=1)
        assert len(narrow.rank("gross margin declined", k=3)) == 1

    def test_empty_corpus_is_safe(self):
        base = HybridRetriever.build([], FakeEmbedder())
        empty = RerankingRetriever.build([], base, KeywordReranker())
        assert empty.rank("anything") == []

    def test_ties_break_deterministically(self, retriever):
        # Every score is 0.0 for a query matching nothing; order must be stable.
        assert retriever.rank("no document contains this", k=3) == retriever.rank(
            "no document contains this", k=3
        )


class TestCommittedRerankCache:
    def test_covers_every_pair_the_suite_scores(self):
        """A stale cache must fail here, not halfway through a CI eval run."""
        from ledgerline.evals import rerank_pairs, reranker

        cached = reranker()
        missing = [p for p in rerank_pairs() if pair_key(*p) not in cached.scores]
        assert not missing, f"{len(missing)} pairs need `ledgerline rerank-cache`"

    def test_fixes_the_queries_dense_retrieval_regressed(self):
        """The whole reason this stage exists.

        Both queries ask about a proposition, not a topic: *why* margin
        declined, and how much pricing is *holding*. Static embeddings put a
        same-topic wrong-aspect chunk first on each; the cross-encoder reads
        query and passage together and does not.
        """
        from ledgerline.evals import hybrid_retriever, reranking_retriever
        from shared.evals.metrics import ndcg_at_k

        hybrid, reranked = hybrid_retriever(), reranking_retriever()
        for question, gold in [
            ("Why did gross margin decline?", ["c-mda-margin", "c-risk-supply"]),
            (
                "How much of the announced pricing is holding with customers?",
                ["c-call-qa-cfo-answer"],
            ),
        ]:
            before = ndcg_at_k(gold, hybrid.rank(question, k=10), 10)
            after = ndcg_at_k(gold, reranked.rank(question, k=10), 10)
            assert after > before, f"{question!r}: {before:.3f} -> {after:.3f}"
