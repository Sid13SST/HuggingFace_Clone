import numpy as np
import pytest

from ledgerline.retrieval.hybrid import DenseIndex, HybridRetriever
from shared.embeddings import (
    CachedEmbedder,
    EmbeddingCacheMiss,
    normalize,
    save_cache,
    text_key,
)


class FakeEmbedder:
    """Deterministic stand-in. Keeps these tests about the plumbing.

    Dimension 3, one axis per topic word, so similarity is inspectable by
    hand rather than being a number nobody can reason about.
    """

    dim = 3
    model_name = "fake"

    VOCAB = {"margin": 0, "revenue": 1, "employees": 2, "employed": 2, "people": 2}

    def encode(self, texts):
        rows = []
        for text in texts:
            vector = np.zeros(3, dtype=np.float32)
            for token in text.lower().split():
                axis = self.VOCAB.get(token.strip(".,?"))
                if axis is not None:
                    vector[axis] += 1.0
            if not vector.any():
                vector[:] = 1e-6
            rows.append(vector)
        return np.vstack(rows)


class TestTextKey:
    def test_is_whitespace_insensitive(self):
        """Reflowing a fixture must not invalidate every vector in the cache."""
        assert text_key("net  revenue\n  grew") == text_key("net revenue grew")

    def test_differs_on_real_edits(self):
        assert text_key("net revenue grew") != text_key("net revenue fell")


class TestNormalize:
    def test_rows_become_unit_length(self):
        result = normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
        assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

    def test_zero_row_does_not_divide_by_zero(self):
        result = normalize(np.zeros((1, 4)))
        assert np.isfinite(result).all()


class TestCachedEmbedder:
    @pytest.fixture
    def cache_path(self, tmp_path):
        return save_cache(tmp_path / "e.npz", ["alpha text", "beta text"], FakeEmbedder())

    def test_round_trips(self, cache_path):
        embedder = CachedEmbedder.from_npz(cache_path)
        assert embedder.dim == 3
        assert embedder.encode(["alpha text"]).shape == (1, 3)

    def test_deduplicates_identical_texts(self, tmp_path):
        path = save_cache(tmp_path / "d.npz", ["same", "same", "other"], FakeEmbedder())
        with np.load(path) as payload:
            assert len(payload["keys"]) == 2

    def test_miss_is_fatal_not_silent(self, cache_path):
        """The property that makes offline eval trustworthy.

        A silent fallback would score half a corpus under one model and half
        under another, producing a retrieval number that means nothing.
        """
        embedder = CachedEmbedder.from_npz(cache_path)
        with pytest.raises(EmbeddingCacheMiss, match="ledgerline embed"):
            embedder.encode(["never encoded"])

    def test_miss_falls_back_when_one_is_supplied(self, cache_path):
        embedder = CachedEmbedder.from_npz(cache_path, fallback=FakeEmbedder())
        assert embedder.encode(["never encoded"]).shape == (1, 3)

    def test_edited_text_misses_rather_than_scoring_a_stale_vector(self, cache_path):
        embedder = CachedEmbedder.from_npz(cache_path)
        with pytest.raises(EmbeddingCacheMiss):
            embedder.encode(["alpha text edited"])

    def test_missing_file_names_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ledgerline embed"):
            CachedEmbedder.from_npz(tmp_path / "absent.npz")


class TestDenseIndex:
    DOCS = [
        ("d-margin", "margin margin"),
        ("d-revenue", "revenue revenue"),
        ("d-people", "we employed people"),
    ]

    def test_retrieves_on_meaning_not_tokens(self):
        """The whole reason dense retrieval was added.

        The document says "employed"; the query says "employees". No lexical
        matcher bridges that -- the shared axis in the embedding does.
        """
        index = DenseIndex.build(self.DOCS, FakeEmbedder())
        assert index.rank("employees", FakeEmbedder(), k=1) == ["d-people"]

    def test_empty_index_is_safe(self):
        index = DenseIndex.build([], FakeEmbedder())
        assert index.rank("anything", FakeEmbedder()) == []

    def test_scores_are_descending(self):
        index = DenseIndex.build(self.DOCS, FakeEmbedder())
        scores = [s for _, s in index.search("margin revenue", FakeEmbedder(), k=3)]
        assert scores == sorted(scores, reverse=True)


class TestHybridRetriever:
    DOCS = [
        ("d-margin", "gross margin declined on steel costs"),
        ("d-revenue", "net revenue grew twelve percent"),
        ("d-people", "we employed six thousand people"),
    ]

    @pytest.fixture
    def retriever(self):
        return HybridRetriever.build(self.DOCS, FakeEmbedder())

    def test_lexical_hit_survives_fusion(self, retriever):
        assert retriever.rank("gross margin declined", k=1) == ["d-margin"]

    def test_dense_only_hit_is_rescued(self, retriever):
        """"employees" appears in no document. BM25 returns nothing; fusion
        still surfaces the right chunk through the dense arm."""
        assert "d-people" in retriever.rank("employees", k=3)

    def test_explain_attributes_each_result_to_an_arm(self, retriever):
        rows = retriever.explain("employees", k=3)
        people = next(r for r in rows if r["doc_id"] == "d-people")
        assert people["dense_rank"] is not None
        assert people["found_only_by"] == "dense"

    def test_fusion_is_deterministic(self, retriever):
        assert retriever.rank("net revenue", k=3) == retriever.rank("net revenue", k=3)

    def test_rrf_k_is_tunable(self):
        low = HybridRetriever.build(self.DOCS, FakeEmbedder(), rrf_k=1)
        high = HybridRetriever.build(self.DOCS, FakeEmbedder(), rrf_k=1000)
        # Not asserting an ordering change -- only that the knob reaches the
        # scores, which is the bug class that made DedupeConfig necessary.
        assert low.search("margin", k=3)[0][1] != high.search("margin", k=3)[0][1]


class TestCommittedCache:
    def test_covers_every_text_the_suites_encode(self):
        """A stale cache must fail here, not halfway through a CI eval run."""
        from ledgerline.evals import embedder, texts_to_embed

        cached = embedder()
        missing = [t for t in texts_to_embed() if t not in cached]
        assert not missing, f"{len(missing)} texts need `ledgerline embed`: {missing[:2]}"

    def test_dimension_matches_the_schema(self):
        """chunks.embedding is vector(256); a mismatch breaks ingest silently."""
        from ledgerline.evals import embedder

        assert embedder().dim == 256
