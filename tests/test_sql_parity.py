"""Python retrieval versus Postgres retrieval.

Split deliberately in two. The arithmetic of comparing two rankings needs no
database and runs everywhere, so it is tested with fakes. The claim that the
two *implementations* agree needs a real Postgres with pgvector, so those tests
skip when one is not reachable -- and run in the CI job that has one.

A skipped test is not a passing test. `ledgerline parity` is wired into the
schema job for exactly that reason.
"""

from __future__ import annotations

import pytest

from ledgerline.evals.parity import ArmParity, _displacement, compare_arm


class TestDisplacement:
    def test_identical_rankings_have_none(self):
        assert _displacement(["a", "b", "c"], ["a", "b", "c"]) == 0.0

    def test_counts_only_shared_documents(self):
        """A document one side never returned has no rank to compare against.

        Scoring it as "maximally displaced" would let a recall difference show
        up as an ordering difference, and the two have different fixes.
        """
        # 'z' is unique to the left; only a and b are shared, and both moved 1.
        assert _displacement(["a", "b", "z"], ["b", "a"]) == 1.0

    def test_disjoint_rankings_are_zero_not_infinite(self):
        assert _displacement(["a"], ["b"]) == 0.0

    def test_reversal_is_symmetric(self):
        forward = _displacement(["a", "b", "c"], ["c", "b", "a"])
        backward = _displacement(["c", "b", "a"], ["a", "b", "c"])
        assert forward == backward > 0


class TestCompareArm:
    def test_perfect_agreement(self):
        rank = ["a", "b", "c"]
        result = compare_arm("dense", ["q1", "q2"], lambda q: rank, lambda q: rank)
        assert result.exact_order == 1.0
        assert result.top1_agreement == 1.0
        assert result.overlap_at_k == 1.0
        assert result.mean_displacement == 0.0

    def test_same_set_different_order_is_not_exact(self):
        """The distinction the whole module exists to make.

        Retrieving the same documents in a different order is a real
        difference -- it changes nDCG and it changes what a reranker sees --
        so overlap staying at 1.0 must not be allowed to hide it.
        """
        result = compare_arm(
            "lexical", ["q"], lambda q: ["a", "b"], lambda q: ["b", "a"]
        )
        assert result.overlap_at_k == 1.0
        assert result.exact_order == 0.0
        assert result.top1_agreement == 0.0

    def test_overlap_is_jaccard_not_prefix(self):
        result = compare_arm("x", ["q"], lambda q: ["a", "b"], lambda q: ["a", "c"])
        assert result.overlap_at_k == pytest.approx(1 / 3)

    def test_one_side_empty_is_recorded_separately(self):
        result = compare_arm("lexical", ["q"], lambda q: ["a"], lambda q: [])
        assert result.asymmetric_empty == 1
        # top-1 cannot be scored against nothing, so it must not count as a hit.
        assert result.top1_agreement == 0.0

    def test_both_empty_agrees(self):
        result = compare_arm("lexical", ["q"], lambda q: [], lambda q: [])
        assert result.asymmetric_empty == 0
        assert result.exact_order == 1.0
        assert result.overlap_at_k == 1.0

    def test_no_queries_does_not_divide_by_zero(self):
        assert compare_arm("dense", [], lambda q: [], lambda q: []).queries == 0

    def test_as_dict_is_prefixed_by_arm(self):
        keys = ArmParity("dense", 1, 1.0, 1.0, 1.0, 0.0, 0).as_dict()
        assert set(keys) == {
            "dense.exact_order",
            "dense.top1_agreement",
            "dense.overlap@k",
            "dense.mean_displacement",
        }


# --------------------------------------------------------------------------
# the half that needs a database
# --------------------------------------------------------------------------


class SchemaWidthEmbedder:
    """Deterministic vectors at the width the column actually declares.

    tests/test_embeddings.py's FakeEmbedder is 3-dimensional, which is right
    for reasoning about cosine by hand and wrong here: `chunks.embedding` is
    `vector(256)` and Postgres rejects anything else. Borrowing that fake was
    the first thing I tried, and the database said no -- correctly.
    """

    model_name = "fake-schema-width"

    def __init__(self) -> None:
        from shared.embeddings import DEFAULT_DIM

        self.dim = DEFAULT_DIM

    def encode(self, texts):
        import hashlib

        import numpy as np

        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rows.append(
                np.random.default_rng(seed).standard_normal(self.dim).astype("float32")
            )
        return np.vstack(rows)


@pytest.fixture(scope="module")
def indexed(db):
    from ledgerline.evals.parity import ingest_fixture_corpus

    ingest_fixture_corpus(db)
    return db


@pytest.fixture(scope="module")
def sql_retriever(indexed):
    from ledgerline.evals import embedder
    from ledgerline.evals.parity import FIXTURE_CIK
    from ledgerline.retrieval.sql import SqlRetriever

    # Scoped to the fixture issuer. A real database holds more than one corpus
    # -- ingesting an actual 10-K alongside these fixtures is what proved it --
    # and an unscoped retriever silently compares two different indexes.
    return SqlRetriever(conn=indexed, embedder=embedder(), cik=FIXTURE_CIK)


@pytest.fixture(scope="module")
def questions():
    import json

    from ledgerline.evals import HERE

    out = []
    with (HERE / "datasets" / "retrieval.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                out.append(json.loads(stripped)["inputs"]["question"])
    return out


class TestIngest:
    def test_every_corpus_chunk_is_searchable(self, sql_retriever):
        from ledgerline.evals import load_corpus

        assert {r["id"] for r in load_corpus()} == sql_retriever.indexed_ids()

    def test_reingest_replaces_rather_than_appends(self, indexed):
        """Idempotence, and the specific bug an upsert would not catch.

        Chunks are written by replacement, so re-running ingest must leave the
        row count unchanged -- not doubled, and not holding stale rows from a
        previous chunking of the same document.
        """
        from ledgerline.evals.parity import ingest_fixture_corpus

        before = indexed.execute("SELECT count(*) FROM ledgerline.chunks").fetchone()[0]
        ingest_fixture_corpus(indexed)
        after = indexed.execute("SELECT count(*) FROM ledgerline.chunks").fetchone()[0]
        assert before == after

    def test_shrinking_a_document_removes_its_orphans(self, indexed):
        """The failure mode that motivated replace-over-upsert.

        Re-chunk a document into fewer pieces and the tail must disappear. An
        `ON CONFLICT (document_id, ordinal) DO UPDATE` upsert leaves those rows
        behind, still indexed and still retrievable.
        """
        from ledgerline.evals.parity import FIXTURE_CIK, ingest_fixture_corpus
        from ledgerline.ingest.pipeline import (
            ChunkRow,
            Document,
            Issuer,
            ingest_document,
        )

        issuer = Issuer(cik=FIXTURE_CIK, name="Northwind Manufacturing Inc.")
        document = Document(cik=FIXTURE_CIK, kind="deck", accession="shrink-test")
        wide = [ChunkRow(f"shrink-{i}", f"chunk number {i}", i) for i in range(5)]
        ingest_document(indexed, issuer, document, wide, SchemaWidthEmbedder())
        indexed.commit()

        narrow = wide[:2]
        ingest_document(indexed, issuer, document, narrow, SchemaWidthEmbedder())
        indexed.commit()

        remaining = indexed.execute(
            "SELECT count(*) FROM ledgerline.chunks WHERE external_id LIKE 'shrink-%%'"
        ).fetchone()[0]
        assert remaining == 2

        indexed.execute(
            "DELETE FROM ledgerline.documents WHERE accession = 'shrink-test'"
        )
        indexed.commit()
        ingest_fixture_corpus(indexed)

    def test_embeddings_round_trip_at_the_declared_dimension(self, indexed):
        from shared.embeddings import DEFAULT_DIM

        dim = indexed.execute(
            "SELECT vector_dims(embedding) FROM ledgerline.chunks "
            "WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()[0]
        assert dim == DEFAULT_DIM


class TestParity:
    def test_dense_arms_are_identical(self, sql_retriever, questions):
        """The one hard assertion.

        Both sides read the same committed vectors and order by cosine, so
        there is no legitimate reason for them to differ. If this fails the
        write path is wrong -- a truncated vector, a normalisation applied on
        one side, a chunk stored against the wrong text.
        """
        from ledgerline.evals import embedder, hybrid_retriever

        offline = hybrid_retriever()
        for question in questions:
            assert offline.dense.rank(question, embedder(), k=10) == (
                sql_retriever.dense_rank(question, k=10)
            ), f"dense divergence on {question!r}"

    def test_lexical_arms_differ_and_that_is_recorded(self, sql_retriever, questions):
        """BM25 and ts_rank_cd are different functions over different tokens.

        This test does not demand agreement; it demands that the two are
        actually retrieving the same *corpus*, which is the only thing that
        would make a comparison meaningful in the first place.
        """
        from ledgerline.evals import hybrid_retriever
        from ledgerline.evals.parity import compare_arm

        offline = hybrid_retriever()
        result = compare_arm(
            "lexical",
            questions,
            lambda q: offline.bm25.rank(q, k=10),
            lambda q: sql_retriever.lexical_rank(q, k=10),
        )
        # Some agreement is required -- zero overlap would mean one side is
        # searching an empty or different index, not merely ranking differently.
        assert result.overlap_at_k > 0.3, f"lexical overlap {result.overlap_at_k:.3f}"

    def test_fused_rankings_mostly_agree(self, sql_retriever, questions):
        """Fusion damps the lexical divergence but cannot erase it.

        The bound is loose on purpose. Tightening it until it barely passes
        would turn a real property into a tripwire that fires on unrelated
        corpus edits.
        """
        from ledgerline.evals import hybrid_retriever
        from ledgerline.evals.parity import compare_arm

        offline = hybrid_retriever()
        result = compare_arm(
            "fused",
            questions,
            lambda q: offline.rank(q, k=10),
            lambda q: sql_retriever.rank(q, k=10),
        )
        assert result.overlap_at_k > 0.5, f"fused overlap {result.overlap_at_k:.3f}"

    def test_candidate_depth_matches_the_offline_retriever(self, sql_retriever):
        """Comparing two searches run to different depths measures nothing."""
        from ledgerline.evals import hybrid_retriever

        assert sql_retriever.candidate_k == hybrid_retriever().candidate_k
