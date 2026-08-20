"""Hybrid retrieval: dense + lexical, fused on rank.

Mirrors `ledgerline.hybrid_search` in schema.sql. Production runs the SQL;
this exists so the fusion can be tuned and gated offline.

Why fuse on rank rather than score: BM25 scores are unbounded and corpus
dependent, cosine similarities sit in [-1, 1], and the two distributions move
independently as the corpus grows. Normalising them into a shared scale is a
calibration problem that has to be re-solved every time the data changes.
Reciprocal rank fusion sidesteps it -- only the ordering matters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ledgerline.retrieval.bm25 import BM25Index
from shared.embeddings import Embedder, normalize

#: From the original RRF paper, and the value the baselines were measured at.
#: Change it and re-run the suite -- it is a tuning knob, not a constant.
RRF_K = 60


@dataclass
class DenseIndex:
    doc_ids: list[str]
    matrix: np.ndarray  # row-normalised, shape (n_docs, dim)

    @classmethod
    def build(
        cls, documents: Sequence[tuple[str, str]], embedder: Embedder
    ) -> DenseIndex:
        doc_ids = [doc_id for doc_id, _ in documents]
        texts = [text for _, text in documents]
        if not texts:
            return cls(doc_ids=[], matrix=np.zeros((0, getattr(embedder, "dim", 1))))
        return cls(doc_ids=doc_ids, matrix=normalize(embedder.encode(texts)))

    def search(
        self, query: str, embedder: Embedder, k: int = 10
    ) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []
        query_vector = normalize(embedder.encode([query]))[0]
        scores = self.matrix @ query_vector
        # argsort over the full set: at fixture scale a partial sort buys
        # nothing, and Postgres does this with an HNSW index in production.
        order = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]

    def rank(self, query: str, embedder: Embedder, k: int = 10) -> list[str]:
        return [doc_id for doc_id, _ in self.search(query, embedder, k)]


@dataclass
class HybridRetriever:
    bm25: BM25Index
    dense: DenseIndex
    embedder: Embedder
    rrf_k: int = RRF_K
    #: Candidates pulled from each arm before fusion. Wider than the final k
    #: on purpose: a document ranked 30th by one arm and 2nd by the other is
    #: exactly the case fusion exists to rescue.
    candidate_k: int = 40

    @classmethod
    def build(
        cls, documents: Sequence[tuple[str, str]], embedder: Embedder, **kwargs
    ) -> HybridRetriever:
        return cls(
            bm25=BM25Index.build(documents),
            dense=DenseIndex.build(documents, embedder),
            embedder=embedder,
            **kwargs,
        )

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        lexical = self.bm25.rank(query, k=self.candidate_k)
        semantic = self.dense.rank(query, self.embedder, k=self.candidate_k)

        scores: dict[str, float] = {}
        for ranking in (lexical, semantic):
            for position, doc_id in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + position)

        ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ordered[:k]

    def rank(self, query: str, k: int = 10) -> list[str]:
        return [doc_id for doc_id, _ in self.search(query, k)]

    def explain(self, query: str, k: int = 10) -> list[dict]:
        """Per-arm ranks behind a fused result.

        The debugging view that answers "why is this document here" without
        guessing -- and the one that shows whether fusion is actually earning
        its keep or the lexical arm is carrying everything.
        """
        lexical = self.bm25.rank(query, k=self.candidate_k)
        semantic = self.dense.rank(query, self.embedder, k=self.candidate_k)
        lex_rank = {d: i + 1 for i, d in enumerate(lexical)}
        den_rank = {d: i + 1 for i, d in enumerate(semantic)}

        return [
            {
                "doc_id": doc_id,
                "fused_score": score,
                "bm25_rank": lex_rank.get(doc_id),
                "dense_rank": den_rank.get(doc_id),
                "found_only_by": (
                    "dense" if doc_id not in lex_rank
                    else "bm25" if doc_id not in den_rank
                    else None
                ),
            }
            for doc_id, score in self.search(query, k)
        ]
