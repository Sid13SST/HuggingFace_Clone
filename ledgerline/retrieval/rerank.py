"""Cross-encoder reranking.

The third retrieval stage, and the one that fixes what the first two could not.

BM25 matches tokens. Static embeddings match topics. Neither reads the query
and the passage *together*, so both answer "why did margin decline" with a
chunk about whether the decline will persist -- same subject, wrong
proposition. A cross-encoder scores the pair jointly and can tell those apart.

The cost is that it cannot be an index: every candidate is a forward pass, so
it only ever runs over a shortlist the cheap stages produced. Retrieve wide,
rerank narrow.

Same offline discipline as embeddings: scores are computed once with an ONNX
cross-encoder (no torch), cached by pair hash, and committed. CI reads the
cache and a miss is fatal rather than silently falling back to a different
model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ledgerline.retrieval.hybrid import HybridRetriever
from shared.logging import get_logger

log = get_logger(__name__)

#: ONNX export of ms-marco-MiniLM-L-6-v2. Small, fast, no torch dependency.
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class RerankCacheMiss(KeyError):
    """A (query, document) pair was not in the committed score cache."""


@runtime_checkable
class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


def pair_key(query: str, document: str) -> str:
    """Content hash of a (query, document) pair.

    Both sides normalised for whitespace, joined by a NUL that cannot occur in
    either, so ("ab", "c") and ("a", "bc") cannot collide.
    """
    normalized = f"{' '.join(query.split())}\x00{' '.join(document.split())}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


class CrossEncoderReranker:
    """fastembed-backed ONNX cross-encoder. Needs the `ledgerline` extra."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised by the extra
            raise ImportError(
                "fastembed is not installed. `pip install -e \".[ledgerline]\"`, "
                "or use the committed score cache instead."
            ) from exc
        self._model = TextCrossEncoder(model_name=model_name)
        self.model_name = model_name

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        return [float(s) for s in self._model.rerank(query, list(documents))]


@dataclass
class CachedReranker:
    """Reads pair scores from a committed `.npz`. What CI uses."""

    scores: dict[str, float]
    model_name: str = DEFAULT_RERANK_MODEL
    fallback: Reranker | None = None

    @classmethod
    def from_npz(cls, path: str | Path, fallback: Reranker | None = None) -> CachedReranker:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"rerank cache missing: {resolved}. Run `ledgerline rerank-cache`."
            )
        with np.load(resolved, allow_pickle=False) as payload:
            keys = [str(k) for k in payload["keys"]]
            values = payload["scores"].astype(np.float32)
            model = str(payload["model"][0]) if "model" in payload else DEFAULT_RERANK_MODEL
        log.debug("rerank.cache.loaded", n=len(keys))
        return cls(scores=dict(zip(keys, map(float, values), strict=True)), model_name=model,
                   fallback=fallback)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        out: list[float] = []
        missing: list[str] = []
        for document in documents:
            value = self.scores.get(pair_key(query, document))
            if value is None:
                if self.fallback is None:
                    missing.append(document[:60])
                    continue
                value = self.fallback.score(query, [document])[0]
            out.append(value)

        if missing:
            raise RerankCacheMiss(
                f"{len(missing)} (query, document) pair(s) not cached -- run "
                f"`ledgerline rerank-cache`. Query: {query[:60]!r}"
            )
        return out


def save_rerank_cache(
    path: str | Path,
    pairs: Sequence[tuple[str, str]],
    reranker: Reranker,
) -> Path:
    """Score every pair and write the cache.

    Pairs are grouped by query so the cross-encoder batches, which is most of
    the runtime.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    by_query: dict[str, list[str]] = {}
    for query, document in pairs:
        by_query.setdefault(query, [])
        if document not in by_query[query]:
            by_query[query].append(document)

    keys: list[str] = []
    values: list[float] = []
    for query, documents in by_query.items():
        for document, value in zip(documents, reranker.score(query, documents), strict=True):
            keys.append(pair_key(query, document))
            values.append(value)

    order = np.argsort(keys)
    np.savez_compressed(
        resolved,
        keys=np.array(keys)[order],
        scores=np.array(values, dtype=np.float32)[order],
        model=np.array([getattr(reranker, "model_name", DEFAULT_RERANK_MODEL)]),
    )
    log.info("rerank.cache.saved", path=str(resolved), pairs=len(keys))
    return resolved


@dataclass
class RerankingRetriever:
    """Retrieve wide with the cheap stages, then rescore the shortlist."""

    base: HybridRetriever
    reranker: Reranker
    documents: dict[str, str] = field(default_factory=dict)
    #: Shortlist size handed to the cross-encoder. Wider costs a forward pass
    #: per extra candidate but is the only way a document the cheap stages
    #: ranked poorly can ever be rescued -- reranking cannot invent recall.
    candidate_k: int = 25

    @classmethod
    def build(
        cls,
        documents: Sequence[tuple[str, str]],
        base: HybridRetriever,
        reranker: Reranker,
        **kwargs,
    ) -> RerankingRetriever:
        return cls(base=base, reranker=reranker, documents=dict(documents), **kwargs)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        candidates = self.base.rank(query, k=self.candidate_k)
        if not candidates:
            return []
        scores = self.reranker.score(query, [self.documents[c] for c in candidates])
        ordered = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: (-pair[1], pair[0])
        )
        return ordered[:k]

    def rank(self, query: str, k: int = 10) -> list[str]:
        return [doc_id for doc_id, _ in self.search(query, k)]

    def explain(self, query: str, k: int = 10) -> list[dict]:
        """Rank before and after reranking, per document.

        `moved` is the diagnostic that matters: a reranker that moves nothing
        is costing a forward pass per candidate for no benefit, and that shows
        up here before it shows up in a latency budget.
        """
        candidates = self.base.rank(query, k=self.candidate_k)
        before = {doc_id: i + 1 for i, doc_id in enumerate(candidates)}
        return [
            {
                "doc_id": doc_id,
                "rerank_score": score,
                "rank_before": before.get(doc_id),
                "rank_after": position,
                "moved": (before.get(doc_id) or 0) - position,
            }
            for position, (doc_id, score) in enumerate(self.search(query, k), start=1)
        ]
