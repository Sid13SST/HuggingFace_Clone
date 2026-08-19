"""Embedding providers, and the cache that keeps CI offline.

The constraint that shapes this file: the eval suite must run with no network,
no GPU, and no model download, because a harness that needs three of those is
a harness people stop running. The resolution is a two-layer design that is
also just good practice in production.

  * A real embedder (model2vec static embeddings -- genuinely semantic, pure
    numpy inference, no torch) generates vectors once, on a developer machine.
  * Those vectors are committed as an `.npz` keyed by content hash, and CI
    reads them. A cache miss is a loud error naming the command to fix it,
    never a silent fallback to a different embedding.

Content-hash keying is what makes this safe: edit a golden-set question or a
corpus chunk and the lookup misses immediately, rather than quietly scoring
the new text against the old vector.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from shared.logging import get_logger

log = get_logger(__name__)

#: Static distilled embeddings. 256-dim, ~30MB, no torch, deterministic.
DEFAULT_MODEL = "minishlab/potion-base-8M"
DEFAULT_DIM = 256


class EmbeddingCacheMiss(KeyError):
    """Text was not in the committed cache.

    Deliberately fatal. Falling back to a different embedder here would mean
    half the corpus scored under one model and half under another, producing a
    retrieval number that means nothing and looks fine.
    """


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def text_key(text: str) -> str:
    """Stable content hash. Whitespace-normalised so reflowing a fixture does
    not invalidate every vector in the cache."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise so a dot product is cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class StaticEmbedder:
    """model2vec-backed embedder. Requires the `ledgerline` extra and a download."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from model2vec import StaticModel
        except ImportError as exc:  # pragma: no cover - exercised by the extra
            raise ImportError(
                "model2vec is not installed. `pip install -e \".[ledgerline]\"`, "
                "or use the committed vector cache instead."
            ) from exc
        self._model = StaticModel.from_pretrained(model_name)
        self.model_name = model_name
        probe = self._model.encode(["dimension probe"])
        self.dim = int(np.asarray(probe).shape[1])

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts)), dtype=np.float32)


class CachedEmbedder:
    """Reads vectors from a committed `.npz`. What CI uses."""

    def __init__(
        self,
        vectors: dict[str, np.ndarray],
        dim: int,
        model_name: str = DEFAULT_MODEL,
        fallback: Embedder | None = None,
    ) -> None:
        self._vectors = vectors
        self.dim = dim
        self.model_name = model_name
        self._fallback = fallback

    @classmethod
    def from_npz(cls, path: str | Path, fallback: Embedder | None = None) -> CachedEmbedder:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"embedding cache missing: {resolved}. Run `ledgerline embed` to build it."
            )
        with np.load(resolved, allow_pickle=False) as payload:
            keys = [str(k) for k in payload["keys"]]
            matrix = payload["vectors"].astype(np.float32)
            model_name = str(payload["model"][0]) if "model" in payload else DEFAULT_MODEL
        vectors = {key: matrix[i] for i, key in enumerate(keys)}
        log.debug("embeddings.cache.loaded", n=len(vectors), dim=int(matrix.shape[1]))
        return cls(vectors, int(matrix.shape[1]), model_name, fallback)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        missing: list[str] = []
        for text in texts:
            vector = self._vectors.get(text_key(text))
            if vector is None:
                if self._fallback is None:
                    missing.append(text)
                    continue
                vector = self._fallback.encode([text])[0]
            rows.append(vector)

        if missing:
            preview = "; ".join(t[:60] for t in missing[:3])
            raise EmbeddingCacheMiss(
                f"{len(missing)} text(s) not in the embedding cache -- run "
                f"`ledgerline embed` after editing fixtures. First: {preview}"
            )
        return np.vstack(rows).astype(np.float32)

    def __contains__(self, text: str) -> bool:
        return text_key(text) in self._vectors


def save_cache(path: str | Path, texts: Sequence[str], embedder: Embedder) -> Path:
    """Encode `texts` and write the cache. Idempotent for unchanged text."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    unique: dict[str, str] = {}
    for text in texts:
        unique.setdefault(text_key(text), text)

    keys = sorted(unique)
    matrix = embedder.encode([unique[k] for k in keys])
    np.savez_compressed(
        resolved,
        keys=np.array(keys),
        vectors=matrix.astype(np.float32),
        model=np.array([getattr(embedder, "model_name", DEFAULT_MODEL)]),
    )
    log.info("embeddings.cache.saved", path=str(resolved), n=len(keys), dim=matrix.shape[1])
    return resolved
