"""Okapi BM25.

This is the *baseline*, and it exists so the harness has a real "before" to
report. Postgres does the lexical half of hybrid retrieval in production
(`ledgerline.hybrid_search`); this pure-Python version is what lets the eval
suite run in CI with no database, no GPU, and no network.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,]\d+)*")

# Deliberately small. An aggressive stoplist strips "risk", "loss", "cost" --
# words that carry real meaning in a filing.
_STOPWORDS = frozenset(
    (
        "a an and are as at be by for from has have in is it its of on or "
        "that the to was were will with"
    ).split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    doc_ids: list[str] = field(default_factory=list)
    _docs: list[Counter] = field(default_factory=list)
    _lengths: list[int] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _avgdl: float = 0.0

    @classmethod
    def build(cls, documents: Iterable[tuple[str, str]], **kwargs: float) -> BM25Index:
        index = cls(**kwargs)  # type: ignore[arg-type]
        for doc_id, text in documents:
            index.add(doc_id, text)
        index.finalize()
        return index

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        counts = Counter(tokens)
        self.doc_ids.append(doc_id)
        self._docs.append(counts)
        self._lengths.append(len(tokens))
        self._df.update(counts.keys())

    def finalize(self) -> None:
        self._avgdl = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        # Robertson/Sparck-Jones idf with the +1 that keeps it non-negative for
        # terms appearing in more than half the corpus.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if not self._docs:
            return []
        if self._avgdl == 0.0:
            self.finalize()

        terms = tokenize(query)
        scored: list[tuple[str, float]] = []
        for i, counts in enumerate(self._docs):
            length = self._lengths[i]
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * (length / self._avgdl if self._avgdl else 1)
                )
                score += self._idf(term) * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((self.doc_ids[i], score))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    def rank(self, query: str, k: int = 10) -> list[str]:
        return [doc_id for doc_id, _ in self.search(query, k)]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, limit: int = 10
) -> list[str]:
    """Fuse ranked lists on rank, not score.

    Same reasoning as the SQL version: BM25 scores and cosine distances are not
    on comparable scales, and normalising them is a calibration problem you do
    not need to have.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    return [doc for doc, _ in sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:limit]]
