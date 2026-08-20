"""Does the database rank the way the offline mirror does?

Every gated number in this repo is produced by `HybridRetriever` reading a
JSONL fixture. Production reads Postgres. If those two disagree, the gates are
protecting a system nobody ships, and the disagreement will be discovered by a
user rather than by CI.

This module measures the gap per arm, because the arms have different
expectations:

  * dense   -- same vectors, same cosine ordering. Expected identical.
                Any divergence is a defect.
  * lexical -- our BM25 versus Postgres `ts_rank_cd` over a snowball-stemmed
                tsvector. Expected to differ. Measured, recorded, watched.
  * fused   -- RRF over both, so it inherits the lexical divergence, damped by
                the fact that fusion cares about rank rather than score.

The output is deliberately a set of numbers rather than a pass/fail, except for
the dense arm. A single "parity: ok" boolean would hide the interesting part.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

Ranker = Callable[[str], list[str]]


@dataclass(frozen=True)
class ArmParity:
    """Agreement between two rankings of the same corpus, one arm at a time."""

    arm: str
    queries: int
    exact_order: float
    top1_agreement: float
    overlap_at_k: float
    mean_displacement: float
    #: Queries where one side returned nothing and the other did not. Usually
    #: means the tsquery matched no document while BM25 still scored something,
    #: which is a recall difference rather than an ordering one.
    asymmetric_empty: int

    def as_dict(self) -> dict[str, float]:
        return {
            f"{self.arm}.exact_order": self.exact_order,
            f"{self.arm}.top1_agreement": self.top1_agreement,
            f"{self.arm}.overlap@k": self.overlap_at_k,
            f"{self.arm}.mean_displacement": self.mean_displacement,
        }


def _displacement(left: Sequence[str], right: Sequence[str]) -> float:
    """Mean |rank difference| over documents both rankings returned.

    Restricted to the intersection on purpose: a document only one side found
    has no rank on the other, and inventing one (k+1, say) would let a recall
    difference masquerade as an ordering difference. Recall differences are
    counted by `overlap_at_k` instead.
    """
    right_rank = {doc_id: i for i, doc_id in enumerate(right)}
    shared = [(i, right_rank[d]) for i, d in enumerate(left) if d in right_rank]
    if not shared:
        return 0.0
    return sum(abs(a - b) for a, b in shared) / len(shared)


def compare_arm(
    arm: str, questions: Sequence[str], offline: Ranker, sql: Ranker
) -> ArmParity:
    exact = top1 = overlap = displacement = 0.0
    asymmetric = 0

    for question in questions:
        left, right = offline(question), sql(question)
        exact += 1.0 if left == right else 0.0
        if left and right:
            top1 += 1.0 if left[0] == right[0] else 0.0
        elif left or right:
            asymmetric += 1
        union = set(left) | set(right)
        overlap += len(set(left) & set(right)) / len(union) if union else 1.0
        displacement += _displacement(left, right)

    n = max(len(questions), 1)
    return ArmParity(
        arm=arm,
        queries=len(questions),
        exact_order=exact / n,
        top1_agreement=top1 / n,
        overlap_at_k=overlap / n,
        mean_displacement=displacement / n,
        asymmetric_empty=asymmetric,
    )


def compare_all(
    questions: Sequence[str],
    offline,
    sql,
    k: int = 10,
) -> list[ArmParity]:
    """Per-arm parity for the offline retriever against the SQL one.

    `offline` is a HybridRetriever and `sql` a SqlRetriever; both are taken
    structurally rather than by type so a fake can stand in for either.
    """
    return [
        compare_arm(
            "dense",
            questions,
            lambda q: offline.dense.rank(q, offline.embedder, k=k),
            lambda q: sql.dense_rank(q, k=k),
        ),
        compare_arm(
            "lexical",
            questions,
            lambda q: offline.bm25.rank(q, k=k),
            lambda q: sql.lexical_rank(q, k=k),
        ),
        compare_arm(
            "fused",
            questions,
            lambda q: offline.rank(q, k=k),
            lambda q: sql.rank(q, k=k),
        ),
    ]


def scored_comparison(examples, offline, sql, k: int = 10) -> dict[str, float]:
    """nDCG and recall for both paths on the same golden set.

    Parity ratios answer "do these rank the same". This answers the question
    that actually decides whether the divergence matters: *is one of them
    worse*. Two systems can disagree on ordering constantly and score
    identically, and that is a very different situation from one of them
    quietly losing a relevant chunk.

    `examples` are the harness's Example objects, so this reuses the labels the
    gated suites are scored against rather than a second set that could drift.
    """
    from shared.evals.metrics import mean, ndcg_at_k, recall_at_k

    rankers = {
        "offline": lambda q: offline.rank(q, k=k),
        "sql": lambda q: sql.rank(q, k=k),
    }
    metrics: dict[str, float] = {}
    for label, rank in rankers.items():
        ranked = [(e, rank(e.inputs["question"])) for e in examples]
        metrics[f"{label}.ndcg@{k}"] = mean(
            ndcg_at_k(e.expected["relevant"], r, k) for e, r in ranked
        )
        metrics[f"{label}.recall@{k}"] = mean(
            recall_at_k(e.expected["relevant"], r, k) for e, r in ranked
        )
    return metrics

# --------------------------------------------------------------------------
# loading the fixture corpus into a real database
# --------------------------------------------------------------------------

#: The synthetic issuer the fixture corpus describes. Not a real company; see
#: the header of fixtures/corpus.jsonl.
FIXTURE_CIK = "0000000000"
FIXTURE_ISSUER_NAME = "Northwind Manufacturing Inc."


def ingest_fixture_corpus(conn, embedder=None) -> int:
    """Write the committed fixture corpus into Postgres. Returns chunks written.

    One document per `kind`, so the corpus lands as three documents rather than
    seventeen single-chunk ones -- retrieval behaves differently when chunks
    share a document, and the fixture should exercise that.

    Vectors come from the same committed cache the offline suite reads. Using
    the live model here instead would make any dense divergence unattributable:
    it could be a bug in the write path or just a different model.
    """
    from ledgerline.evals import embedder as cached_embedder
    from ledgerline.evals import load_corpus
    from ledgerline.ingest.pipeline import ChunkRow, Document, Issuer, ingest_document

    resolved = embedder or cached_embedder()
    issuer = Issuer(cik=FIXTURE_CIK, name=FIXTURE_ISSUER_NAME, ticker="NWM")

    by_kind: dict[str, list[dict]] = {}
    for record in load_corpus():
        by_kind.setdefault(record.get("kind", "filing"), []).append(record)

    written = 0
    for kind, records in sorted(by_kind.items()):
        document = Document(
            cik=FIXTURE_CIK,
            kind=kind,
            accession=f"fixture-{kind}",
            title=f"{FIXTURE_ISSUER_NAME} fixture {kind}",
            fiscal_period="FY2025",
        )
        chunks = [
            ChunkRow(
                external_id=record["id"],
                content=record["text"],
                ordinal=ordinal,
                section=record.get("section"),
                speaker=record.get("speaker"),
            )
            for ordinal, record in enumerate(records)
        ]
        written += ingest_document(conn, issuer, document, chunks, resolved).chunks
    conn.commit()
    return written
