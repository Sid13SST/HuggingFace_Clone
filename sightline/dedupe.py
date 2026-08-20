"""Duplicate resolution for 311 reports.

Mirrors `sightline.duplicate_candidates` in schema.sql. Production runs the
SQL, because pushing the spatial blocking into the GiST index is what keeps it
tractable; this Python version exists so the scoring can be tuned and gated in
CI without a database.

If you change the weights here, change them there too -- and re-run
`evalctl run sightline` before believing the change helped.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Includes 311-specific noise: almost every report says "please fix" and
# "reported", so those words carry no signal about *which* defect it is.
_NOISE = frozenset(
    (
        "the a an of on at in near by and to is are was were please fix "
        "report reported there this that it its my our"
    ).split()
)


@dataclass(frozen=True)
class DedupeConfig:
    """Every tuning knob, in one object.

    These started life as module constants used as function defaults, which
    meant they were bound at import and could not actually be varied -- a
    threshold sweep silently returned the same number for every value. Passing
    config explicitly is what makes `sweep_threshold` honest, and an ablation
    table you cannot trust is worse than none.

    Distance dominates the weighting because it is the only signal that is
    never wrong; text similarity is a tiebreak, not a driver, since 311 free
    text is written by whoever was annoyed enough to file it.

    Mirrored in `sightline.duplicate_candidates` (schema.sql). Change both.
    """

    w_distance: float = 0.45
    w_text: float = 0.35
    w_segment: float = 0.20
    radius_m: float = 30.0
    window_days: int = 45
    threshold: float = 0.62
    #: The text similarity to score with. `None` means the lexical baseline.
    #:
    #: This lives in the config rather than beside it because a threshold is
    #: only meaningful against a particular similarity scale -- Jaccard over
    #: 311 text sits near zero for unrelated reports, cosine sits near 0.4, and
    #: a threshold calibrated for one is wrong for the other by construction.
    #: Keeping them in separate places is how you end up running a semantic
    #: similarity against a lexical threshold and reporting the result.
    similarity_fn: Callable[[str, str], float] | None = None

    def __post_init__(self) -> None:
        total = self.w_distance + self.w_text + self.w_segment
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0, got {total}")

    def similarity(self) -> Callable[[str, str], float]:
        return self.similarity_fn or text_similarity


#: The lexical baseline, kept as the permanent control for the embedding
#: ablation. Not what production runs -- see `sightline.similarity`, which
#: needs an embedder and therefore cannot be constructed here.
DEFAULT_CONFIG = DedupeConfig()
LEXICAL_CONFIG = DEFAULT_CONFIG


@dataclass(frozen=True)
class ReportPoint:
    id: str
    lat: float
    lon: float
    text: str = ""
    category: str | None = None
    segment_id: str | None = None
    reported_at: datetime | None = None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def text_similarity(a: str, b: str) -> float:
    """Jaccard over content tokens.

    Stands in for the sentence-embedding cosine used in production. It is a
    weaker signal, which is the point: it is the baseline the embedding has to
    beat on the same labelled pairs before it earns its place in the pipeline.
    """
    ta = {t for t in _TOKEN_RE.findall(a.lower()) if t not in _NOISE}
    tb = {t for t in _TOKEN_RE.findall(b.lower()) if t not in _NOISE}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def pair_score(
    a: ReportPoint,
    b: ReportPoint,
    *,
    config: DedupeConfig | None = None,
    similarity_fn: Callable[[str, str], float] | None = None,
) -> float:
    """Combined duplicate score in [0, 1]. Zero if the pair is not a candidate.

    `similarity_fn` is the seam where the sentence-embedding cosine drops in
    to replace the lexical baseline, so the swap can be measured on the same
    labelled pairs rather than argued about.
    """
    cfg = config or DEFAULT_CONFIG
    if a.category and b.category and a.category != b.category:
        return 0.0

    distance = haversine_m(a.lat, a.lon, b.lat, b.lon)
    if distance > cfg.radius_m:
        return 0.0

    sim = similarity_fn or cfg.similarity()
    proximity = 1.0 - min(distance / cfg.radius_m, 1.0)
    same_segment = (
        1.0 if (a.segment_id is not None and a.segment_id == b.segment_id) else 0.0
    )
    return (
        cfg.w_distance * proximity
        + cfg.w_text * sim(a.text, b.text)
        + cfg.w_segment * same_segment
    )


def is_duplicate(
    a: ReportPoint,
    b: ReportPoint,
    *,
    config: DedupeConfig | None = None,
    similarity_fn: Callable[[str, str], float] | None = None,
) -> bool:
    cfg = config or DEFAULT_CONFIG
    # Same spot, same words, two years apart means the road was repaired and
    # failed again -- a new work order, not a duplicate.
    if (
        a.reported_at
        and b.reported_at
        and abs(a.reported_at - b.reported_at) > timedelta(days=cfg.window_days)
    ):
        return False
    return pair_score(a, b, config=cfg, similarity_fn=similarity_fn) >= cfg.threshold


def sweep_threshold(
    pairs: list[tuple[ReportPoint, ReportPoint, bool]],
    thresholds: list[float],
    *,
    config: DedupeConfig | None = None,
) -> list[tuple[float, float, float]]:
    """(threshold, precision, recall) across a sweep.

    The curve, not the point estimate, is what tells you whether 0.62 is a
    considered choice or the first number that happened to work.
    """
    base = config or DEFAULT_CONFIG
    scores = [(pair_score(a, b, config=base), label) for a, b, label in pairs]

    curve: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        tp = sum(1 for s, label in scores if s >= threshold and label)
        fp = sum(1 for s, label in scores if s >= threshold and not label)
        fn = sum(1 for s, label in scores if s < threshold and label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        curve.append((threshold, precision, recall))
    return curve


def with_threshold(threshold: float, config: DedupeConfig | None = None) -> DedupeConfig:
    return replace(config or DEFAULT_CONFIG, threshold=threshold)


def cluster(
    reports: list[ReportPoint],
    *,
    config: DedupeConfig | None = None,
) -> list[list[str]]:
    """Group reports into defects via union-find over duplicate edges.

    Transitive closure is the right call here and it is also the risk: three
    reports strung 25 m apart along a block become one defect. That is why the
    eval measures *pairwise* precision -- a false merge is the expensive error,
    because the second pothole silently leaves the queue.
    """
    parent = {r.id: r.id for r in reports}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, a in enumerate(reports):
        for b in reports[i + 1 :]:
            if is_duplicate(a, b, config=config):
                union(a.id, b.id)

    groups: dict[str, list[str]] = {}
    for report in reports:
        groups.setdefault(find(report.id), []).append(report.id)
    return sorted((sorted(v) for v in groups.values()), key=lambda g: g[0])
