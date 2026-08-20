"""Sightline eval suites.

Two layers, deliberately kept apart:

  * perception  -- mAP, calibration, extent error. Classical ML metrics.
  * decision    -- dedupe precision/recall, band accuracy, abstention.

They fail differently. A detector can improve mAP while its confidence
calibration degrades, which leaves the headline number looking healthy while
crews get routed by a score that no longer means what the router thinks it
means. Reporting both is the whole point.
"""

from __future__ import annotations

from shared.config import REPO_ROOT
from shared.evals.dataset import Example
from shared.evals.detection import Detection, GroundTruth, mean_average_precision
from shared.evals.metrics import binary_prf, expected_calibration_error, mean
from shared.evals.registry import Gate, Suite, register_suite
from sightline.dedupe import LEXICAL_CONFIG, DedupeConfig, ReportPoint, is_duplicate
from sightline.severity import CameraIntrinsics, estimate

HERE = REPO_ROOT / "sightline" / "evals"


# --------------------------------------------------------------------------
# suite: dedupe
# --------------------------------------------------------------------------


def _to_point(raw: dict) -> ReportPoint:
    return ReportPoint(
        id=raw["id"],
        lat=float(raw["lat"]),
        lon=float(raw["lon"]),
        text=raw.get("text", ""),
        category=raw.get("category"),
        segment_id=raw.get("segment_id"),
    )


EMBEDDING_CACHE_PATH = HERE / "fixtures" / "embeddings.npz"


def dedupe_embedder():
    """Committed vectors for every report text in the golden set.

    No lru_cache: the suites run once each and the embedder is cheap to build,
    while a cached instance would outlive a fixture edit and serve stale
    vectors to the second suite in the same process.
    """
    from shared.embeddings import CachedEmbedder

    return CachedEmbedder.from_npz(EMBEDDING_CACHE_PATH)


def semantic_dedupe_config() -> DedupeConfig:
    from sightline.similarity import semantic_config

    return semantic_config(dedupe_embedder())


def _score_dedupe(examples: list[Example], config: DedupeConfig) -> dict[str, float]:
    """Shared scoring, so the lexical and semantic paths are comparable.

    Same labelled pairs, same metrics, same slices -- only the similarity
    function and its matched threshold vary. That is what makes the difference
    an ablation rather than an anecdote.
    """
    y_true = [bool(e.expected["duplicate"]) for e in examples]
    y_pred = [
        is_duplicate(_to_point(e.inputs["a"]), _to_point(e.inputs["b"]), config=config)
        for e in examples
    ]

    prf = binary_prf(y_true, y_pred)
    metrics: dict[str, float] = {
        **prf.as_dict(prefix="pair_"),
        "accuracy": mean(
            1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred, strict=True)
        ),
        # False merges are the expensive error: the duplicate report vanishes
        # and the second defect never reaches a crew. Tracked separately so it
        # can be gated even when F1 looks fine.
        "false_merge_rate": mean(
            1.0 if (p and not t) else 0.0
            for t, p in zip(y_true, y_pred, strict=True)
        ),
    }

    hard = [
        (t, p)
        for e, t, p in zip(examples, y_true, y_pred, strict=True)
        if "hard" in e.tags
    ]
    if hard:
        metrics["accuracy.hard"] = mean(
            1.0 if t == p else 0.0 for t, p in hard
        )
    return metrics


def run_dedupe(examples: list[Example]) -> dict[str, float]:
    """The system under test: sentence-embedding cosine."""
    return _score_dedupe(examples, semantic_dedupe_config())


def run_dedupe_lexical(examples: list[Example]) -> dict[str, float]:
    """The Jaccard control, kept in CI forever.

    A "before" number that only lives in a README decays the moment someone
    edits the golden set. Re-measuring it every run keeps the ablation true.
    """
    return _score_dedupe(examples, LEXICAL_CONFIG)


# --------------------------------------------------------------------------
# suite: severity
# --------------------------------------------------------------------------


def run_severity(examples: list[Example]) -> dict[str, float]:
    errors: list[float] = []
    band_hits: list[float] = []
    abstain_true: list[bool] = []
    abstain_pred: list[bool] = []

    for example in examples:
        inputs = example.inputs
        intrinsics = CameraIntrinsics.from_fov(int(inputs["width"]), int(inputs["height"]))
        result = estimate(
            mask_area_px=float(inputs["mask_area_px"]),
            depth_m=float(inputs["depth_m"]),
            intrinsics=intrinsics,
            detector_score=float(inputs["detector_score"]),
            oblique_correction=float(inputs.get("oblique_correction", 1.0)),
        )

        expected_severity = example.expected["severity"]
        abstain_true.append(expected_severity == "review")
        abstain_pred.append(not result.confident)
        band_hits.append(1.0 if result.severity == expected_severity else 0.0)

        gold_extent = example.expected.get("extent_cm")
        if gold_extent and result.confident:
            errors.append(abs(result.extent_cm - float(gold_extent)) / float(gold_extent))

    abstention = binary_prf(abstain_true, abstain_pred)
    return {
        # Mean absolute relative error on extent. This is the number behind any
        # "+/- N%" claim about severity, and it is measured only on the cases
        # the system chose to answer.
        "extent_abs_rel_error": mean(errors),
        "extent_within_15pct": mean(1.0 if e <= 0.15 else 0.0 for e in errors),
        "band_accuracy": mean(band_hits),
        **abstention.as_dict(prefix="abstain_"),
    }


# --------------------------------------------------------------------------
# suite: detection
# --------------------------------------------------------------------------


def run_detection(examples: list[Example]) -> dict[str, float]:
    predictions: list[tuple[int, Detection]] = []
    ground_truths: list[tuple[int, GroundTruth]] = []
    confidences: list[float] = []
    correct: list[bool] = []

    for image_index, example in enumerate(examples):
        gold_boxes = example.expected["boxes"]
        for raw in gold_boxes:
            ground_truths.append(
                (image_index, GroundTruth(box=tuple(raw["box"]), label=raw["label"]))
            )
        for raw in example.inputs["predictions"]:
            detection = Detection(
                box=tuple(raw["box"]), label=raw["label"], score=float(raw["score"])
            )
            predictions.append((image_index, detection))

            # Calibration: is a box the model called 0.9 right nine times in ten?
            matched = any(
                gold["label"] == detection.label
                and _iou(tuple(gold["box"]), detection.box) >= 0.5
                for gold in gold_boxes
            )
            confidences.append(detection.score)
            correct.append(matched)

    metrics = mean_average_precision(predictions, ground_truths)
    metrics["ece"] = expected_calibration_error(confidences, correct)
    metrics["n_predictions"] = float(len(predictions))
    return metrics


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    from shared.evals.detection import iou

    return iou(a, b)  # type: ignore[arg-type]


register_suite(
    Suite(
        name="sightline.dedupe",
        project="sightline",
        dataset=HERE / "datasets" / "dedupe_pairs.jsonl",
        run=run_dedupe,
        description="Spatial blocking plus sentence-embedding cosine.",
        gates=[
            Gate("pair_precision", min_value=0.80, max_regression=0.03),
            Gate("pair_recall", min_value=0.85, max_regression=0.05),
            Gate("pair_f1", max_regression=0.03),
            # Lower is better, so this needs a ceiling and a flipped regression
            # direction. It was written as `max_regression=0.05` on a
            # higher-is-better gate, which meant a false-merge rate climbing
            # from 0.00 to 0.10 read as a 0.10 *improvement* and passed. The
            # one failure the comment called out as worth blocking a merge over
            # was the one failure this gate could not detect.
            Gate("false_merge_rate", max_value=0.05, higher_is_better=False),
        ],
    )
)

register_suite(
    Suite(
        name="sightline.dedupe_lexical",
        project="sightline",
        dataset=HERE / "datasets" / "dedupe_pairs.jsonl",
        run=run_dedupe_lexical,
        description="Jaccard control. The permanent 'before' for the embedding ablation.",
        # Ungated on purpose: this suite is not supposed to improve. Gating it
        # would create pressure to strengthen the straw man.
        gates=[],
    )
)

register_suite(
    Suite(
        name="sightline.severity",
        project="sightline",
        dataset=HERE / "datasets" / "severity.jsonl",
        run=run_severity,
        description="Depth x mask area vs hand-measured extent, plus abstention behaviour.",
        gates=[
            Gate("band_accuracy", min_value=0.70, max_regression=0.05),
            Gate("abstain_recall", min_value=0.90),
        ],
    )
)

register_suite(
    Suite(
        name="sightline.detection",
        project="sightline",
        dataset=HERE / "datasets" / "detection.jsonl",
        run=run_detection,
        description="mAP@50-95 and calibration on the held-out city split.",
        gates=[
            Gate("map_50", max_regression=0.03),
            Gate("map_50_95", max_regression=0.03),
        ],
    )
)
