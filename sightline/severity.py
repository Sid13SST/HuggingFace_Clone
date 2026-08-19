"""Severity as a measurement, not an adjective.

Monocular depth plus a segmentation mask gives an approximate physical extent.
That number is what the municipal standards key on, so the chain from pixel to
SLA is auditable end to end: a supervisor can ask why a defect was called
urgent and get "38 cm across, clause 4.2.1" rather than "the model thought so".

The pinhole relation used throughout:

    real_size = pixel_size * depth / focal_length_px

which assumes the defect surface is roughly perpendicular to the camera axis.
For road surface shot from a vehicle that is wrong by the cosine of the
incidence angle, so `oblique_correction` exists and the eval reports error
against hand-measured ground truth rather than pretending the model is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Synthetic stand-in for a real municipal standard. Replace with the actual
# clause numbers from the city's published maintenance manual before quoting
# any of this to anyone -- the numbers below are illustrative.
SEVERITY_BANDS: tuple[tuple[str, float, int, str], ...] = (
    # (severity, min_extent_cm, sla_days, standard_clause)
    ("urgent", 45.0, 1, "4.2.1(a)"),
    ("priority", 20.0, 7, "4.2.1(b)"),
    ("routine", 0.0, 30, "4.2.2"),
)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Minimal intrinsics.

    `focal_px` is the only field the size estimate needs. Mapillary exposes
    camera parameters for many sequences; when it does not, estimating focal
    length from image width and a typical 70-degree horizontal field of view is
    accurate enough for banding and is what `from_fov` does.
    """

    focal_px: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, fov_degrees: float = 70.0) -> CameraIntrinsics:
        focal = (width / 2) / math.tan(math.radians(fov_degrees / 2))
        return cls(focal_px=focal, width=width, height=height)


@dataclass(frozen=True)
class SeverityEstimate:
    extent_cm: float
    area_m2: float
    severity: str
    sla_days: int
    standard_clause: str
    confident: bool
    note: str = ""


def extent_cm(
    *,
    mask_area_px: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    oblique_correction: float = 1.0,
) -> float:
    """Longest-axis extent in centimetres, approximating the mask as a disc.

    A disc rather than the bounding box: boxes overstate extent badly for the
    elongated cracks that make up most of the routine band, and overstating
    routine defects is how a queue loses its ordering.
    """
    if mask_area_px <= 0 or depth_m <= 0 or intrinsics.focal_px <= 0:
        return 0.0
    metres_per_px = depth_m / intrinsics.focal_px
    area_m2 = mask_area_px * (metres_per_px**2) * oblique_correction
    diameter_m = 2.0 * math.sqrt(area_m2 / math.pi)
    return diameter_m * 100.0


def area_m2(
    *, mask_area_px: float, depth_m: float, intrinsics: CameraIntrinsics,
    oblique_correction: float = 1.0,
) -> float:
    if mask_area_px <= 0 or depth_m <= 0 or intrinsics.focal_px <= 0:
        return 0.0
    metres_per_px = depth_m / intrinsics.focal_px
    return mask_area_px * (metres_per_px**2) * oblique_correction


def band(extent: float) -> tuple[str, int, str]:
    for severity, threshold, sla_days, clause in SEVERITY_BANDS:
        if extent >= threshold:
            return severity, sla_days, clause
    return SEVERITY_BANDS[-1][0], SEVERITY_BANDS[-1][2], SEVERITY_BANDS[-1][3]


def estimate(
    *,
    mask_area_px: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    detector_score: float,
    oblique_correction: float = 1.0,
    min_confidence: float = 0.55,
    max_depth_m: float = 25.0,
) -> SeverityEstimate:
    """Full severity decision, including when to refuse to make one.

    Two abstain conditions, both learned the boring way: monocular depth
    degrades badly past ~25 m, and a low-confidence detection sized precisely
    is worse than no answer, because a precise number invites trust the
    pipeline has not earned.
    """
    extent = extent_cm(
        mask_area_px=mask_area_px,
        depth_m=depth_m,
        intrinsics=intrinsics,
        oblique_correction=oblique_correction,
    )
    area = area_m2(
        mask_area_px=mask_area_px,
        depth_m=depth_m,
        intrinsics=intrinsics,
        oblique_correction=oblique_correction,
    )
    severity, sla_days, clause = band(extent)

    if detector_score < min_confidence:
        return SeverityEstimate(
            extent, area, "review", 0, clause, False,
            note=f"detector score {detector_score:.2f} below {min_confidence:.2f}",
        )
    if depth_m > max_depth_m:
        return SeverityEstimate(
            extent, area, "review", 0, clause, False,
            note=f"depth {depth_m:.1f} m beyond reliable range {max_depth_m:.0f} m",
        )
    return SeverityEstimate(extent, area, severity, sla_days, clause, True)
