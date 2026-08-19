from datetime import UTC, datetime

import pytest

from shared.evals.detection import Detection, GroundTruth, average_precision, iou, mask_iou
from sightline.dedupe import (
    DedupeConfig,
    ReportPoint,
    cluster,
    haversine_m,
    is_duplicate,
    sweep_threshold,
    text_similarity,
    with_threshold,
)
from sightline.ingest.mapillary import bbox_around, parse_image
from sightline.ingest.socrata import parse_chicago_potholes
from sightline.severity import CameraIntrinsics, band, estimate, extent_cm


class TestGeometry:
    def test_haversine_matches_known_distance(self):
        # One degree of latitude is ~111.2 km anywhere on the globe.
        assert haversine_m(0, 0, 1, 0) == pytest.approx(111_195, rel=0.001)

    def test_zero_distance(self):
        assert haversine_m(41.8781, -87.6298, 41.8781, -87.6298) == pytest.approx(0.0)

    def test_bbox_is_centred_and_ordered(self):
        west, south, east, north = bbox_around(41.8781, -87.6298, 40)
        assert west < -87.6298 < east
        assert south < 41.8781 < north


class TestDedupe:
    def _point(self, id_, lat, lon, text, **kwargs):
        return ReportPoint(id=id_, lat=lat, lon=lon, text=text, **kwargs)

    def test_near_identical_reports_are_duplicates(self):
        a = self._point("a", 41.88400, -87.63500, "Pothole in street needs repair urgently",
                        category="Pothole in Street", segment_id="s")
        b = self._point("b", 41.88403, -87.63500, "Pothole in street needs urgent repair",
                        category="Pothole in Street", segment_id="s")
        assert is_duplicate(a, b)

    def test_different_category_is_never_a_duplicate(self):
        a = self._point("a", 41.881, -87.630, "Deep pothole", category="Pothole in Street")
        b = self._point("b", 41.881, -87.630, "Deep pothole", category="Street Light Out")
        assert not is_duplicate(a, b)

    def test_beyond_radius_is_not_a_duplicate(self):
        a = self._point("a", 41.8781, -87.6298, "Pothole")
        b = self._point("b", 41.8791, -87.6298, "Pothole")  # ~111 m
        assert not is_duplicate(a, b)

    def test_time_window_separates_recurrences(self):
        # Same spot, same words, two years apart: the road was fixed and broke
        # again. That is a new work order, not a duplicate.
        a = self._point("a", 41.884, -87.635, "Pothole in street needs urgent repair",
                        reported_at=datetime(2024, 1, 1, tzinfo=UTC))
        b = self._point("b", 41.884, -87.635, "Pothole in street needs urgent repair",
                        reported_at=datetime(2026, 1, 1, tzinfo=UTC))
        assert not is_duplicate(a, b)

    def test_text_similarity_is_symmetric_and_bounded(self):
        a, b = "large pothole in road", "road has a large pothole"
        assert text_similarity(a, b) == text_similarity(b, a)
        assert 0.0 <= text_similarity(a, b) <= 1.0
        assert text_similarity("", "anything") == 0.0

    def test_clustering_is_transitive(self):
        points = [
            self._point("a", 41.8840, -87.6350, "pothole in street urgent repair",
                        segment_id="s"),
            self._point("b", 41.88403, -87.6350, "pothole in street urgent repair",
                        segment_id="s"),
            self._point("c", 41.88406, -87.6350, "pothole in street urgent repair",
                        segment_id="s"),
            self._point("z", 41.9000, -87.6350, "unrelated", segment_id="other"),
        ]
        groups = cluster(points)
        assert ["a", "b", "c"] in groups
        assert ["z"] in groups

    def test_threshold_is_actually_tunable(self):
        """Regression test for a knob that could not be turned.

        The thresholds used to be module constants consumed as function
        defaults, so they were bound at import: passing a different value
        changed nothing and a sweep returned one number repeated. Any tuning
        parameter has to be resolved at call time.
        """
        a = self._point("a", 41.884, -87.635, "pothole in street needs urgent repair",
                        segment_id="s")
        b = self._point("b", 41.88403, -87.635, "pothole in street needs repair urgently",
                        segment_id="s")
        assert is_duplicate(a, b, config=with_threshold(0.30))
        assert not is_duplicate(a, b, config=with_threshold(0.99))

    def test_sweep_produces_a_monotone_recall_curve(self):
        pairs = [
            (self._point("a", 41.884, -87.635, "pothole urgent repair", segment_id="s"),
             self._point("b", 41.88403, -87.635, "pothole urgent repair", segment_id="s"),
             True),
            (self._point("c", 41.884, -87.635, "pothole", segment_id="s"),
             self._point("d", 41.8848, -87.635, "streetlight", segment_id="t"),
             False),
        ]
        curve = sweep_threshold(pairs, [0.1, 0.5, 0.9, 1.1])
        recalls = [recall for _, _, recall in curve]
        # Raising the threshold can only ever lose recall.
        assert recalls == sorted(recalls, reverse=True)
        assert recalls[-1] == 0.0

    def test_config_rejects_weights_that_do_not_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            DedupeConfig(w_distance=0.5, w_text=0.5, w_segment=0.5)

    def test_clustering_covers_every_report_exactly_once(self):
        points = [self._point(str(i), 41.88 + i * 0.01, -87.63, "pothole") for i in range(5)]
        groups = cluster(points)
        flat = [pid for group in groups for pid in group]
        assert sorted(flat) == sorted(p.id for p in points)


class TestSeverity:
    intrinsics = CameraIntrinsics.from_fov(1024, 768)

    def test_focal_length_from_fov(self):
        # 70 degree horizontal FOV on a 1024px sensor.
        assert self.intrinsics.focal_px == pytest.approx(731.2, rel=0.01)

    def test_extent_scales_linearly_with_depth(self):
        near = extent_cm(mask_area_px=1000, depth_m=4.0, intrinsics=self.intrinsics)
        far = extent_cm(mask_area_px=1000, depth_m=8.0, intrinsics=self.intrinsics)
        assert far == pytest.approx(near * 2, rel=1e-6)

    def test_extent_scales_with_sqrt_of_area(self):
        small = extent_cm(mask_area_px=1000, depth_m=5.0, intrinsics=self.intrinsics)
        big = extent_cm(mask_area_px=4000, depth_m=5.0, intrinsics=self.intrinsics)
        assert big == pytest.approx(small * 2, rel=1e-6)

    def test_degenerate_inputs_are_zero_not_nan(self):
        assert extent_cm(mask_area_px=0, depth_m=5, intrinsics=self.intrinsics) == 0.0
        assert extent_cm(mask_area_px=100, depth_m=0, intrinsics=self.intrinsics) == 0.0

    @pytest.mark.parametrize(
        ("extent", "expected"),
        [(60.0, "urgent"), (45.0, "urgent"), (25.0, "priority"), (20.0, "priority"),
         (5.0, "routine"), (0.0, "routine")],
    )
    def test_banding_boundaries(self, extent, expected):
        assert band(extent)[0] == expected

    def test_abstains_on_low_confidence(self):
        result = estimate(mask_area_px=3000, depth_m=5.5, intrinsics=self.intrinsics,
                          detector_score=0.41)
        assert result.severity == "review"
        assert not result.confident
        assert "below" in result.note

    def test_abstains_beyond_reliable_depth(self):
        result = estimate(mask_area_px=5000, depth_m=30.0, intrinsics=self.intrinsics,
                          detector_score=0.87)
        assert result.severity == "review"
        assert "beyond reliable range" in result.note

    def test_confident_estimate_carries_sla_and_clause(self):
        result = estimate(mask_area_px=9000, depth_m=6.0, intrinsics=self.intrinsics,
                          detector_score=0.91)
        assert result.confident
        assert result.severity == "urgent"
        assert result.sla_days == 1
        assert result.standard_clause


class TestDetectionMetrics:
    def test_iou_identical_boxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_iou_disjoint_boxes(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_iou_half_overlap(self):
        # 10x10 boxes offset by 5 in x: intersection 50, union 150.
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)

    def test_perfect_detector_scores_one(self):
        preds = [(0, Detection((0, 0, 10, 10), "pothole", 0.9))]
        gts = [(0, GroundTruth((0, 0, 10, 10), "pothole"))]
        assert average_precision(preds, gts, "pothole", 0.5) == pytest.approx(1.0)

    def test_box_in_the_wrong_image_does_not_match(self):
        preds = [(1, Detection((0, 0, 10, 10), "pothole", 0.9))]
        gts = [(0, GroundTruth((0, 0, 10, 10), "pothole"))]
        assert average_precision(preds, gts, "pothole", 0.5) == 0.0

    def test_high_scoring_false_positive_lowers_ap(self):
        preds = [
            (0, Detection((50, 50, 60, 60), "pothole", 0.95)),  # spurious, ranked first
            (0, Detection((0, 0, 10, 10), "pothole", 0.90)),
        ]
        gts = [(0, GroundTruth((0, 0, 10, 10), "pothole"))]
        assert average_precision(preds, gts, "pothole", 0.5) == pytest.approx(0.5)

    def test_trailing_false_positive_does_not_lower_ap(self):
        """Standard AP semantics, pinned because it surprises people.

        A false positive ranked *below* every true positive adds no recall, so
        all-point-interpolated AP is unchanged. It matters operationally: mAP
        will not notice a detector that emits a tail of low-confidence junk,
        which is why calibration and a confidence floor are measured
        separately rather than trusted to the headline number.
        """
        preds = [
            (0, Detection((0, 0, 10, 10), "pothole", 0.90)),
            (0, Detection((50, 50, 60, 60), "pothole", 0.10)),
        ]
        gts = [(0, GroundTruth((0, 0, 10, 10), "pothole"))]
        assert average_precision(preds, gts, "pothole", 0.5) == pytest.approx(1.0)

    def test_duplicate_box_on_one_target_is_not_double_counted(self):
        preds = [
            (0, Detection((0, 0, 10, 10), "pothole", 0.9)),
            (0, Detection((1, 1, 11, 11), "pothole", 0.8)),  # same target
        ]
        gts = [(0, GroundTruth((0, 0, 10, 10), "pothole"))]
        # Recall caps at 1.0: the second box cannot match an already-matched
        # target, so it is scored as a false positive even though its IoU is
        # above threshold.
        assert average_precision(preds, gts, "pothole", 0.5) == pytest.approx(1.0)

    def test_mask_iou(self):
        pred = [[True, True], [False, False]]
        gold = [[True, False], [False, False]]
        assert mask_iou(pred, gold) == pytest.approx(0.5)


class TestIngestParsing:
    def test_socrata_reads_nested_location(self):
        rows = [{
            "sr_number": "SR24-001", "sr_type": "Pothole in Street",
            "created_date": "2026-03-01T09:15:00.000",
            "status": "Open", "street_address": "1420 W Example St",
            "location": {"type": "Point", "coordinates": [-87.6298, 41.8781]},
        }]
        report = parse_chicago_potholes(rows)[0]
        assert (report.lat, report.lon) == (41.8781, -87.6298)
        assert report.reported_at.year == 2026
        assert report.has_location

    def test_socrata_falls_back_to_flat_columns(self):
        rows = [{"sr_number": "SR24-002", "latitude": "41.9", "longitude": "-87.7"}]
        report = parse_chicago_potholes(rows)[0]
        assert (report.lat, report.lon) == (41.9, -87.7)

    def test_socrata_tolerates_missing_coordinates(self):
        rows = [{"sr_number": "SR24-003"}]
        report = parse_chicago_potholes(rows)[0]
        assert not report.has_location

    def test_socrata_drops_rows_with_no_identifier(self):
        assert parse_chicago_potholes([{"sr_type": "Pothole in Street"}]) == []

    def test_mapillary_converts_epoch_millis(self):
        image = parse_image({
            "id": "123", "captured_at": 1751328000000,
            "geometry": {"coordinates": [-87.6298, 41.8781]},
            "width": 1024, "height": 768, "compass_angle": 91.5,
        })
        assert image.lat == 41.8781
        assert image.captured_at.year == 2025
        assert image.compass_angle == 91.5

    def test_mapillary_tolerates_missing_fields(self):
        image = parse_image({"id": "456"})
        assert image.external_id == "456"
        assert image.captured_at is None
        assert image.width is None
