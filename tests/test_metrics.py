import math

import pytest

from shared.evals.metrics import (
    UnparseableNumber,
    binary_prf,
    cohens_kappa,
    expected_calibration_error,
    ndcg_at_k,
    numeric_match,
    parse_number,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestParseNumber:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1,842.6", 1842.6),
            ("$1,842.6 million", 1_842_600_000.0),
            ("$312.4 million", 312_400_000.0),
            ("34.2%", 34.2),
            ("6,480", 6480.0),
            ("1.2 billion", 1_200_000_000.0),
            ("-45.5", -45.5),
            ("(1,234)", -1234.0),  # accounting negative
            ("75", 75.0),
        ],
    )
    def test_parses(self, text, expected):
        assert parse_number(text) == pytest.approx(expected)

    def test_scale_hint_applies_when_string_has_no_scale(self):
        # Table header said "in thousands"; the cell just says "1,234".
        assert parse_number("1,234", scale_hint=1000) == pytest.approx(1_234_000)

    def test_explicit_scale_beats_hint(self):
        # "$4.1 billion" inside a thousands-scaled table is still 4.1e9 --
        # the writer overrode the header on purpose.
        assert parse_number("$4.1 billion", scale_hint=1000) == pytest.approx(4.1e9)

    def test_percent_is_not_rescaled_by_hint(self):
        assert parse_number("22.6%", scale_hint=1_000_000) == pytest.approx(22.6)

    def test_rejects_garbage(self):
        with pytest.raises(UnparseableNumber):
            parse_number("not disclosed")


class TestNumericMatch:
    def test_tolerates_rounding(self):
        assert numeric_match("$1,842.6 million", "1842600000")
        assert numeric_match("34.2%", "34.2")

    def test_rejects_wrong_number(self):
        assert not numeric_match("$1,639.3 million", "1842600000")

    def test_none_is_never_a_match(self):
        assert not numeric_match(None, "100")

    def test_unparseable_prediction_is_not_a_match(self):
        assert not numeric_match("approximately half", "50")

    def test_scale_confusion_is_caught(self):
        # thousands vs millions is the classic silent failure
        assert not numeric_match("1,842.6 thousand", "$1,842.6 million")


class TestRetrievalMetrics:
    def test_recall_and_precision(self):
        ranked = ["a", "b", "c", "d"]
        assert recall_at_k(["a", "d"], ranked, 2) == pytest.approx(0.5)
        assert recall_at_k(["a", "d"], ranked, 4) == pytest.approx(1.0)
        assert precision_at_k(["a", "d"], ranked, 4) == pytest.approx(0.5)

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["c"], ["a", "b", "c"]) == pytest.approx(1 / 3)
        assert reciprocal_rank(["z"], ["a", "b", "c"]) == 0.0

    def test_ndcg_rewards_earlier_hits(self):
        early = ndcg_at_k(["a"], ["a", "b", "c"], 3)
        late = ndcg_at_k(["a"], ["b", "c", "a"], 3)
        assert early == pytest.approx(1.0)
        assert late < early

    def test_ndcg_is_one_for_perfect_ordering(self):
        assert ndcg_at_k(["a", "b"], ["a", "b", "c"], 3) == pytest.approx(1.0)

    def test_empty_gold_is_zero_not_a_crash(self):
        assert ndcg_at_k([], ["a"], 3) == 0.0
        assert recall_at_k([], ["a"], 3) == 0.0


class TestClassificationMetrics:
    def test_binary_prf(self):
        prf = binary_prf([True, True, False, False], [True, False, True, False])
        assert prf.precision == pytest.approx(0.5)
        assert prf.recall == pytest.approx(0.5)
        assert prf.f1 == pytest.approx(0.5)
        assert prf.support == 2

    def test_prf_handles_no_positive_predictions(self):
        prf = binary_prf([True, False], [False, False])
        assert prf.precision == 0.0
        assert prf.recall == 0.0
        assert prf.f1 == 0.0

    def test_kappa_perfect_and_chance(self):
        assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
        # Total disagreement on a balanced two-label set is worse than chance.
        assert cohens_kappa(["a", "b"], ["b", "a"]) < 0

    def test_kappa_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cohens_kappa(["a"], ["a", "b"])

    def test_ece_zero_for_perfect_calibration(self):
        # Every prediction at confidence 1.0 and every one correct.
        assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)

    def test_ece_flags_overconfidence(self):
        error = expected_calibration_error([0.95, 0.95, 0.95, 0.95], [True, False, False, False])
        assert error == pytest.approx(0.70, abs=0.01)

    def test_ece_empty_is_zero(self):
        assert expected_calibration_error([], []) == 0.0
        assert not math.isnan(expected_calibration_error([], []))
