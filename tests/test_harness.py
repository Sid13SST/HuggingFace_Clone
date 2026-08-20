"""Tests for the eval harness itself.

The harness is what every other number in this repo depends on, so a broken
gate has to fail loudly here rather than silently pass a regression through CI.
"""

import json

import pytest

from shared.evals.dataset import Example, filter_by_tag, load_jsonl
from shared.evals.registry import Gate, Suite, list_suites
from shared.evals.report import render_markdown
from shared.evals.runner import RunResult, evaluate_gates, run_suite


class TestDataset:
    def test_loads_and_skips_comments(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            "# a comment\n"
            '{"id": "1", "inputs": {"q": "x"}, "expected": {"a": 1}, "tags": ["t"]}\n'
            "\n"
            '{"id": "2", "inputs": {"q": "y"}, "expected": {"a": 2}}\n',
            encoding="utf-8",
        )
        examples = load_jsonl(path)
        assert [e.id for e in examples] == ["1", "2"]
        assert filter_by_tag(examples, "t") == [examples[0]]

    def test_duplicate_ids_are_rejected(self, tmp_path):
        path = tmp_path / "d.jsonl"
        line = '{"id": "1", "inputs": {}, "expected": {}}\n'
        path.write_text(line * 2, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate example id"):
            load_jsonl(path)

    def test_missing_keys_name_the_line(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text('{"id": "1"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=r":1:.*expected"):
            load_jsonl(path)

    def test_empty_file_is_an_error(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text("# only a comment\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_jsonl(path)

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "nope.jsonl")


class TestGates:
    def test_floor_blocks_a_bad_absolute_value(self):
        results = evaluate_gates({"m": 0.5}, {}, [Gate("m", min_value=0.6)])
        assert not results[0].passed
        assert "below floor" in results[0].reason

    def test_regression_blocks_a_drop_against_baseline(self):
        results = evaluate_gates({"m": 0.90}, {"m": 0.95}, [Gate("m", max_regression=0.02)])
        assert not results[0].passed
        assert "regressed" in results[0].reason

    def test_small_drop_within_tolerance_passes(self):
        results = evaluate_gates({"m": 0.94}, {"m": 0.95}, [Gate("m", max_regression=0.02)])
        assert results[0].passed

    def test_improvement_always_passes(self):
        results = evaluate_gates({"m": 0.99}, {"m": 0.95}, [Gate("m", max_regression=0.02)])
        assert results[0].passed
        assert results[0].delta == pytest.approx(0.04)

    def test_missing_metric_is_a_failure_not_a_skip(self):
        # A suite that quietly stops reporting a gated metric must not pass.
        results = evaluate_gates({}, {"m": 0.9}, [Gate("m", max_regression=0.02)])
        assert not results[0].passed
        assert "not reported" in results[0].reason

    def test_no_baseline_yet_passes_but_says_so(self):
        results = evaluate_gates({"m": 0.5}, {}, [Gate("m", max_regression=0.02)])
        assert results[0].passed
        assert "no baseline" in results[0].reason

    def test_ceiling_blocks_a_metric_where_lower_is_better(self):
        """Calibration error, fabrication rate, cost per run: the harness had
        no way to gate any of them until `max_value` existed, which is why they
        were all reported and none were enforced."""
        results = evaluate_gates({"ece": 0.3}, {}, [Gate("ece", max_value=0.2)])
        assert not results[0].passed
        assert "above ceiling" in results[0].reason

    def test_ceiling_passes_at_the_boundary(self):
        assert evaluate_gates({"ece": 0.2}, {}, [Gate("ece", max_value=0.2)])[0].passed

    def test_regression_direction_follows_higher_is_better(self):
        """An error rate rising from 0.10 to 0.20 is a regression. With the
        default direction the guard would read that as a 0.10 improvement and
        fire on the fix instead of the break."""
        gate = Gate("err", max_regression=0.02, higher_is_better=False)
        worse = evaluate_gates({"err": 0.20}, {"err": 0.10}, [gate])
        better = evaluate_gates({"err": 0.05}, {"err": 0.10}, [gate])
        assert not worse[0].passed
        assert better[0].passed

    def test_a_gate_can_carry_a_floor_and_a_ceiling_at_once(self):
        band = Gate("rate", min_value=0.4, max_value=0.6)
        assert evaluate_gates({"rate": 0.5}, {}, [band])[0].passed
        assert not evaluate_gates({"rate": 0.3}, {}, [band])[0].passed
        assert not evaluate_gates({"rate": 0.7}, {}, [band])[0].passed

    def test_describe_mentions_every_bound(self):
        described = Gate("m", min_value=0.1, max_value=0.9, max_regression=0.02).describe()
        assert ">= 0.1" in described and "<= 0.9" in described and "regression" in described


class TestRunner:
    def test_a_raising_suite_fails_rather_than_crashing(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text('{"id": "1", "inputs": {}, "expected": {}}\n', encoding="utf-8")

        def boom(_examples):
            raise RuntimeError("model server is down")

        result = run_suite(
            Suite(name="t.boom", project="t", dataset=path, run=boom, gates=[Gate("m")])
        )
        assert not result.passed
        assert "model server is down" in result.error

    def test_successful_run_reports_metrics_and_count(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            '{"id": "1", "inputs": {}, "expected": {}}\n'
            '{"id": "2", "inputs": {}, "expected": {}}\n',
            encoding="utf-8",
        )
        result = run_suite(
            Suite(name="t.ok", project="t", dataset=path, run=lambda ex: {"m": len(ex) / 2})
        )
        assert result.passed
        assert result.n_examples == 2
        assert result.metrics["m"] == 1.0


class TestReport:
    def test_markdown_contains_verdict_and_deltas(self):
        result = RunResult(
            suite="s", project="p", metrics={"m": 0.9}, baseline={"m": 0.8},
            n_examples=3, seconds=0.1,
        )
        markdown = render_markdown([result])
        assert "**PASS**" in markdown
        assert "+0.1000" in markdown

    def test_markdown_reports_an_errored_suite(self):
        result = RunResult(
            suite="s", project="p", metrics={}, baseline={}, n_examples=0,
            seconds=0.0, error="RuntimeError: nope",
        )
        markdown = render_markdown([result])
        assert "**FAIL**" in markdown
        assert "errored" in markdown


class TestRegisteredSuites:
    """The committed suites must actually load and run.

    This is the test that catches a golden set edited into invalid JSON, which
    is otherwise only discovered when CI fails on an unrelated pull request.
    """

    def test_both_projects_register_suites(self):
        names = {s.name for s in list_suites()}
        assert {"ledgerline.retrieval", "ledgerline.numeric"} <= names
        assert {"sightline.dedupe", "sightline.severity", "sightline.detection"} <= names

    @pytest.mark.parametrize(
        "suite_name",
        ["ledgerline.retrieval", "ledgerline.numeric",
         "sightline.dedupe", "sightline.severity", "sightline.detection"],
    )
    def test_suite_runs_without_error(self, suite_name):
        from shared.evals.registry import get_suite

        result = run_suite(get_suite(suite_name))
        assert result.error is None, result.error
        assert result.n_examples > 0
        assert result.metrics

    def test_every_gated_metric_is_actually_reported(self):
        """A gate on a metric the suite never emits is a gate that never fires."""
        from shared.evals.registry import get_suite

        for suite in list_suites():
            result = run_suite(get_suite(suite.name))
            for gate in suite.gates:
                assert gate.metric in result.metrics, (
                    f"{suite.name} gates on {gate.metric!r} but does not report it"
                )

    def test_golden_sets_are_valid_jsonl(self):
        for suite in list_suites():
            with suite.dataset.open(encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        try:
                            json.loads(stripped)
                        except json.JSONDecodeError as exc:
                            pytest.fail(f"{suite.dataset}:{lineno}: {exc}")


class TestExample:
    def test_from_dict_requires_the_core_keys(self):
        with pytest.raises(ValueError, match="missing required keys"):
            Example.from_dict({"id": "1", "inputs": {}})
