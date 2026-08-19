from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from shared.config import REPO_ROOT
from shared.evals.dataset import load_jsonl
from shared.evals.registry import Gate, Suite, list_suites
from shared.logging import get_logger

log = get_logger(__name__)

BASELINE_DIR = REPO_ROOT / "evals" / "baselines"


@dataclass(frozen=True)
class GateResult:
    gate: Gate
    value: float | None
    baseline: float | None
    passed: bool
    reason: str

    @property
    def delta(self) -> float | None:
        if self.value is None or self.baseline is None:
            return None
        return self.value - self.baseline


@dataclass
class RunResult:
    suite: str
    project: str
    metrics: dict[str, float]
    baseline: dict[str, float]
    n_examples: int
    seconds: float
    gates: list[GateResult] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(g.passed for g in self.gates)

    def deltas(self) -> dict[str, float | None]:
        return {
            key: (value - self.baseline[key] if key in self.baseline else None)
            for key, value in self.metrics.items()
        }


def baseline_path(suite_name: str) -> Path:
    return BASELINE_DIR / f"{suite_name}.json"


def load_baseline(suite_name: str) -> dict[str, float]:
    path = baseline_path(suite_name)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in payload.get("metrics", {}).items()}


def write_baseline(result: RunResult) -> Path:
    """Freeze the current numbers as the bar future runs must clear.

    Records the commit so a surprising baseline can be traced back to the code
    that produced it -- a baseline with no provenance is just a number someone
    typed.
    """
    path = baseline_path(result.suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "suite": result.suite,
                "metrics": result.metrics,
                "n_examples": result.n_examples,
                "recorded_at": datetime.now(UTC).isoformat(),
                "git_sha": _git_sha(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def evaluate_gates(
    metrics: dict[str, float], baseline: dict[str, float], gates: list[Gate]
) -> list[GateResult]:
    results: list[GateResult] = []
    for gate in gates:
        value = metrics.get(gate.metric)
        prior = baseline.get(gate.metric)

        if value is None:
            results.append(
                GateResult(gate, None, prior, False, f"metric {gate.metric!r} not reported")
            )
            continue

        if gate.min_value is not None and value < gate.min_value:
            results.append(
                GateResult(
                    gate, value, prior, False, f"below floor {gate.min_value:g}"
                )
            )
            continue

        if gate.max_regression is not None and prior is not None:
            drop = prior - value
            if drop > gate.max_regression:
                results.append(
                    GateResult(
                        gate,
                        value,
                        prior,
                        False,
                        f"regressed {drop:.4f} (allowed {gate.max_regression:g})",
                    )
                )
                continue

        note = "ok" if prior is not None else "ok (no baseline yet)"
        results.append(GateResult(gate, value, prior, True, note))
    return results


def run_suite(suite: Suite) -> RunResult:
    started = time.perf_counter()
    baseline = load_baseline(suite.name)

    try:
        examples = load_jsonl(suite.dataset)
        metrics = {k: float(v) for k, v in suite.run(examples).items()}
    except Exception as exc:  # a broken suite is a failing suite, not a crash
        log.error("eval.suite.failed", suite=suite.name, error=str(exc))
        return RunResult(
            suite=suite.name,
            project=suite.project,
            metrics={},
            baseline=baseline,
            n_examples=0,
            seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    result = RunResult(
        suite=suite.name,
        project=suite.project,
        metrics=metrics,
        baseline=baseline,
        n_examples=len(examples),
        seconds=time.perf_counter() - started,
    )
    result.gates = evaluate_gates(metrics, baseline, suite.gates)
    log.info(
        "eval.suite.done",
        suite=suite.name,
        n=result.n_examples,
        passed=result.passed,
        **{k: round(v, 4) for k, v in metrics.items()},
    )
    return result


def run_suites(
    projects: list[str] | None = None, only: list[str] | None = None
) -> list[RunResult]:
    suites = list_suites(projects)
    if only:
        wanted = set(only)
        suites = [s for s in suites if s.name in wanted]
    return [run_suite(suite) for suite in suites]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
