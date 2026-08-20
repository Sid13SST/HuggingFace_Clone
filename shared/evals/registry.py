from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from shared.evals.dataset import Example

MetricFn = Callable[[list[Example]], dict[str, float]]

_SUITES: dict[str, Suite] = {}

#: Projects that expose an `evals` module with a `register()` entrypoint.
KNOWN_PROJECTS = ("ledgerline", "sightline")


@dataclass(frozen=True)
class Gate:
    """A condition the suite must satisfy for CI to pass.

    Absolute bounds and a relative guard, and all three are needed.
    `min_value` is a floor for things that must never be bad (groundedness).
    `max_value` is a ceiling for the metrics where *lower* is better --
    calibration error, fabrication rate, cost per run. `max_regression` is the
    relative guard that catches the prompt edit quietly costing three points of
    nDCG, which no absolute bound loose enough to live with would ever notice.

    `higher_is_better` flips the direction of the relative guard only. Without
    it, `max_regression` on an error rate would fire when the error *improved*,
    which is the sort of gate that gets deleted rather than fixed.
    """

    metric: str
    min_value: float | None = None
    max_value: float | None = None
    max_regression: float | None = None
    higher_is_better: bool = True

    def describe(self) -> str:
        parts = []
        if self.min_value is not None:
            parts.append(f">= {self.min_value:g}")
        if self.max_value is not None:
            parts.append(f"<= {self.max_value:g}")
        if self.max_regression is not None:
            parts.append(f"regression <= {self.max_regression:g}")
        return f"{self.metric} {' and '.join(parts)}" if parts else self.metric

    def regression(self, value: float, prior: float) -> float:
        """How much worse `value` is than `prior`. Negative means improved."""
        return prior - value if self.higher_is_better else value - prior


@dataclass(frozen=True)
class Suite:
    name: str
    project: str
    dataset: Path
    run: MetricFn
    gates: list[Gate] = field(default_factory=list)
    description: str = ""


def register_suite(suite: Suite) -> Suite:
    if suite.name in _SUITES:
        raise ValueError(f"suite {suite.name!r} already registered")
    _SUITES[suite.name] = suite
    return suite


def get_suite(name: str) -> Suite:
    load_projects()
    if name not in _SUITES:
        raise KeyError(f"unknown suite {name!r}; known: {sorted(_SUITES)}")
    return _SUITES[name]


def list_suites(projects: list[str] | None = None) -> list[Suite]:
    load_projects(projects)
    wanted = set(projects or KNOWN_PROJECTS)
    return sorted(
        (s for s in _SUITES.values() if s.project in wanted), key=lambda s: s.name
    )


def load_projects(projects: list[str] | None = None) -> None:
    """Import project eval modules so their registrations happen.

    A project whose optional dependencies are not installed is skipped rather
    than fatal: running only the Ledgerline suites should not require the
    computer-vision extras.
    """
    for project in projects or KNOWN_PROJECTS:
        try:
            importlib.import_module(f"{project}.evals")
        except ModuleNotFoundError:
            continue
