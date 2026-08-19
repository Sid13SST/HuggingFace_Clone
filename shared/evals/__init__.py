"""The eval harness. Built before the agents, on purpose.

Both projects register suites here. A suite turns a labelled dataset plus a
system-under-test into a `RunResult`, which the runner compares against a
stored baseline and either passes or fails per declared gates.
"""

from shared.evals.dataset import Example, load_jsonl
from shared.evals.registry import Suite, get_suite, list_suites, register_suite
from shared.evals.runner import RunResult, run_suites

__all__ = [
    "Example",
    "RunResult",
    "Suite",
    "get_suite",
    "list_suites",
    "load_jsonl",
    "register_suite",
    "run_suites",
]
