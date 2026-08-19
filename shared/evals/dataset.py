from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from shared.config import REPO_ROOT


@dataclass(frozen=True)
class Example:
    """One labelled item.

    `inputs` is whatever the system under test needs; `expected` is whatever
    the metric needs. Both stay loose dicts so a suite can evolve its label
    schema without a migration -- but `id` and `tags` are fixed, because
    slicing a report by tag is how you find which *kind* of question regressed.
    """

    id: str
    inputs: dict
    expected: dict
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict) -> Example:
        missing = {"id", "inputs", "expected"} - raw.keys()
        if missing:
            raise ValueError(f"example missing required keys: {sorted(missing)}")
        return cls(
            id=str(raw["id"]),
            inputs=raw["inputs"],
            expected=raw["expected"],
            tags=list(raw.get("tags", [])),
        )


def load_jsonl(path: str | Path) -> list[Example]:
    """Read a golden set. One JSON object per line, blank lines and # ignored."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"golden set not found: {resolved}")

    examples: list[Example] = []
    seen: set[str] = set()
    for lineno, line in enumerate(_lines(resolved), start=1):
        try:
            example = Example.from_dict(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{resolved}:{lineno}: {exc}") from exc
        if example.id in seen:
            raise ValueError(f"{resolved}:{lineno}: duplicate example id {example.id!r}")
        seen.add(example.id)
        examples.append(example)

    if not examples:
        raise ValueError(f"golden set is empty: {resolved}")
    return examples


def filter_by_tag(examples: list[Example], tag: str) -> list[Example]:
    return [e for e in examples if tag in e.tags]


def _lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped
