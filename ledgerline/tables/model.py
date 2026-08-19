from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from shared.evals.metrics import UnparseableNumber, parse_number

# Units whose magnitude is absolute, so the table's "in thousands" header must
# NOT be applied to them. Getting this wrong turns a 34.2% margin into 34,200
# and a 6,480 headcount into 6.48 million -- silently, and only in the rows a
# reader is least likely to spot-check.
UNSCALED_UNITS = frozenset({"percent", "count", "ratio", "per_share"})


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    raw: str
    value: float | None

    @property
    def is_numeric(self) -> bool:
        return self.value is not None


@dataclass
class Table:
    id: str
    caption: str
    columns: list[str]
    row_labels: list[str]
    #: Per-row unit override; absent means the table default.
    row_units: dict[str, str] = field(default_factory=dict)
    scale_hint: float = 1.0
    unit: str = "USD"
    document_id: str | None = None
    page: int | None = None
    cells: dict[tuple[int, int], Cell] = field(default_factory=dict)

    def effective_scale(self, row_label: str) -> float:
        return 1.0 if self.row_units.get(row_label) in UNSCALED_UNITS else self.scale_hint

    def unit_for(self, row_label: str) -> str:
        return self.row_units.get(row_label, self.unit)

    def cell(self, row_label: str, column: str) -> Cell | None:
        try:
            r = self.row_labels.index(row_label)
            c = self.columns.index(column)
        except ValueError:
            return None
        return self.cells.get((r, c))

    @classmethod
    def from_dict(cls, raw: dict) -> Table:
        table = cls(
            id=raw["id"],
            caption=raw.get("caption", ""),
            columns=list(raw["columns"]),
            row_labels=[r["label"] for r in raw["rows"]],
            row_units={
                r["label"]: r["unit"] for r in raw["rows"] if r.get("unit")
            },
            scale_hint=float(raw.get("scale_hint", 1)),
            unit=raw.get("unit", "USD"),
            document_id=raw.get("document_id"),
            page=raw.get("page"),
        )
        for r, row in enumerate(raw["rows"]):
            values = row["values"]
            if len(values) != len(table.columns):
                raise ValueError(
                    f"table {table.id!r} row {row['label']!r} has {len(values)} values "
                    f"for {len(table.columns)} columns"
                )
            scale = table.effective_scale(row["label"])
            for c, cell_raw in enumerate(values):
                table.cells[(r, c)] = Cell(
                    row=r, col=c, raw=cell_raw, value=_safe_parse(cell_raw, scale)
                )
        return table


def _safe_parse(raw: str, scale: float) -> float | None:
    """A cell that will not parse is empty, not zero.

    Financial tables are full of em-dashes, "n/a", and footnote markers.
    Coercing those to 0.0 produces answers that are confidently, quietly wrong.
    """
    if raw is None or not str(raw).strip() or str(raw).strip() in {"-", "--", "—", "n/a", "N/A"}:
        return None
    try:
        return parse_number(str(raw), scale_hint=scale)
    except UnparseableNumber:
        return None


@dataclass
class TableStore:
    tables: list[Table] = field(default_factory=list)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> TableStore:
        resolved = Path(path)
        tables: list[Table] = []
        with resolved.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    tables.append(Table.from_dict(json.loads(stripped)))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    raise ValueError(f"{resolved}:{lineno}: {exc}") from exc
        return cls(tables=tables)

    def periods(self) -> set[str]:
        return {column for table in self.tables for column in table.columns}
