"""Tables as data, never as flattened prose.

The single decision this project is built to defend: a figure is answered by
resolving it to a *cell*, not by asking a language model to read digits out of
a rendered string. The cell carries its own scale and unit, so "1,842,600" in
a table stated in thousands becomes $1.8426bn without anyone guessing.

`ledgerline.numeric` measures the difference. `ledgerline.numeric_baseline`
keeps measuring the prose approach forever, so the ablation is a number that
re-runs in CI rather than a claim in a README.
"""

from ledgerline.tables.model import Cell, Table, TableStore
from ledgerline.tables.query import Answer, answer_numeric

__all__ = ["Answer", "Cell", "Table", "TableStore", "answer_numeric"]
