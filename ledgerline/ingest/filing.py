"""Parsing a real SEC filing into narrative text and structured tables.

Everything measured in this repo so far ran against a seventeen-chunk synthetic
fixture. This is the module that replaces it, and the first thing real filings
teach you is that the fixture was polite.

A Caterpillar 10-K is 6 MB of inline XBRL containing 150 `<table>` elements, of
which a quarter are layout rather than data. The file opens with
`<?xml encoding='ASCII'?>` while containing UTF-8 throughout, so a parser that
believes the declaration turns every em-dash into a replacement character --
and in a financial table an em-dash means "nil", not decoration.

Two alignment traps, both of which produced numbers that parsed cleanly and
were attributed to the wrong year:

  * Cells span. A header row of 26 `<td>`s and a body row of 50 describe the
    same width once `colspan` is applied. Padding flat cell lists aligns
    nothing.
  * Even on a correct grid, position is not meaning. The currency symbol lives
    in its own cell, so a row opening a block renders `$ | 63,980` while the
    row beneath renders `3,609` in the cell where the `$` was. One grid column
    holds a year, a dollar sign and a number depending on which row you read.

The governing decision is **decline rather than mangle**. A table this module
cannot read correctly is skipped with a reason and counted. A wrong number that
arrives with a citation attached is the worst output this system can produce,
and a silent extraction bug produces exactly that.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from ledgerline.tables.model import Cell, Table, _safe_parse
from shared.logging import get_logger

log = get_logger(__name__)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
#: Both orderings a filing uses to state its scale. Caterpillar writes
#: "(Dollars in millions)" above the statements and "(Millions of dollars)"
#: above the MD&A and segment tables -- 9 of its 28 readable tables use the
#: second form and only 1 uses the first. Matching only "in <scale>" left those
#: nine relying on a scale inherited from elsewhere in the section, which is
#: correct here by luck and off by a factor of a million anywhere it is not.
_SCALE_RE = re.compile(
    r"\(?\s*(?:dollars|amounts|shares)?\s*in\s+(thousands|millions|billions)"
    r"|(thousands|millions|billions)\s+of\s+(?:dollars|shares|units)",
    re.I,
)


_SCALES = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}


def _scale_of(match: re.Match) -> float:
    """The multiplier a `_SCALE_RE` match names, whichever branch matched."""
    return _SCALES[(match.group(1) or match.group(2)).lower()]

#: Row labels whose magnitude is absolute and must not take the table's scale.
#: Getting this wrong turns a 34.2% margin into 34,200,000.
_PERCENT_HINT = re.compile(r"%|percent|\brate\b|\bmargin\b", re.I)
_PER_SHARE_HINT = re.compile(r"per\s+(common\s+)?share|per\s+diluted", re.I)
_COUNT_HINT = re.compile(r"employees|headcount|number of|shares outstanding", re.I)


@dataclass(frozen=True)
class SkippedTable:
    """A table this module refused to read, and why.

    Kept rather than discarded because the skip rate is a headline number: it
    is the fraction of the filing this system cannot see, and a parser change
    that quietly raises it is a regression even when every extracted table
    stays correct.
    """

    index: int
    reason: str
    preview: str


@dataclass
class ParsedFiling:
    text: str
    tables: list[Table] = field(default_factory=list)
    skipped: list[SkippedTable] = field(default_factory=list)

    @property
    def extraction_rate(self) -> float:
        total = len(self.tables) + len(self.skipped)
        return len(self.tables) / total if total else 0.0

    def skip_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for skip in self.skipped:
            counts[skip.reason] = counts.get(skip.reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _norm(text: str) -> str:
    return re.sub(r"[\s ]+", " ", text).strip()


def _document(raw: bytes):
    """Parse the filing, ignoring its own encoding declaration.

    EDGAR filings routinely declare `encoding='ASCII'` and then contain UTF-8.
    Trusting the declaration replaces every em-dash with U+FFFD, and an em-dash
    in a financial table means the value is nil.
    """
    import lxml.html

    return lxml.html.fromstring(raw, parser=lxml.html.HTMLParser(encoding="utf-8"))


def _grid(table) -> list[list[str]]:
    """Rows expanded into a real grid, honouring colspan and rowspan."""
    occupied: dict[tuple[int, int], str] = {}
    for r, tr in enumerate(table.xpath(".//tr")):
        col = 0
        for cell in tr.xpath("./td|./th"):
            while (r, col) in occupied:
                col += 1
            text = _norm(" ".join(cell.itertext()))
            try:
                colspan = max(1, int(cell.get("colspan", 1)))
                rowspan = max(1, int(cell.get("rowspan", 1)))
            except ValueError:
                colspan = rowspan = 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    # Only the top-left position carries the text; the spanned
                    # positions are placeholders so later cells step past them.
                    occupied[(r + dr, col + dc)] = text if (dr, dc) == (0, 0) else ""
            col += colspan

    if not occupied:
        return []
    height = max(r for r, _ in occupied) + 1
    width = max(c for _, c in occupied) + 1
    return [[occupied.get((r, c), "") for c in range(width)] for r in range(height)]


def _row_values(cells: list[str]) -> list[str]:
    """The values a row actually states, in reading order.

    Grid position cannot be trusted for alignment (see the module docstring),
    but financial tables are regular in a different way: every body row states
    exactly as many values as there are columns, in order. So values are
    collected in sequence and matched to the header positionally, and a row
    yielding the wrong count is discarded rather than guessed at.

    Parentheses and currency symbols fold back onto the number they belong to
    instead of being dropped -- `(` `1,234` `)` is negative 1,234, and
    discarding the brackets as ornament flips the sign of every expense line in
    the filing.
    """
    values: list[str] = []
    prefix = ""
    for raw in cells:
        cell = raw.strip()
        if not cell:
            continue
        if cell in {"$", "€", "£"}:
            continue
        if cell in {"(", "$("}:
            prefix = "("
            continue
        if cell in {")", ")%", "%"}:
            if values:
                values[-1] += cell
            continue
        values.append(prefix + cell)
        prefix = ""
    return values


def _header_row(rows: list[list[str]]) -> tuple[int, list[str]] | None:
    """(index, column names) for the row naming the columns, or None.

    The last year-like row wins, because a stacked header puts the grouping
    band above the years and the years are what address a column.
    """
    found: tuple[int, list[str]] | None = None
    for i, row in enumerate(rows[:10]):
        years = [v.strip() for v in _row_values(row) if _YEAR_RE.fullmatch(v.strip())]
        if len(years) >= 2 and len(set(years)) == len(years):
            found = (i, years)
    return found


def _is_stacked(rows: list[list[str]], header: int) -> bool:
    """Does a second header band sit above this one?

    A stacked header means one visual column is addressed by a pair -- segment
    *and* year -- and this module's flat (row_label, column) model cannot
    express that. Reading only the years would merge three segments into one
    set of columns and attribute numbers to the wrong entity.
    """
    for row in rows[max(0, header - 2) : header]:
        labels = [
            v
            for v in _row_values(row)
            if not _YEAR_RE.search(v) and not _SCALE_RE.search(v) and len(v) > 2
        ]
        if len(labels) >= 2:
            return True
    return False


def _scale_and_unit(
    rows: list[list[str]], caption: str, inherited: float | None = None
) -> tuple[float, str, bool]:
    """(scale, unit, inherited?) for a table.

    A filing states its scale once per section, not once per table: "(Dollars
    in millions except per share data)" sits above the statement and every
    table under it inherits. A table read in isolation therefore looks
    unscaled, and its figures come out a million times too small -- `(168)`
    where the filing means minus 168 million. That is a confidently wrong
    number with a citation attached, which is the worst thing this system can
    emit.

    So an explicit note in or above the table wins, and failing that the most
    recent note seen earlier in the document is inherited. The third return
    value says which happened, because an inherited scale is an inference and
    the numeric analyst should be able to weigh it accordingly.
    """
    own = " ".join(" ".join(r) for r in rows)
    unit = "USD" if re.search(r"dollar|\$", own + caption, re.I) else "unknown"

    # Precedence is by source, not by string position. Putting the caption and
    # the table's own rows in one haystack meant a section note reading
    # "(Dollars in millions)" outranked the table's own "(in thousands)"
    # purely because it appeared earlier in the concatenation, and every figure
    # in that table came out a thousand times too large.
    for text in (own, caption):
        match = _SCALE_RE.search(text)
        if match:
            return _scale_of(match), unit, False
    if inherited is not None:
        return inherited, unit, True
    return 1.0, unit, False


def _row_unit(label: str) -> str | None:
    """Per-row override for rows the table's scale must not touch."""
    if _PER_SHARE_HINT.search(label):
        return "per_share"
    if _PERCENT_HINT.search(label):
        return "percent"
    if _COUNT_HINT.search(label):
        return "count"
    return None


def _parse_table(
    index: int, element, caption: str, document_id: str,
    inherited_scale: float | None = None,
) -> Table | str:
    """Build a Table, or return a string explaining why it was declined."""
    rows = _grid(element)
    if len(rows) < 3:
        return "too few rows"

    found = _header_row(rows)
    if found is None:
        return "no year header"
    header, columns = found
    if _is_stacked(rows, header):
        return "stacked header"

    # A filing titles a table *inside* the table, as a single-value row above
    # the header: "Sales and Revenues by Segment", then "(Millions of
    # dollars)", then the years. `_caption_for` looks above the element and
    # finds nothing, because EDGAR makes each table the first child of its own
    # wrapper div -- all 28 extracted tables came out with an empty caption.
    #
    # That emptiness was not cosmetic. Two segment tables carry the identical
    # row label "Construction Industries", one holding depreciation and the
    # other capital expenditures, and `answer_numeric` charges its
    # unmatched-word penalty against the caption. With no caption to charge,
    # "capital expenditures" and "depreciation" cost a question exactly the
    # same, the two tables tie, and the tie goes to whichever the document
    # happens to reach first -- so a question about capital expenditure was
    # answered, confidently and with a citation, out of the depreciation table.
    title = [v[0] for v in (_row_values(r) for r in rows[:header]) if len(v) == 1]
    if title:
        caption = " | ".join([*title, caption]) if caption else " | ".join(title)

    body: list[tuple[str, list[str]]] = []
    ragged = 0
    for row in rows[header + 1 :]:
        values = _row_values(row)
        if len(values) < 2:
            continue  # blank row, or a section heading such as "Revenues:"
        label, data = values[0], values[1:]
        if len(data) != len(columns):
            # Wrong arity means a value is missing or an extra token crept in,
            # and there is no way to tell which column lost it. Guessing here
            # is how a figure ends up under the wrong year.
            ragged += 1
            continue
        body.append((label, data))

    if not body:
        return "no aligned rows"
    if ragged > len(body):
        # More rows unreadable than readable: the shape assumption does not
        # hold here, so nothing from this table should be trusted.
        return "mostly ragged"
    if len({label for label, _ in body}) != len(body):
        return "duplicate row labels"

    scale, unit, inherited = _scale_and_unit(rows, caption, inherited_scale)
    if inherited:
        caption = f"{caption} [scale inherited from section]"
    table = Table(
        id=f"{document_id}-t{index}",
        caption=caption[:200],
        columns=columns,
        row_labels=[label for label, _ in body],
        row_units={
            label: override for label, _ in body if (override := _row_unit(label))
        },
        scale_hint=scale,
        unit=unit,
        document_id=document_id,
    )
    for r, (label, values) in enumerate(body):
        effective = table.effective_scale(label)
        for c, raw in enumerate(values):
            table.cells[(r, c)] = Cell(
                row=r, col=c, raw=raw, value=_safe_parse(raw, effective)
            )
    return table


def _caption_for(element) -> str:
    """Nearest preceding text that reads like a title.

    Financial tables in iXBRL rarely use `<caption>`; the title sits in a
    paragraph above them, and the scale note ("Dollars in millions") often
    lives there rather than inside the table.

    On EDGAR's own generator this finds nothing -- each table is the first
    child of its wrapper `<div>` and has no preceding sibling at all. Climbing
    to the wrapper's siblings was tried and reaches page furniture (a page
    number, then "Table of Contents"), which is worse than an empty caption
    because `answer_numeric` treats caption words as things the table explains.
    `_parse_table` reads the title out of the table's own rows instead.

    Kept for the shape it does handle: a table that really is a sibling of its
    heading, which is what every non-EDGAR source produces.
    """
    parts: list[str] = []
    node = element.getprevious()
    while node is not None and len(parts) < 3:
        text = _norm(" ".join(node.itertext()))
        if text:
            parts.append(text)
        node = node.getprevious()
    return " | ".join(reversed(parts))


def _narrative_text(doc) -> str:
    """Document text with tables removed.

    Tables are dropped rather than flattened because flattening is the failure
    this project exists to argue against: a table rendered into prose loses its
    scale and its column headers, and the numbers left behind read as
    authoritative while meaning nothing.
    """
    for table in doc.xpath("//table"):
        table.getparent().remove(table)
    for tag in doc.xpath("//script|//style"):
        tag.getparent().remove(tag)
    lines = (_norm(line) for line in doc.itertext())
    return "\n".join(line for line in lines if line)


def parse_html(raw: bytes, document_id: str = "filing") -> ParsedFiling:
    """Split a filing into narrative text and the tables that parsed cleanly."""
    doc = _document(raw)
    tables: list[Table] = []
    skipped: list[SkippedTable] = []

    # One pass in document order, carrying the most recent scale note forward.
    # Reading a filing is sequential and so is this: the note above Statement 1
    # governs every table beneath it until another note supersedes it.
    running_scale: float | None = None
    index = -1
    for element in doc.iter():
        if element.tag != "table":
            for text in (element.text, element.tail):
                match = _SCALE_RE.search(text) if text else None
                if match:
                    running_scale = _scale_of(match)
            continue

        index += 1
        outcome = _parse_table(
            index, element, _caption_for(element), document_id, running_scale
        )
        if isinstance(outcome, str):
            preview = _norm(" ".join(element.itertext()))[:80]
            skipped.append(SkippedTable(index=index, reason=outcome, preview=preview))
        else:
            tables.append(outcome)

    text = _narrative_text(_document(raw))
    log.info(
        "filing.parsed",
        document_id=document_id,
        tables=len(tables),
        skipped=len(skipped),
        chars=len(text),
    )
    return ParsedFiling(text=text, tables=tables, skipped=skipped)


def iter_numeric_cells(table: Table) -> Iterator[tuple[str, str, float]]:
    """(row label, column, value) for every cell that parsed to a number."""
    for r, label in enumerate(table.row_labels):
        for c, column in enumerate(table.columns):
            cell = table.cells.get((r, c))
            if cell is not None and cell.value is not None:
                yield label, column, cell.value
