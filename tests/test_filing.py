"""Filing HTML parsing.

Every fixture here is a reduction of something a real Caterpillar 10-K does.
They are inline rather than a committed 6 MB filing because a test that needs a
6 MB fixture is a test people stop running, and because a reduction states
which pathology it is about -- `test_currency_symbol_does_not_shift_values`
names the bug it was written for, where a slice of real filing would only say
"this used to be wrong".
"""

from __future__ import annotations

import pytest

lxml = pytest.importorskip("lxml.html", reason="lxml not installed")

from ledgerline.ingest.filing import (  # noqa: E402
    _grid,
    _header_row,
    _row_values,
    parse_html,
)


def table_html(body: str, prelude: str = "") -> bytes:
    return f"<html><body>{prelude}<table>{body}</table></body></html>".encode()


class TestGrid:
    def test_colspan_expands_into_real_columns(self):
        """A header row of 3 cells and a body row of 5 describe the same width
        once colspan is applied. Padding flat cell lists aligns nothing, which
        is how the 2025 figure ended up under the 2024 heading."""
        doc = lxml.fromstring(
            table_html(
                "<tr><td>Label</td><td colspan='2'>2025</td><td colspan='2'>2024</td></tr>"
                "<tr><td>Revenue</td><td>a</td><td>b</td><td>c</td><td>d</td></tr>"
            )
        )
        grid = _grid(doc.xpath("//table")[0])
        assert [len(r) for r in grid] == [5, 5]
        assert grid[0] == ["Label", "2025", "", "2024", ""]

    def test_rowspan_pushes_later_rows_right(self):
        doc = lxml.fromstring(
            table_html(
                "<tr><td rowspan='2'>Both</td><td>x</td></tr><tr><td>y</td></tr>"
            )
        )
        grid = _grid(doc.xpath("//table")[0])
        assert grid[0] == ["Both", "x"]
        assert grid[1] == ["", "y"]

    def test_empty_table_is_not_a_crash(self):
        doc = lxml.fromstring(table_html(""))
        assert _grid(doc.xpath("//table")[0]) == []


class TestRowValues:
    def test_currency_symbol_does_not_shift_values(self):
        """The alignment bug that survived the colspan fix.

        A row opening a block renders `$ | 63,980`; the row beneath renders
        `3,609` in the cell where the `$` was. Same grid column, different
        meaning -- so values are matched in sequence, not by position.
        """
        with_symbol = _row_values(["Sales", "", "$", "63,980", "", "$", "61,363"])
        without = _row_values(["Revenues", "", "3,609", "", "", "3,446", ""])
        assert with_symbol == ["Sales", "63,980", "61,363"]
        assert without == ["Revenues", "3,609", "3,446"]
        assert len(with_symbol) == len(without)

    def test_parentheses_survive_as_a_negative(self):
        """Dropping brackets as ornament flips the sign of every expense line
        in the filing."""
        assert _row_values(["Loss", "(", "168", ")"]) == ["Loss", "(168)"]

    def test_percent_sign_rejoins_its_number(self):
        assert _row_values(["Rate", "3.90", "%"]) == ["Rate", "3.90%"]

    def test_dollar_open_paren_is_a_negative_too(self):
        assert _row_values(["Loss", "$(", "42", ")"]) == ["Loss", "(42)"]

    def test_em_dash_is_kept_as_a_stated_nil(self):
        """A nil holds its column. Dropping it would shift every value after it
        one place left and silently reattribute them."""
        assert _row_values(["Row", "1", "—", "3"]) == ["Row", "1", "—", "3"]


class TestHeaderRow:
    def test_finds_the_year_band(self):
        rows = [["", "2025", "2024"], ["Revenue", "1", "2"]]
        assert _header_row(rows) == (0, ["2025", "2024"])

    def test_prefers_the_lower_band_when_headers_stack(self):
        rows = [["", "Consolidated", "Consolidated"], ["", "2025", "2024"]]
        assert _header_row(rows) == (1, ["2025", "2024"])

    def test_no_years_means_no_header(self):
        assert _header_row([["Item", "Amount"], ["Cash", "5"]]) is None

    def test_repeated_years_are_not_a_header(self):
        """A stacked table repeats 2025 under each segment band. Treating that
        as three distinct columns would merge segments."""
        assert _header_row([["", "2025", "2025", "2024"]]) is None


class TestParseHtml:
    def test_reads_a_clean_statement(self):
        raw = table_html(
            "<tr><td>(Dollars in millions)</td><td>2025</td><td>2024</td></tr>"
            "<tr><td>Total revenues</td><td>$</td><td>67,589</td><td>$</td><td>64,809</td></tr>"
            "<tr><td>Operating profit</td><td>11,151</td><td>13,072</td></tr>"
        )
        parsed = parse_html(raw, "doc")
        assert len(parsed.tables) == 1
        table = parsed.tables[0]
        assert table.columns == ["2025", "2024"]
        assert table.scale_hint == 1e6
        assert table.cell("Total revenues", "2025").value == 67_589_000_000

    def test_percent_rows_opt_out_of_the_table_scale(self):
        """Applying "in millions" to a 34.2% margin gives 34,200,000 -- wrong
        in the rows a reader is least likely to spot-check."""
        raw = table_html(
            "<tr><td>(Dollars in millions)</td><td>2025</td><td>2024</td></tr>"
            "<tr><td>Revenue</td><td>100</td><td>90</td></tr>"
            "<tr><td>Effective tax rate</td><td>24.0</td><td>22.5</td></tr>"
        )
        table = parse_html(raw, "doc").tables[0]
        assert table.cell("Revenue", "2025").value == 100_000_000
        assert table.cell("Effective tax rate", "2025").value == 24.0

    def test_per_share_rows_opt_out_too(self):
        raw = table_html(
            "<tr><td>(in millions except per share)</td><td>2025</td><td>2024</td></tr>"
            "<tr><td>Profit per common share</td><td>18.90</td><td>21.90</td></tr>"
            "<tr><td>Total revenues</td><td>100</td><td>90</td></tr>"
        )
        table = parse_html(raw, "doc").tables[0]
        assert table.cell("Profit per common share", "2025").value == 18.90

    def test_scale_is_inherited_from_the_section_above(self):
        """A filing states its scale once per section, not once per table. A
        table read in isolation comes out a million times too small."""
        # The note is four paragraphs up, out of caption range, which is the
        # real case: one note governs a run of tables further down the section.
        raw = (
            b"<html><body><p>(Dollars in millions)</p>"
            b"<p>Filler one.</p><p>Filler two.</p><p>Filler three.</p>"
            b"<p>Filler four.</p>"
            b"<table><tr><td>x</td><td>2025</td><td>2024</td></tr>"
            b"<tr><td>Revenue</td><td>100</td><td>90</td></tr>"
            b"<tr><td>Profit</td><td>10</td><td>9</td></tr></table>"
            b"</body></html>"
        )
        table = parse_html(raw, "doc").tables[0]
        assert table.scale_hint == 1e6
        assert "inherited" in table.caption

    def test_an_explicit_note_beats_an_inherited_one(self):
        raw = (
            b"<html><body><p>(Dollars in millions)</p>"
            b"<table><tr><td>(in thousands)</td><td>2025</td><td>2024</td></tr>"
            b"<tr><td>Revenue</td><td>100</td><td>90</td></tr>"
            b"<tr><td>Profit</td><td>10</td><td>9</td></tr></table>"
            b"</body></html>"
        )
        assert parse_html(raw, "doc").tables[0].scale_hint == 1e3

    def test_stacked_headers_are_declined_not_guessed(self):
        """One visual column addressed by segment *and* year cannot be
        expressed by a flat (row, column) model. Reading only the years merges
        three segments and attributes numbers to the wrong entity."""
        raw = table_html(
            "<tr><td></td><td colspan='2'>Consolidated</td><td colspan='2'>Financial</td></tr>"
            "<tr><td>x</td><td>2025</td><td>2024</td><td>2025b</td><td>2024b</td></tr>"
            "<tr><td>Revenue</td><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
        )
        parsed = parse_html(raw, "doc")
        assert not parsed.tables
        assert parsed.skipped[0].reason == "stacked header"

    def test_ragged_rows_are_dropped_rather_than_padded(self):
        """A row with the wrong number of values has lost one somewhere, and
        nothing says which column. Guessing puts a figure under the wrong
        year."""
        raw = table_html(
            "<tr><td>(in millions)</td><td>2025</td><td>2024</td></tr>"
            "<tr><td>Good</td><td>1</td><td>2</td></tr>"
            "<tr><td>Bad</td><td>3</td></tr>"
        )
        table = parse_html(raw, "doc").tables[0]
        assert table.row_labels == ["Good"]

    def test_section_headings_are_not_rows(self):
        raw = table_html(
            "<tr><td>(in millions)</td><td>2025</td><td>2024</td></tr>"
            "<tr><td>Sales and revenues:</td><td></td><td></td></tr>"
            "<tr><td>Total</td><td>1</td><td>2</td></tr>"
        )
        assert parse_html(raw, "doc").tables[0].row_labels == ["Total"]

    def test_utf8_survives_a_lying_encoding_declaration(self):
        """EDGAR filings declare ASCII and contain UTF-8. Believing the
        declaration turns every em-dash into U+FFFD, and an em-dash in a
        financial table means nil."""
        raw = (
            "<?xml version='1.0' encoding='ASCII'?><html><body><table>"
            "<tr><td>(in millions)</td><td>2025</td><td>2024</td></tr>"
            "<tr><td>Revenue</td><td>1</td><td>—</td></tr>"
            "<tr><td>Profit</td><td>2</td><td>3</td></tr>"
            "</table></body></html>"
        ).encode()
        table = parse_html(raw, "doc").tables[0]
        assert "�" not in table.cell("Revenue", "2024").raw
        # An em-dash is a stated nil, which is not zero and not a number.
        assert table.cell("Revenue", "2024").value is None

    def test_narrative_text_excludes_tables(self):
        """Flattening a table into prose loses its scale and column headers,
        and the numbers left behind read as authoritative while meaning
        nothing. That is the failure this project argues against."""
        raw = (
            b"<html><body><p>Gross margin declined.</p>"
            b"<table><tr><td>x</td><td>2025</td><td>2024</td></tr>"
            b"<tr><td>Revenue</td><td>99999</td><td>88888</td></tr></table>"
            b"</body></html>"
        )
        parsed = parse_html(raw, "doc")
        assert "Gross margin declined." in parsed.text
        assert "99999" not in parsed.text

    def test_skip_reasons_are_counted(self):
        raw = table_html("<tr><td>a</td></tr>")
        parsed = parse_html(raw, "doc")
        assert parsed.skip_reasons() == {"too few rows": 1}
        assert parsed.extraction_rate == 0.0

    def test_a_document_with_no_tables_is_fine(self):
        parsed = parse_html(b"<html><body><p>Words.</p></body></html>", "doc")
        assert parsed.tables == [] and parsed.extraction_rate == 0.0

    def test_the_title_row_becomes_the_caption(self):
        """EDGAR puts a table's title in the table, as a single-value row above
        the header, and makes the table the first child of its own wrapper --
        so looking above the element finds nothing at all."""
        raw = table_html(
            "<tr><td>Reconciliation of Capital expenditures:</td></tr>"
            "<tr><td>(Millions of dollars)</td><td>2025</td><td>2024</td></tr>"
            "<tr><td>Construction Industries</td><td>358</td><td>323</td></tr>"
            "<tr><td>Resource Industries</td><td>353</td><td>228</td></tr>"
        )
        table = parse_html(raw, "doc").tables[0]
        assert "Capital expenditures" in table.caption
        assert table.cell("Construction Industries", "2025").value == 358_000_000

    def test_a_caption_is_what_separates_two_identical_row_labels(self):
        """The failure this was written for.

        Two segment tables in the same filing both have a row called
        "Construction Industries" -- one depreciation, one capital expenditure.
        `answer_numeric` charges its unmatched-word penalty against the
        caption, so with both captions empty the tables tie and the tie goes to
        document order: a capex question was answered out of the depreciation
        table, confidently, with a citation attached.
        """
        from ledgerline.tables import Answer, TableStore, answer_numeric

        def segment_table(title: str, value: str) -> str:
            return (
                f"<tr><td>{title}</td></tr>"
                "<tr><td>(Millions of dollars)</td><td>2025</td><td>2024</td></tr>"
                f"<tr><td>Construction Industries</td><td>{value}</td><td>1</td></tr>"
                "<tr><td>Resource Industries</td><td>2</td><td>3</td></tr>"
            )

        # Depreciation first, so document order favours the wrong answer.
        raw = (
            "<html><body>"
            f"<table>{segment_table('Reconciliation of Depreciation:', '266')}</table>"
            f"<table>{segment_table('Reconciliation of Capital expenditures:', '358')}</table>"
            "</body></html>"
        ).encode()
        store = TableStore(tables=parse_html(raw, "doc").tables)

        answer = answer_numeric(
            "What were Construction Industries capital expenditures in 2025?", store
        )
        assert isinstance(answer, Answer)
        assert answer.value == 358_000_000


class TestUnitInference:
    """What a row measures, and whether the table's scale applies to it.

    These are two questions, and the parser used to answer the second by
    looking the first up in a set. That works until a filing writes a share
    count in millions -- genuinely a count, genuinely scaled -- at which point
    the only honest answer is to read what the filing actually says.
    """

    def test_a_percent_in_the_cell_beats_a_silent_label(self):
        """Nine of Caterpillar's eleven percentage rows have no hint in the
        label: "Commercial paper", "Weighted-average volatility". The cell says
        `3.8%` and that is first-hand evidence; the label is a description."""
        table = parse_html(
            table_html(
                "<tr><td>Weighted-average interest rates:</td></tr>"
                "<tr><td>(Millions of dollars)</td><td>2025</td><td>2024</td></tr>"
                "<tr><td>Commercial paper</td><td>3.8%</td><td>4.5%</td></tr>"
                "<tr><td>Notes payable to banks</td><td>10.1%</td><td>10.8%</td></tr>"
            ),
            "doc",
        ).tables[0]
        assert table.unit_for("Commercial paper") == "percent"
        assert table.cell("Commercial paper", "2025").value == 3.8

    def test_a_label_headed_by_an_amount_is_not_a_rate(self):
        """The regression this class exists for.

        "Amount that, if recognized, would impact the effective tax rate" ends
        in the word `rate` and holds $1,199 million. A hint matching anywhere
        in the label called it a percentage and stripped six orders of
        magnitude off it, silently, in a row no reader spot-checks.
        """
        table = parse_html(
            table_html(
                "<tr><td>Reconciliation of unrecognized tax benefits:</td></tr>"
                "<tr><td>(Millions of dollars)</td><td>2025</td><td>2024</td></tr>"
                "<tr><td>Amount that, if recognized, would impact the "
                "effective tax rate</td><td>1,199</td><td>1,137</td></tr>"
                "<tr><td>Additions for tax positions</td><td>96</td><td>82</td></tr>"
            ),
            "doc",
        ).tables[0]
        label = "Amount that, if recognized, would impact the effective tax rate"
        assert table.unit_for(label) == "USD"
        assert table.cell(label, "2025").value == 1_199_000_000

    def test_a_rate_that_really_is_a_rate_still_reads_as_one(self):
        """The other side of the same rule: nothing here is headed by an
        amount noun, so the label's hint stands even with no `%` in the cell."""
        table = parse_html(
            table_html(
                "<tr><td>Selected Operating Data</td></tr>"
                "<tr><td>(Dollars in thousands)</td><td>2025</td><td>2024</td></tr>"
                "<tr><td>Gross margin</td><td>34.2</td><td>36.0</td></tr>"
                "<tr><td>Effective tax rate</td><td>22.6</td><td>24.1</td></tr>"
            ),
            "doc",
        ).tables[0]
        assert table.cell("Gross margin", "2025").value == 34.2
        assert table.cell("Effective tax rate", "2025").value == 22.6

    def test_a_duration_does_not_take_a_dollar_scale(self):
        """`7 years` in a table of millions is seven years. It used to be seven
        million, then get refused for implausibility -- which is worse than a
        parse error, because a parse error is counted and this was not."""
        table = parse_html(
            table_html(
                "<tr><td>Grant Year</td></tr>"
                "<tr><td>(Millions of dollars)</td><td>2025</td><td>2024</td></tr>"
                "<tr><td>Weighted-average expected lives</td>"
                "<td>7 years</td><td>7 years</td></tr>"
                "<tr><td>Weighted-average volatility</td><td>30.5%</td><td>30.7%</td></tr>"
            ),
            "doc",
        ).tables[0]
        assert table.unit_for("Weighted-average expected lives") == "duration"
        assert table.cell("Weighted-average expected lives", "2025").value == 7

    def test_a_stated_row_scale_carries_to_its_siblings(self):
        """Caterpillar marks one share-count row `(in millions)` and leaves the
        two above it unmarked, because a reader can see they are the same kind
        of thing at the same magnitude. All three are millions."""
        table = parse_html(
            table_html(
                "<tr><td>Computations of profit per share:</td></tr>"
                "<tr><td>(Millions of dollars)</td><td>2025</td><td>2024</td></tr>"
                "<tr><td>Profit for the period</td><td>8,884</td><td>10,792</td></tr>"
                "<tr><td>Weighted average number of common shares outstanding</td>"
                "<td>470.0</td><td>486.7</td></tr>"
                "<tr><td>Shares outstanding as of December 31, (in millions)</td>"
                "<td>465.3</td><td>477.9</td></tr>"
            ),
            "doc",
        ).tables[0]
        unmarked = "Weighted average number of common shares outstanding"
        marked = "Shares outstanding as of December 31, (in millions)"
        assert table.cell(marked, "2025").value == 465_300_000
        assert table.cell(unmarked, "2025").value == 470_000_000
        # Still a count. What it measures and how it is written are separate
        # questions, and conflating them is what this change undid.
        assert table.unit_for(unmarked) == "count"

    def test_a_count_the_filing_did_not_scale_is_left_alone(self):
        """The guard on the rule above, and the reason it is opt-in.

        Nothing in a row label separates 6,480 employees from 470.0 million
        shares. So the default is the safe one -- leave counts unscaled -- and
        a scale carries only where the filing stated one. Without this, a
        headcount in a table of thousands becomes 6.48 million.
        """
        table = parse_html(
            table_html(
                "<tr><td>Selected Operating Data</td></tr>"
                "<tr><td>(Dollars in thousands)</td><td>2025</td><td>2024</td></tr>"
                "<tr><td>Total revenue</td><td>1,240</td><td>1,180</td></tr>"
                "<tr><td>Employees worldwide</td><td>6,480</td><td>6,210</td></tr>"
            ),
            "doc",
        ).tables[0]
        assert table.cell("Total revenue", "2025").value == 1_240_000
        assert table.cell("Employees worldwide", "2025").value == 6_480


class TestBandedTables:
    """Tables whose columns are entities and whose periods are row bands.

    Caterpillar reports segment results this way: a header naming seven
    geographies, then "2025" alone on a line, then a block of rows, then
    "2024". `_header_row` finds no row of years and declines the table
    outright, which is how $25.060 billion of Construction Industries revenue
    stayed unreachable in a filing that states it twice.
    """

    @staticmethod
    def banded(extra: str = "") -> bytes:
        return table_html(
            "<tr><td>Sales and Revenues by Geographic Region</td></tr>"
            "<tr><td>(Millions of dollars)</td><td>North America</td>"
            "<td>EAME</td><td>Total Sales and Revenues</td></tr>"
            "<tr><td>2025</td></tr>"
            "<tr><td>Construction Industries</td><td>14,064</td>"
            "<td>4,595</td><td>25,060</td></tr>"
            "<tr><td>Resource Industries</td><td>4,643</td>"
            "<td>2,061</td><td>12,474</td></tr>"
            "<tr><td>2024</td></tr>"
            "<tr><td>Construction Industries</td><td>14,576</td>"
            "<td>4,315</td><td>25,455</td></tr>"
            "<tr><td>Resource Industries</td><td>4,597</td>"
            "<td>1,809</td><td>12,471</td></tr>" + extra
        )

    def test_the_band_becomes_the_column(self):
        """The transposition. A banded table is already flat, just oriented
        the other way, and turning it the right way up gives the (metric,
        year) shape every other part of this system speaks."""
        table = parse_html(self.banded(), "doc").tables[0]
        assert table.columns == ["2025", "2024"]
        assert table.cell(
            "Construction Industries -- Total Sales and Revenues", "2025"
        ).value == 25_060_000_000
        assert table.cell(
            "Construction Industries -- Total Sales and Revenues", "2024"
        ).value == 25_455_000_000

    def test_every_column_of_every_band_is_addressable(self):
        """Two segments by three columns by two years is twelve figures, and
        none of them was reachable before."""
        table = parse_html(self.banded(), "doc").tables[0]
        assert len(table.row_labels) == 6
        assert table.cell("Resource Industries -- EAME", "2025").value == 2_061_000_000

    def test_a_row_missing_from_a_band_is_dropped(self):
        """Present in 2025 and absent from 2024. Carrying it would file a
        figure under a year the filing did not state it for."""
        raw = table_html(
            "<tr><td>Segment data</td></tr>"
            "<tr><td>(Millions of dollars)</td><td>North America</td>"
            "<td>Total Sales and Revenues</td></tr>"
            "<tr><td>2025</td></tr>"
            "<tr><td>Construction Industries</td><td>14,064</td><td>25,060</td></tr>"
            "<tr><td>Discontinued Line</td><td>12</td><td>19</td></tr>"
            "<tr><td>2024</td></tr>"
            "<tr><td>Construction Industries</td><td>14,576</td><td>25,455</td></tr>"
        )
        table = parse_html(raw, "doc").tables[0]
        assert any(r.startswith("Construction Industries") for r in table.row_labels)
        assert not any(r.startswith("Discontinued Line") for r in table.row_labels)

    def test_repeated_column_names_are_declined(self):
        """The MD&A prints the same table with a `% Chg` column beside every
        region. Seven columns all called `% Chg` cannot address anything, and
        this is the case the reader has to refuse rather than guess at."""
        raw = table_html(
            "<tr><td>Sales and Revenues by Geographic Region</td></tr>"
            "<tr><td>(Millions of dollars)</td><td>North America</td><td>% Chg</td>"
            "<td>EAME</td><td>% Chg</td></tr>"
            "<tr><td>2025</td></tr>"
            "<tr><td>Construction Industries</td><td>14,064</td><td>(4%)</td>"
            "<td>4,595</td><td>6%</td></tr>"
            "<tr><td>2024</td></tr>"
            "<tr><td>Construction Industries</td><td>14,576</td><td>(2%)</td>"
            "<td>4,315</td><td>3%</td></tr>"
        )
        parsed = parse_html(raw, "doc")
        assert not parsed.tables
        assert parsed.skipped[0].reason == "no year header"

    def test_one_band_is_not_a_dimension(self):
        """A single period is a list, not a table addressed by year. Two bands
        is where the period starts carrying information."""
        raw = table_html(
            "<tr><td>Segment data</td></tr>"
            "<tr><td>(Millions of dollars)</td><td>North America</td>"
            "<td>Total Sales and Revenues</td></tr>"
            "<tr><td>2025</td></tr>"
            "<tr><td>Construction Industries</td><td>14,064</td><td>25,060</td></tr>"
            "<tr><td>Resource Industries</td><td>4,643</td><td>12,474</td></tr>"
        )
        assert not parse_html(raw, "doc").tables

    def test_a_year_header_still_wins(self):
        """The banded reader runs only where the year path already declined,
        so it can add coverage and cannot change a figure already read."""
        table = parse_html(
            table_html(
                "<tr><td>(Millions of dollars)</td><td>2025</td><td>2024</td></tr>"
                "<tr><td>Total sales and revenues</td><td>67,589</td><td>64,809</td></tr>"
                "<tr><td>Operating profit</td><td>11,151</td><td>13,072</td></tr>"
            ),
            "doc",
        ).tables[0]
        assert table.columns == ["2025", "2024"]
        assert table.cell("Total sales and revenues", "2025").value == 67_589_000_000
