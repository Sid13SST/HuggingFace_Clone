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
