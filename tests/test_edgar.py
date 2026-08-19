from datetime import date

import pytest

from ledgerline.ingest.edgar import normalize_cik, parse_recent_filings

# Shape of the real https://data.sec.gov/submissions/CIK##########.json payload:
# `recent` is column-oriented, so every field is a parallel array.
SUBMISSIONS = {
    "cik": "18230",
    "name": "Example Industrial Corp",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000018230-26-000012",
                "0000018230-26-000009",
                "0000018230-25-000044",
                "0000018230-25-000041",
            ],
            "form": ["10-K", "8-K", "10-Q", "4"],
            "filingDate": ["2026-02-12", "2026-01-28", "2025-10-30", "2025-10-15"],
            "reportDate": ["2025-12-31", "2026-01-28", "2025-09-30", ""],
            "primaryDocument": [
                "exi-20251231.htm",
                "exi-20260128.htm",
                "exi-20250930.htm",
                "xslF345X05/primary_doc.xml",
            ],
            "primaryDocDescription": ["10-K", "8-K", "10-Q", "FORM 4"],
            "items": ["", "2.02,7.01", "", ""],
        }
    },
}


class TestNormalizeCik:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("18230", "0000018230"),
            (18230, "0000018230"),
            ("0000018230", "0000018230"),
            ("CIK0000018230", "0000018230"),
        ],
    )
    def test_pads_to_ten_digits(self, raw, expected):
        assert normalize_cik(raw) == expected

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="not a CIK"):
            normalize_cik("AAPL")


class TestParseRecentFilings:
    def test_transposes_the_column_oriented_payload(self):
        filings = parse_recent_filings(SUBMISSIONS)
        assert [f.form for f in filings] == ["10-K", "8-K", "10-Q"]
        assert filings[0].filed_at == date(2026, 2, 12)
        assert filings[0].period_end == date(2025, 12, 31)

    def test_filters_to_requested_forms(self):
        filings = parse_recent_filings(SUBMISSIONS, forms=("10-K",))
        assert len(filings) == 1
        assert filings[0].accession == "0000018230-26-000012"

    def test_form_4_is_excluded_by_default(self):
        assert all(f.form != "4" for f in parse_recent_filings(SUBMISSIONS))

    def test_parses_8k_items(self):
        eight_k = next(f for f in parse_recent_filings(SUBMISSIONS) if f.form == "8-K")
        assert eight_k.items == ["2.02", "7.01"]

    def test_blank_report_date_becomes_none(self):
        filings = parse_recent_filings(SUBMISSIONS, forms=("4",))
        assert filings[0].period_end is None

    def test_respects_limit(self):
        assert len(parse_recent_filings(SUBMISSIONS, limit=2)) == 2

    def test_builds_the_archive_url(self):
        filing = parse_recent_filings(SUBMISSIONS, forms=("10-K",))[0]
        # Path uses the un-padded CIK and the accession with dashes stripped.
        assert filing.url == (
            "https://www.sec.gov/Archives/edgar/data/18230/"
            "000001823026000012/exi-20251231.htm"
        )

    def test_missing_filings_block_yields_nothing(self):
        assert parse_recent_filings({"cik": "18230"}) == []
