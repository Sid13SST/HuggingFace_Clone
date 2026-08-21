import pytest

from ledgerline.tables.model import Table, TableStore
from ledgerline.tables.query import (
    Answer,
    Declined,
    answer_numeric,
    requested_year,
    score_row,
    tokenize,
)

INCOME = {
    "id": "t-income",
    "caption": "Consolidated Statements of Operations",
    "scale_hint": 1000,
    "unit": "USD",
    "columns": ["Fiscal 2025", "Fiscal 2024"],
    "rows": [
        {"label": "Net revenue", "values": ["1,842,600", "1,639,300"]},
        {"label": "Gross profit", "values": ["630,200", "590,100"]},
        {"label": "Gross margin", "unit": "percent", "values": ["34.2", "36.0"]},
        {"label": "Employees worldwide", "unit": "count", "values": ["6,480", "6,210"]},
        {"label": "Restructuring charge", "values": ["—", "12,400"]},
    ],
}


@pytest.fixture
def table():
    return Table.from_dict(INCOME)


@pytest.fixture
def store(table):
    return TableStore(tables=[table])


class TestTableModel:
    def test_applies_the_table_scale(self, table):
        assert table.cell("Net revenue", "Fiscal 2025").value == pytest.approx(1.8426e9)

    def test_percent_rows_opt_out_of_scaling(self, table):
        """The failure this guards against: a 34.2% margin becoming 34,200."""
        assert table.cell("Gross margin", "Fiscal 2025").value == pytest.approx(34.2)

    def test_count_rows_opt_out_of_scaling(self, table):
        assert table.cell("Employees worldwide", "Fiscal 2025").value == pytest.approx(6480)

    def test_prior_period_column(self, table):
        assert table.cell("Net revenue", "Fiscal 2024").value == pytest.approx(1.6393e9)

    def test_em_dash_is_empty_not_zero(self, table):
        """Coercing an unparseable cell to 0.0 produces quietly wrong answers."""
        cell = table.cell("Restructuring charge", "Fiscal 2025")
        assert cell.value is None
        assert not cell.is_numeric

    def test_unknown_row_or_column_is_none(self, table):
        assert table.cell("Nonexistent", "Fiscal 2025") is None
        assert table.cell("Net revenue", "Fiscal 2019") is None

    def test_row_length_mismatch_is_rejected(self):
        bad = {**INCOME, "rows": [{"label": "Net revenue", "values": ["1"]}]}
        with pytest.raises(ValueError, match="1 values for 2 columns"):
            Table.from_dict(bad)

    def test_internal_consistency_of_the_fixture(self, table):
        """Gross profit over net revenue must reproduce the stated margin.

        A cross-check an agent can run for free, and the reason the fixture was
        built with consistent numbers rather than plausible-looking ones.
        """
        revenue = table.cell("Net revenue", "Fiscal 2025").value
        profit = table.cell("Gross profit", "Fiscal 2025").value
        margin = table.cell("Gross margin", "Fiscal 2025").value
        assert profit / revenue * 100 == pytest.approx(margin, abs=0.05)


class TestQueryHelpers:
    def test_tokenize_drops_interrogatives(self):
        assert tokenize("What was the net revenue in fiscal 2025?") == [
            "net", "revenue", "fiscal", "2025",
        ]

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("net revenue in fiscal 2025", "2025"),
            ("revenue for FY2024", "2024"),
            ("revenue in 2023", "2023"),
            ("what was net revenue", None),
        ],
    )
    def test_requested_year(self, question, expected):
        assert requested_year(question) == expected

    def test_score_row_prefers_the_more_specific_label(self):
        tokens = set(tokenize("What was Industrial Systems segment operating income?"))
        assert score_row(tokens, "Industrial Systems operating income") > score_row(
            tokens, "Industrial Systems revenue"
        )


class TestAnswerNumeric:
    def test_resolves_a_figure_to_a_cell_with_a_citation(self, store):
        result = answer_numeric("What was net revenue in fiscal 2025?", store)
        assert isinstance(result, Answer)
        assert result.value == pytest.approx(1.8426e9)
        assert result.column == "Fiscal 2025"
        assert result.citation() == "t-income['Net revenue', 'Fiscal 2025']"

    def test_named_prior_period_selects_the_right_column(self, store):
        result = answer_numeric("What was net revenue in fiscal 2024?", store)
        assert isinstance(result, Answer)
        assert result.value == pytest.approx(1.6393e9)

    def test_unnamed_period_defaults_to_the_current_column(self, store):
        result = answer_numeric("What was gross margin?", store)
        assert isinstance(result, Answer)
        assert result.column == "Fiscal 2025"

    def test_refuses_a_period_the_table_does_not_have(self, store):
        """The most dangerous failure in filing analysis, refused explicitly.

        Answering a fiscal-2023 question from the fiscal-2024 column is worse
        than not answering: it is wrong in a way that looks completely right.
        """
        result = answer_numeric("What was net revenue in fiscal 2023?", store)
        assert isinstance(result, Declined)
        assert "period" in result.reason
        assert result.best_row == "Net revenue"

    def test_refuses_a_metric_that_is_not_in_any_table(self, store):
        result = answer_numeric("What was free cash flow in fiscal 2025?", store)
        assert isinstance(result, Declined)

    def test_refuses_when_the_question_is_about_something_else(self, store):
        """A perfect row match is not a guarantee the question is about it.

        "What share of net revenue came from the largest customer?" covers the
        row "Net revenue" completely, and the answer still is not that cell.
        Unmatched content words are what catch it.
        """
        result = answer_numeric(
            "What share of net revenue came from the single largest customer?", store
        )
        assert isinstance(result, Declined)

    def test_refuses_an_empty_cell_rather_than_reporting_zero(self, store):
        result = answer_numeric("What was the restructuring charge in fiscal 2025?", store)
        assert isinstance(result, Declined)
        assert "empty" in result.reason

    def test_empty_store_declines(self):
        result = answer_numeric("What was net revenue?", TableStore(tables=[]))
        assert isinstance(result, Declined)


class TestCommittedFixture:
    """The shipped tables must stay consistent with the shipped prose corpus."""

    @pytest.fixture
    def store(self):
        from ledgerline.evals import table_store

        return table_store()

    def test_loads(self, store):
        assert len(store.tables) == 4

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("What was net revenue in fiscal 2025?", 1.8426e9),
            ("What was net revenue in fiscal 2024?", 1.6393e9),
            ("What was gross margin in fiscal 2025?", 34.2),
            ("How much was drawn on the revolving credit facility?", 7.5e7),
            ("How much of the revolving facility remained available?", 3.154e8),
            ("What was Industrial Systems segment operating income?", 1.713e8),
            ("What was Aerospace Components segment revenue?", 7.379e8),
            ("What was the effective tax rate in fiscal 2025?", 22.6),
            ("What were fiscal 2025 capital expenditures?", 1.482e8),
            ("How many employees are in the United States?", 4100),
            ("What accrual was recorded for the class action?", 1.4e7),
        ],
    )
    def test_golden_questions_resolve(self, store, question, expected):
        result = answer_numeric(question, store)
        assert isinstance(result, Answer), getattr(result, "reason", result)
        assert result.value == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize(
        "question",
        [
            "What was free cash flow in fiscal 2025?",
            "What was Industrial Systems segment revenue in fiscal 2023?",
            "What was the dividend per share declared in fiscal 2025?",
        ],
    )
    def test_undisclosed_questions_are_refused(self, store, question):
        assert isinstance(answer_numeric(question, store), Declined)

    def test_segments_sum_to_consolidated_revenue(self, store):
        industrial = answer_numeric("Industrial Systems revenue in fiscal 2025", store)
        aerospace = answer_numeric("Aerospace Components revenue in fiscal 2025", store)
        total = answer_numeric("What was net revenue in fiscal 2025?", store)
        assert industrial.value + aerospace.value == pytest.approx(total.value)


class TestRowScoring:
    """A row label and a question have to account for each other.

    Label coverage alone made brevity beat specificity: any label the question
    happens to contain scores a perfect 1.000, and nothing longer can catch it.
    """

    def test_a_bare_label_does_not_beat_the_one_that_was_asked_for(self):
        """The failure this replaced.

        Three tables in Caterpillar's 10-K carry a row called simply
        "Construction Industries" -- assets, capital expenditures,
        depreciation. Asked for the segment's sales, the old scorer gave the
        bare label 1.000 and the row actually holding the figure 0.750, and
        answered with the segment's assets.
        """
        question = set(tokenize("What were Construction Industries total sales in 2025?"))
        bare = score_row(question, "Construction Industries")
        specific = score_row(question, "Construction Industries -- Total Sales and Revenues")
        assert specific > bare

    def test_specificity_only_wins_when_it_was_asked_for(self):
        """The other direction, and why this is not simply 'prefer longer'.

        A question that says nothing about geography must not be pulled into
        the North America column just because that label is longer.
        """
        question = set(tokenize("What were Construction Industries total sales in 2025?"))
        asked = score_row(question, "Construction Industries -- Total Sales and Revenues")
        unasked = score_row(question, "Construction Industries -- North America")
        assert asked > unasked

    def test_an_unrelated_label_scores_nothing(self):
        question = set(tokenize("What was operating profit in 2025?"))
        assert score_row(question, "Raw materials") == 0.0

    def test_the_segment_sales_question_resolves_to_the_segment_sales(self):
        """End to end, on the committed real filing: the figure the README
        called unreachable, reached, and not confused with the assets row that
        used to win."""
        from ledgerline.evals.real import table_store

        answer = answer_numeric(
            "What were Construction Industries total sales in 2025?", table_store()
        )
        assert isinstance(answer, Answer), getattr(answer, "reason", answer)
        assert answer.value == pytest.approx(25_060_000_000)
