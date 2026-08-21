"""The committed Caterpillar corpus, and the anchors labels resolve through.

These tests do not check that retrieval is *good* on this corpus -- that is
what the suites are for. They check that the corpus is loadable, that its ids
are unique, and that a label pointing at text the corpus no longer contains
fails loudly instead of scoring a smaller golden set than the one on disk.
"""

from __future__ import annotations

import pytest

from ledgerline.evals.real import (
    CORPUS_PATH,
    DOCUMENT_ID,
    MAX_ANCHOR_MATCHES,
    NUMERIC_PATH,
    RETRIEVAL_PATH,
    TABLES_PATH,
    AnchorNotFound,
    corpus_by_id,
    load_corpus,
    read_jsonl,
    relevant_ids,
    resolve_anchor,
)


class TestCorpus:
    def test_it_is_the_whole_filing_not_a_sample(self):
        """484 chunks is the point. A twenty-chunk excerpt of a real filing has
        the same problem as the synthetic fixture: no boilerplate, no
        near-duplicates, and nothing for a retriever to get wrong."""
        assert len(load_corpus()) > 400

    def test_ids_are_unique(self):
        records = load_corpus()
        assert len(corpus_by_id()) == len(records)

    def test_every_chunk_carries_text(self):
        assert all(r["text"].strip() for r in load_corpus())

    def test_ids_name_the_document_they_came_from(self):
        assert all(r["id"].startswith(DOCUMENT_ID) for r in load_corpus())

    def test_tables_are_not_flattened_into_the_narrative(self):
        """The corpus is prose only; figures live in cat_tables.jsonl.

        If a statement's numbers leak into the narrative text they arrive
        stripped of scale and column headers, and a retriever that finds them
        looks right while the figure it surfaces means nothing.
        """
        joined = " ".join(r["text"] for r in load_corpus())
        assert "67,589" not in joined

    def test_utf8_survived_the_export(self):
        """EDGAR declares ASCII and ships UTF-8. A replacement character here
        means the parser believed the declaration."""
        assert "�" not in CORPUS_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def store():
    from ledgerline.tables import TableStore

    return TableStore.from_jsonl(TABLES_PATH)


class TestTables:
    def test_every_extracted_table_round_trips(self, store):
        assert len(store.tables) == 32

    def test_values_are_the_filings_raw_strings(self, store):
        """Committing parsed floats would make the parser a fixed point: it
        would be graded against its own output and could never be wrong."""
        income = next(t for t in store.tables if "Total sales and revenues" in t.row_labels)
        assert income.cell("Total sales and revenues", "2025").raw.strip() == "67,589"

    def test_the_income_statement_reconciles(self, store):
        """Arithmetic the filing itself guarantees. If this fails, alignment
        broke again -- a figure has landed under the wrong year."""
        income = next(t for t in store.tables if "Total sales and revenues" in t.row_labels)
        for column in ("2025", "2024", "2023"):
            machinery = income.cell("Sales of Machinery, Power & Energy", column).value
            financial = income.cell("Revenues of Financial Products", column).value
            total = income.cell("Total sales and revenues", column).value
            assert machinery + financial == pytest.approx(total)


class TestAnchors:
    def test_a_phrase_resolves_to_the_chunks_holding_it(self):
        anchor = "Total sales and revenues for 2025 were $67.589 billion"
        ids = resolve_anchor(anchor)
        assert ids and all(anchor in corpus_by_id()[i]["text"] for i in ids)

    def test_overlapping_windows_may_both_count(self):
        """The chunker slides its window, so a sentence near a boundary really
        is in two chunks. Retrieving either has served the reader, so both are
        relevant -- requiring uniqueness would make labelling a rule about the
        chunker's stride, which is what anchoring exists to avoid.
        """
        ids = resolve_anchor("Total sales and revenues")
        assert 1 < len(ids) <= MAX_ANCHOR_MATCHES

    def test_missing_text_raises_rather_than_returning_nothing(self):
        """The failure mode this exists to prevent: a re-chunk invalidates a
        label, the label is quietly dropped, and the suite reports a healthy
        number for a golden set smaller than the file it read."""
        with pytest.raises(AnchorNotFound, match="no chunk contains"):
            resolve_anchor("a phrase that is certainly not in this filing")

    def test_boilerplate_is_rejected_as_a_locator(self):
        """An anchor matching most of the filing names a topic, not a passage."""
        with pytest.raises(AnchorNotFound, match="locates a topic"):
            resolve_anchor("the")

    def test_relevant_ids_dedupes_across_anchors(self):
        """Two anchors in one label may land in the same chunk. Counting it
        twice would inflate recall for that example alone."""
        anchor = "Total sales and revenues for 2025 were $67.589 billion"
        ids = relevant_ids({"relevant": [anchor, anchor]})
        assert len(ids) == len(set(ids))


class TestGoldenSets:
    """The labels themselves, checked against the corpus they name."""

    def test_every_retrieval_anchor_still_resolves(self):
        """This is the test that fires when the corpus is re-exported.

        It is the whole reason anchors are stored instead of chunk ids: a
        chunking change turns into a failing test naming the broken label,
        rather than into a quietly different number.
        """
        for record in read_jsonl(RETRIEVAL_PATH):
            for anchor in record["expected"]["relevant"]:
                assert resolve_anchor(anchor), record["id"]

    def test_declines_say_which_kind_they_are(self):
        """"The filing does not say" and "our parser could not read it" are
        different failures with different fixes, and collapsing them would let
        a low extraction rate read as good judgement."""
        for record in read_jsonl(NUMERIC_PATH):
            if record["expected"].get("not_disclosed"):
                assert record["expected"]["reason"] in {"not-stated", "not-extracted"}

    @pytest.mark.parametrize(
        ("label_id", "restated_as"),
        [
            ("nc-001", "$67.589 billion"),
            ("nc-007", "$11.151 billion"),
            ("nc-013", "Profit per share was $18.81"),
        ],
    )
    def test_labels_agree_with_the_filing_s_own_prose(self, label_id, restated_as):
        """Cross-check the tables against the narrative, which is what a careful
        analyst does.

        A 10-K states its headline figures twice: once in the statements and
        again, in words, in the MD&A. The two are independent of each other and
        of this parser, so a label that matches the table but not the prose is
        a label to re-read. nc-013 is the useful case -- the table gives basic
        EPS of $18.90 and the MD&A quotes diluted, $18.81, and reading only one
        of them would make either number look like a contradiction.
        """
        narrative = " ".join(r["text"] for r in load_corpus())
        assert restated_as in narrative, f"{label_id}: prose does not restate it"

    def test_not_extracted_labels_are_honest(self, store):
        """Every `not-extracted` label claims no committed table holds the
        figure, and the claim is checked rather than trusted.

        This test has already earned its place. Widening the scale-note pattern
        made the segment asset reconciliation readable, which put a "Total
        assets" row into the store while nc-034 still asserted the figure was
        out of reach. The label was stale, not the parser, and it was re-cut as
        answerable -- a coverage gap should close by turning into a question the
        system answers, never by remaining a refusal it passes.
        """
        from ledgerline.tables.query import tokenize

        # Three hand-written probes used to stand here, and they were exactly
        # as good as the guesses behind them: the banded reader made nc-036
        # reachable and not one of the three mentioned it. The claim a
        # `not-extracted` label makes is checkable in general -- no committed
        # row addresses this question -- so check that instead of three nouns.
        rows = [
            (table.id, label, set(tokenize(label)))
            for table in store.tables
            for label in table.row_labels
        ]
        for record in read_jsonl(NUMERIC_PATH):
            if record["expected"].get("reason") != "not-extracted":
                continue
            question = record["inputs"]["question"]
            wanted = {t for t in tokenize(question) if not t.isdigit()}
            for table_id, label, label_tokens in rows:
                assert not wanted <= label_tokens, (
                    f"{record['id']} claims the filing does not expose this, but "
                    f"{table_id} row {label!r} covers every word of it; re-cut it"
                )
