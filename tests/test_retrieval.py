import pytest

from ledgerline.retrieval.bm25 import BM25Index, reciprocal_rank_fusion, tokenize
from ledgerline.retrieval.chunking import chunk_text, split_sections

FILING = """Item 1. Business

We manufacture industrial systems and aerospace components.

Item 1A. Risk Factors

Sustained increases in the cost of raw materials have adversely affected our
results of operations. We may be unable to pass these costs on to customers.

Item 7. MD&A

Net revenue for fiscal 2025 was $1,842.6 million, an increase of 12.4 percent.
"""


class TestChunking:
    def test_splits_on_item_headers(self):
        sections = split_sections(FILING)
        labels = [label for label, _, _ in sections if label]
        assert labels == ["Item 1. Business", "Item 1A. Risk Factors", "Item 7. MD&A"]

    def test_no_headers_yields_one_span(self):
        assert split_sections("just some prose") == [(None, 0, len("just some prose"))]

    def test_offsets_round_trip_to_the_source(self):
        # The whole point of tracking offsets: a citation has to resolve back
        # to the exact characters in the original document.
        for chunk in chunk_text(FILING, target_chars=120, overlap_chars=20):
            assert FILING[chunk.char_start : chunk.char_end] == chunk.content

    def test_chunks_carry_their_section(self):
        chunks = chunk_text(FILING, target_chars=120, overlap_chars=20)
        risk = [c for c in chunks if "raw materials" in c.content]
        assert risk and all(c.section == "Item 1A. Risk Factors" for c in risk)

    def test_ordinals_are_dense_and_ordered(self):
        chunks = chunk_text(FILING, target_chars=100, overlap_chars=10)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_short_text_is_a_single_chunk(self):
        chunks = chunk_text("One short sentence.", target_chars=1200)
        assert len(chunks) == 1
        assert chunks[0].content == "One short sentence."

    def test_rejects_overlap_larger_than_target(self):
        with pytest.raises(ValueError, match="overlap_chars"):
            chunk_text(FILING, target_chars=100, overlap_chars=100)

    def test_terminates_on_a_single_huge_sentence(self):
        # No sentence boundary inside the window -- must hard-cut rather than
        # loop forever.
        chunks = chunk_text("x" * 5000, target_chars=200, overlap_chars=50)
        assert len(chunks) > 1


class TestBM25:
    @pytest.fixture
    def index(self):
        return BM25Index.build(
            [
                ("d1", "Net revenue for fiscal 2025 was 1,842.6 million dollars"),
                ("d2", "Gross margin declined 180 basis points to 34.2 percent"),
                ("d3", "We employed approximately 6,480 people worldwide"),
                ("d4", "The revolving credit facility had 75.0 million drawn"),
            ]
        )

    def test_ranks_the_obvious_document_first(self, index):
        assert index.rank("what was net revenue", k=1) == ["d1"]
        assert index.rank("how many people worldwide", k=1) == ["d3"]

    def test_vocabulary_mismatch_returns_nothing(self, index):
        """The baseline's defining weakness, pinned deliberately.

        The corpus says "employed"; the analyst asks about "employees". No
        lexical matcher bridges that -- not this one, and not Postgres's
        english stemmer either, which produces 'employe' and 'employ'. This is
        precisely the gap the dense half of hybrid retrieval closes, and the
        nDCG delta from closing it is the first real number in the README.
        """
        assert index.rank("how many employees") == []

    def test_returns_nothing_for_out_of_vocabulary_query(self, index):
        assert index.rank("cryptocurrency holdings") == []

    def test_empty_index_is_safe(self):
        assert BM25Index.build([]).rank("anything") == []

    def test_stopwords_do_not_drive_ranking(self, index):
        # A query that is only stopwords carries no signal.
        assert index.rank("the and of") == []

    def test_scores_are_descending(self, index):
        scores = [s for _, s in index.search("million revenue margin", k=4)]
        assert scores == sorted(scores, reverse=True)

    def test_tokenizer_keeps_numbers(self):
        assert "1,842.6".replace(",", "") or True  # sanity
        assert "34.2" in tokenize("margin was 34.2 percent")


class TestRRF:
    def test_fuses_on_rank_not_score(self):
        # b is mid-ranked in both lists; a and c are first in one and absent
        # from the other. Consistent presence should win.
        fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], limit=3)
        assert fused[0] == "b"

    def test_preserves_a_single_ranking(self):
        assert reciprocal_rank_fusion([["a", "b", "c"]], limit=2) == ["a", "b"]

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
