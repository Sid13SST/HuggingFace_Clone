"""Sentence-embedding similarity, and the dedupe path it feeds."""

from __future__ import annotations

import numpy as np
import pytest

from sightline.dedupe import (
    LEXICAL_CONFIG,
    DedupeConfig,
    ReportPoint,
    is_duplicate,
    pair_score,
    text_similarity,
)
from sightline.similarity import SEMANTIC_THRESHOLD, EmbeddingSimilarity, texts_in


class AxisEmbedder:
    """Two-axis fake: 'surface defect' and 'lighting'.

    Deliberately not random. The point of these tests is that semantically
    close texts sharing no tokens score high, and that only works if the fake
    encodes meaning rather than characters.
    """

    dim = 2
    model_name = "fake-axis"

    SURFACE = ("pothole", "road", "damage", "pavement", "broken", "sunken", "utility")
    LIGHT = ("streetlight", "lamp", "dark", "bulb", "light")

    def encode(self, texts):
        rows = []
        for text in texts:
            tokens = text.lower().split()
            vector = np.zeros(2, dtype=np.float32)
            vector[0] = sum(1.0 for t in tokens if any(s in t for s in self.SURFACE))
            vector[1] = sum(1.0 for t in tokens if any(s in t for s in self.LIGHT))
            if not vector.any():
                vector[:] = 1e-6
            rows.append(vector)
        return np.vstack(rows)


class TestEmbeddingSimilarity:
    @pytest.fixture
    def sim(self):
        return EmbeddingSimilarity(embedder=AxisEmbedder())

    def test_scores_meaning_where_jaccard_scores_zero(self, sim):
        """The whole reason this replaces the lexical baseline.

        "Road damage" and "Broken pavement here" are the same defect and share
        no tokens, so no amount of tuning a set-overlap score reaches them.
        """
        a, b = "Road damage", "Broken pavement here"
        assert text_similarity(a, b) == 0.0
        assert sim(a, b) > 0.9

    def test_separates_different_defect_kinds(self, sim):
        assert sim("pothole in the road", "streetlight is dark") < 0.2

    def test_empty_text_is_no_signal_not_an_error(self, sim):
        assert sim("", "pothole") == 0.0
        assert sim("   ", "  ") == 0.0

    def test_identical_text_scores_one(self, sim):
        assert sim("pothole", "pothole") == pytest.approx(1.0)

    def test_is_symmetric(self, sim):
        assert sim("road damage", "broken pavement") == sim("broken pavement", "road damage")

    def test_output_stays_in_range(self, sim):
        for a, b in [("pothole", "dark lamp"), ("road", "road"), ("", "x")]:
            assert 0.0 <= sim(a, b) <= 1.0

    def test_default_floor_matches_the_sql_mirror(self):
        """schema.sql scores text as `1 - (a <=> b)`, which is raw cosine.

        A floor here would make Python and Postgres compute different numbers
        from the same vectors -- the bug class the retrieval parity work
        existed to find.
        """
        assert EmbeddingSimilarity(embedder=AxisEmbedder()).floor == 0.0

    def test_a_floor_rescales_rather_than_clipping_everything(self):
        floored = EmbeddingSimilarity(embedder=AxisEmbedder(), floor=0.5)
        assert floored("pothole in the road", "streetlight is dark") == 0.0
        assert floored("pothole", "pothole") == pytest.approx(1.0)

    def test_vectors_are_cached_per_text(self, sim):
        sim("pothole", "pothole")
        assert "pothole" in sim._cache


class TestConfigCarriesItsThreshold:
    def test_similarity_and_threshold_travel_together(self):
        """A threshold is only meaningful against a similarity scale. Keeping
        them in separate places is how a semantic similarity ends up being run
        against a lexical threshold."""
        config = DedupeConfig(threshold=0.47, similarity_fn=EmbeddingSimilarity(AxisEmbedder()))
        assert config.similarity() is config.similarity_fn
        assert LEXICAL_CONFIG.similarity() is text_similarity

    def test_lexical_config_keeps_its_own_threshold(self):
        """The control must not drift onto the semantic operating point, or the
        ablation stops measuring anything."""
        assert LEXICAL_CONFIG.threshold == 0.62
        assert LEXICAL_CONFIG.threshold != SEMANTIC_THRESHOLD

    def test_explicit_similarity_argument_overrides_the_config(self):
        a = ReportPoint(id="a", lat=41.88, lon=-87.63, text="Road damage")
        b = ReportPoint(id="b", lat=41.88, lon=-87.63, text="Broken pavement here")
        lexical = pair_score(a, b, config=LEXICAL_CONFIG)
        semantic = pair_score(
            a, b, config=LEXICAL_CONFIG, similarity_fn=EmbeddingSimilarity(AxisEmbedder())
        )
        assert semantic > lexical


class TestSemanticDedupe:
    @pytest.fixture
    def config(self):
        return DedupeConfig(
            threshold=SEMANTIC_THRESHOLD, similarity_fn=EmbeddingSimilarity(AxisEmbedder())
        )

    def test_recovers_a_duplicate_the_lexical_path_misses(self, config):
        a = ReportPoint(id="a", lat=41.8800, lon=-87.6300, text="Road damage")
        b = ReportPoint(id="b", lat=41.8801, lon=-87.6300, text="Broken pavement here")
        assert not is_duplicate(a, b, config=LEXICAL_CONFIG)
        assert is_duplicate(a, b, config=config)

    def test_distance_still_gates_before_text(self, config):
        """Text similarity is a tiebreak, not a driver. Two identical reports a
        kilometre apart are two defects, and no similarity score should change
        that."""
        a = ReportPoint(id="a", lat=41.8800, lon=-87.6300, text="Pothole")
        b = ReportPoint(id="b", lat=41.8900, lon=-87.6300, text="Pothole")
        assert pair_score(a, b, config=config) == 0.0

    def test_category_mismatch_still_gates_before_text(self, config):
        a = ReportPoint(id="a", lat=41.88, lon=-87.63, text="damage", category="pothole")
        b = ReportPoint(id="b", lat=41.88, lon=-87.63, text="damage", category="streetlight")
        assert pair_score(a, b, config=config) == 0.0


class TestCommittedCache:
    def test_covers_every_text_the_suite_scores(self):
        """A stale cache must fail here, not halfway through a CI eval run."""
        from shared.config import REPO_ROOT
        from sightline.evals import dedupe_embedder

        dataset = REPO_ROOT / "sightline" / "evals" / "datasets" / "dedupe_pairs.jsonl"
        embedder = dedupe_embedder()
        missing = [t for t in texts_in(dataset) if t not in embedder]
        assert not missing, f"{len(missing)} texts need `sightline embed`"

    def test_the_ablation_actually_moves(self):
        """Pins the claim the README makes. If the embedding stops beating the
        lexical baseline on the hard slice, this is the first thing to say so.
        """
        from shared.evals.dataset import load_jsonl
        from sightline.evals import HERE, run_dedupe, run_dedupe_lexical

        examples = load_jsonl(HERE / "datasets" / "dedupe_pairs.jsonl")
        semantic, lexical = run_dedupe(examples), run_dedupe_lexical(examples)
        assert semantic["pair_recall"] > lexical["pair_recall"]
        assert semantic["accuracy.hard"] > lexical["accuracy.hard"]
        # Recall bought with false merges is not a win: a merged duplicate
        # silently removes a real defect from the queue.
        assert semantic["false_merge_rate"] <= lexical["false_merge_rate"]

    def test_the_threshold_sits_inside_the_measured_margin(self):
        """0.47 is the midpoint between the highest non-duplicate and the
        lowest duplicate, not an edge of the band. If a fixture edit narrows
        that margin past the threshold, this fails before the suite does."""
        from shared.evals.dataset import load_jsonl
        from sightline.evals import HERE, _to_point, semantic_dedupe_config

        config = semantic_dedupe_config()
        scored = [
            (
                pair_score(_to_point(e.inputs["a"]), _to_point(e.inputs["b"]), config=config),
                bool(e.expected["duplicate"]),
            )
            for e in load_jsonl(HERE / "datasets" / "dedupe_pairs.jsonl")
        ]
        highest_negative = max(s for s, dup in scored if not dup)
        lowest_positive = min(s for s, dup in scored if dup)
        assert highest_negative < SEMANTIC_THRESHOLD < lowest_positive
