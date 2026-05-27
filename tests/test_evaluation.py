"""Tests for extraction evaluation metrics."""

from docpii.evaluation.metrics import score_extraction
from docpii.templates.nuextract import default_extraction_template


class TestScoreExtraction:
    """Metric scoring tests."""

    def test_exact_match_scores_one(self, sample_ground_truth: dict) -> None:
        """Identical prediction and ground truth should score 1.0."""
        template = default_extraction_template()
        result = score_extraction(sample_ground_truth, sample_ground_truth, template)
        assert result.average_score == 1.0

    def test_missing_entity_lowers_score(self, sample_ground_truth: dict) -> None:
        """Missing entities should reduce the average score."""
        template = default_extraction_template()
        predicted = {"document_id": "DOC-1", "entities": []}
        result = score_extraction(predicted, sample_ground_truth, template)
        assert result.average_score < 1.0

    def test_verbatim_string_partial_match(self) -> None:
        """Verbatim strings use indel similarity, not exact match only."""
        template = {"name": "verbatim-string"}
        result = score_extraction({"name": "Jane Doe"}, {"name": "Jane Do"}, template)
        assert 0.0 < result.average_score < 1.0
