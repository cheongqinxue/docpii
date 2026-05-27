"""Tests for PII schema models."""

from docpii.schemas.pii import DocumentExtraction, Entity


class TestDocumentExtraction:
    """DocumentExtraction model tests."""

    def test_empty_entities(self) -> None:
        """An empty entity list should validate."""
        doc = DocumentExtraction(document_id="X", entities=[])
        assert doc.document_id == "X"
        assert doc.entities == []

    def test_entity_with_financial_account(self) -> None:
        """Financial account nested models should validate."""
        entity = Entity.model_validate(
            {
                "entity_type": "person",
                "name": "Jane Doe",
                "financial_accounts": [
                    {
                        "account_type": "credit_card",
                        "account_number": "4111111111111111",
                    }
                ],
            }
        )
        assert entity.financial_accounts[0].account_type == "credit_card"
