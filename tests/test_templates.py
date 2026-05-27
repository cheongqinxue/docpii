"""Tests for NuExtract template conversion."""

import pytest

from docpii.schemas.pii import DocumentExtraction
from docpii.templates.nuextract import (
    default_extraction_template,
    pydantic_to_nuextract_template,
    template_to_json,
)


class TestNuExtractTemplate:
    """Template conversion tests."""

    def test_default_template_has_entities(self) -> None:
        """Default template should include an entities array."""
        template = default_extraction_template()
        assert "entities" in template
        assert isinstance(template["entities"], list)

    def test_template_serializes_to_json(self) -> None:
        """Template JSON serialization should be valid JSON."""
        template = pydantic_to_nuextract_template(DocumentExtraction)
        text = template_to_json(template)
        assert '"entities"' in text

    def test_unsupported_schema_raises(self) -> None:
        """Schemas with unsupported branches should raise ValueError."""
        from pydantic import BaseModel

        class BadModel(BaseModel):
            value: dict[str, str]

        with pytest.raises(ValueError, match="unsupported"):
            pydantic_to_nuextract_template(BadModel)
