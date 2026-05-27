"""Convert Pydantic schemas to NuExtract JSON templates."""

from __future__ import annotations

import json
from typing import Any

from numind.nuextract_utils import convert_json_schema_to_nuextract_template
from pydantic import BaseModel

from docpii.schemas.pii import DocumentExtraction

DEFAULT_INSTRUCTIONS = (
    "Extract every person, organization, or other entity mentioned in the document. "
    "For each entity, collect all associated private or sensitive information including "
    "names, government IDs, contact details, addresses, credit card numbers, bank "
    "account numbers, and health identifiers. Use verbatim-string fields exactly as "
    "printed. Return null or empty lists when a field is not present."
)


def pydantic_to_nuextract_template(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model JSON schema to a NuExtract template.

    Args:
        model: Pydantic model class describing the desired extraction output.

    Returns:
        NuExtract-compatible template dictionary.

    Raises:
        ValueError: If the schema cannot be fully converted (branches dropped).
    """
    template, dropped = convert_json_schema_to_nuextract_template(model.model_json_schema())
    if dropped:
        raise ValueError(f"Unsupported schema branches were dropped: {dropped}")
    return template


def default_extraction_template() -> dict[str, Any]:
    """Return the default PII extraction template for NuExtract3.

    Returns:
        Template derived from ``DocumentExtraction``.
    """
    return pydantic_to_nuextract_template(DocumentExtraction)


def template_to_json(template: dict[str, Any], *, indent: int | None = 4) -> str:
    """Serialize a NuExtract template to JSON.

    Args:
        template: NuExtract template dictionary.
        indent: JSON indentation; ``None`` for compact output.

    Returns:
        JSON string suitable for ``chat_template_kwargs.template``.
    """
    return json.dumps(template, indent=indent, ensure_ascii=False)
