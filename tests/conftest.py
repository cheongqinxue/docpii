"""Shared pytest fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_ground_truth() -> dict:
    """Minimal ground-truth extraction for metric tests."""
    return {
        "document_id": "DOC-1",
        "entities": [
            {
                "entity_type": "person",
                "name": "Jane Doe",
                "role_or_context": "patient",
                "contact": {"email": "jane@example.com", "phone": "555-0100", "fax": None, "website": None},
                "address": None,
                "government_ids": [],
                "financial_accounts": [],
                "health_identifiers": [],
                "other_sensitive_fields": [],
            }
        ],
    }


@pytest.fixture
def project_root() -> Path:
    """Repository root directory."""
    return Path(__file__).resolve().parents[1]
