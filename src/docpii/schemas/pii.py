"""PII/PCI/PHI extraction schemas."""

from typing import Literal

from pydantic import BaseModel, Field


EntityType = Literal[
    "person",
    "organization",
    "location",
    "contact",
    "government_id",
    "financial",
    "health",
    "other",
]


class ContactInfo(BaseModel):
    """Contact details associated with an entity."""

    email: str | None = Field(default=None, description="email")
    phone: str | None = Field(default=None, description="phone number")
    fax: str | None = Field(default=None, description="fax number")
    website: str | None = Field(default=None, description="website URL")


class Address(BaseModel):
    """Postal or physical address."""

    street: str | None = Field(default=None, description="verbatim-string")
    city: str | None = Field(default=None, description="verbatim-string")
    state_or_province: str | None = Field(default=None, description="verbatim-string")
    postal_code: str | None = Field(default=None, description="verbatim-string")
    country: str | None = Field(default=None, description="country")
    full_address: str | None = Field(
        default=None,
        description="verbatim-string; full address as printed when not split across fields",
    )


class GovernmentId(BaseModel):
    """Government-issued identifiers (PII)."""

    id_type: Literal[
        "ssn",
        "national_id",
        "passport",
        "drivers_license",
        "tax_id",
        "other",
    ] = Field(description="type of government identifier")
    id_number: str = Field(description="verbatim-string")
    issuing_country: str | None = Field(default=None, description="country")


class FinancialAccount(BaseModel):
    """Payment or banking identifiers (PCI / financial PII)."""

    account_type: Literal[
        "credit_card",
        "debit_card",
        "bank_account",
        "iban",
        "routing_number",
        "other",
    ] = Field(description="financial account type")
    account_number: str = Field(description="verbatim-string")
    institution_name: str | None = Field(default=None, description="verbatim-string")
    cardholder_name: str | None = Field(default=None, description="verbatim-string")
    expiry_date: str | None = Field(default=None, description="date")
    cvv: str | None = Field(default=None, description="verbatim-string")


class HealthIdentifier(BaseModel):
    """Health-related identifiers (PHI)."""

    id_type: Literal[
        "medical_record_number",
        "insurance_member_id",
        "insurance_policy_number",
        "npi",
        "other",
    ] = Field(description="health identifier type")
    id_number: str = Field(description="verbatim-string")
    provider_name: str | None = Field(default=None, description="verbatim-string")


class Entity(BaseModel):
    """A discovered entity and its associated sensitive attributes."""

    entity_type: EntityType = Field(description="coarse entity category")
    name: str | None = Field(
        default=None,
        description="verbatim-string; primary label (person name, org name, etc.)",
    )
    role_or_context: str | None = Field(
        default=None,
        description="string; role in document (e.g. patient, guarantor, cardholder)",
    )
    contact: ContactInfo | None = None
    address: Address | None = None
    government_ids: list[GovernmentId] = Field(default_factory=list)
    financial_accounts: list[FinancialAccount] = Field(default_factory=list)
    health_identifiers: list[HealthIdentifier] = Field(default_factory=list)
    other_sensitive_fields: list[str] = Field(
        default_factory=list,
        description="verbatim-string; catch-all for sensitive values not covered above",
    )


class DocumentExtraction(BaseModel):
    """Top-level extraction result for a single document."""

    document_id: str | None = Field(
        default=None,
        description="verbatim-string; external document identifier if present",
    )
    entities: list[Entity] = Field(default_factory=list)
