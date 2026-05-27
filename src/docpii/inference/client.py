"""OpenAI-compatible client for NuExtract3 served via vLLM."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from docpii.config import Settings
from docpii.schemas.pii import DocumentExtraction
from docpii.templates.nuextract import (
    DEFAULT_INSTRUCTIONS,
    default_extraction_template,
    template_to_json,
)

logger = logging.getLogger(__name__)

_JSON_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_model_json(raw: str) -> dict[str, Any]:
    """Parse JSON from a model response, stripping optional markdown fences.

    Args:
        raw: Raw model output string.

    Returns:
        Parsed JSON object.

    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
    """
    cleaned = _JSON_FENCE_PATTERN.sub("", raw.strip())
    return json.loads(cleaned)


class NuExtractClient:
    """Client for structured extraction with NuExtract3."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the client.

        Args:
            settings: Optional settings; defaults are loaded from the environment.
        """
        self.settings = settings or Settings()
        self._client = OpenAI(
            api_key=self.settings.vllm_api_key,
            base_url=self.settings.vllm_base_url,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def extract_from_images(
        self,
        image_data_urls: list[str],
        *,
        template: dict[str, Any] | None = None,
        instructions: str | None = DEFAULT_INSTRUCTIONS,
        enable_thinking: bool | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Run structured extraction on one or more document page images.

        Args:
            image_data_urls: Ordered page images as data URLs.
            template: NuExtract JSON template; defaults to PII schema.
            instructions: Optional extraction instructions for the model.
            enable_thinking: Override reasoning mode from settings.
            temperature: Override sampling temperature from settings.

        Returns:
            Parsed extraction JSON as a dictionary.
        """
        template = template or default_extraction_template()
        response = self._client.chat.completions.create(
            model=self.settings.vllm_model,
            temperature=temperature if temperature is not None else self.settings.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}} for url in image_data_urls
                    ],
                }
            ],
            extra_body={
                "chat_template_kwargs": {
                    "template": template_to_json(template),
                    "instructions": instructions,
                    "enable_thinking": (
                        enable_thinking
                        if enable_thinking is not None
                        else self.settings.enable_thinking
                    ),
                }
            },
        )
        content = response.choices[0].message.content or ""
        return parse_model_json(content)

    def extract_document(
        self,
        document_path: str,
        *,
        template: dict[str, Any] | None = None,
        instructions: str | None = DEFAULT_INSTRUCTIONS,
    ) -> DocumentExtraction:
        """Extract structured PII from a document file path.

        Args:
            document_path: Path to PDF or image file.
            template: Optional NuExtract template override.
            instructions: Optional extraction instructions.

        Returns:
            Validated ``DocumentExtraction`` result.

        Raises:
            ValidationError: If model output does not match the schema.
        """
        from docpii.inference.documents import load_document_images

        images = load_document_images(document_path)
        raw = self.extract_from_images(
            images,
            template=template,
            instructions=instructions,
        )
        try:
            return DocumentExtraction.model_validate(raw)
        except ValidationError:
            logger.exception("Model output failed schema validation: %s", raw)
            raise
