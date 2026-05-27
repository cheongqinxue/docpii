"""Dataset I/O and message formatting for fine-tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docpii.templates.nuextract import DEFAULT_INSTRUCTIONS, template_to_json


@dataclass(frozen=True)
class TrainingExample:
    """One supervised fine-tuning example for NuExtract-style extraction."""

    example_id: str
    document_path: Path
    ground_truth: dict[str, Any]
    instructions: str = DEFAULT_INSTRUCTIONS


def build_training_messages(
    example: TrainingExample,
    template: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build chat messages for TRL / SFT training on a single document.

    The format mirrors NuExtract3 structured extraction: user message contains
    the document image; assistant message is the ground-truth JSON.

    Args:
        example: Labeled training example.
        template: NuExtract template used during training.

    Returns:
        OpenAI-style messages list (image paths stored as placeholders for the
        collator to resolve during training).
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(example.document_path),
                }
            ],
        },
        {
            "role": "assistant",
            "content": json.dumps(example.ground_truth, ensure_ascii=False),
        },
        {
            "role": "metadata",
            "content": {
                "template": template_to_json(template, indent=None),
                "instructions": example.instructions,
                "enable_thinking": False,
            },
        },
    ]


def load_training_jsonl(path: str | Path) -> list[TrainingExample]:
    """Load training examples from JSONL.

    Args:
        path: Path to JSONL with ``id``, ``document``, and ``ground_truth``.

    Returns:
        Parsed training examples.
    """
    file_path = Path(path)
    base_dir = file_path.parent
    examples: list[TrainingExample] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        doc_path = Path(record["document"])
        if not doc_path.is_absolute():
            doc_path = (base_dir / doc_path).resolve()
        examples.append(
            TrainingExample(
                example_id=record["id"],
                document_path=doc_path,
                ground_truth=record["ground_truth"],
                instructions=record.get("instructions", DEFAULT_INSTRUCTIONS),
            )
        )
    return examples


def write_training_jsonl(examples: list[TrainingExample], path: str | Path) -> None:
    """Write training examples to JSONL.

    Args:
        examples: Training examples to persist.
        path: Output JSONL path.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "id": ex.example_id,
                "document": str(ex.document_path),
                "ground_truth": ex.ground_truth,
                "instructions": ex.instructions,
            },
            ensure_ascii=False,
        )
        for ex in examples
    ]
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
