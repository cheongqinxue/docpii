"""Batch evaluation runner for systematic PII extraction tests."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docpii.evaluation.metrics import EvaluationResult, score_extraction
from docpii.inference.client import NuExtractClient
from docpii.inference.documents import load_document_images
from docpii.templates.nuextract import default_extraction_template

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkSample:
    """One labeled document in an evaluation set."""

    sample_id: str
    document_path: Path
    ground_truth: dict[str, Any]
    instructions: str | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    """Summary of a full benchmark run."""

    results: list[EvaluationResult]
    mean_score: float
    failure_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-friendly dictionary."""
        return {
            "mean_score": self.mean_score,
            "failure_rate": self.failure_rate,
            "results": [
                {
                    "document_id": r.document_id,
                    "average_score": r.average_score,
                    "parse_error": r.parse_error,
                    "num_leaves": len(r.leaf_scores),
                }
                for r in self.results
            ],
        }


def load_manifest(manifest_path: str | Path) -> list[BenchmarkSample]:
    """Load a benchmark manifest JSONL or JSON file.

    Each record must contain ``id``, ``document``, and ``ground_truth`` keys.
    Optional ``instructions`` override default extraction instructions.

    Args:
        manifest_path: Path to ``manifest.jsonl`` or ``manifest.json``.

    Returns:
        Parsed benchmark samples.
    """
    path = Path(manifest_path)
    raw_records: list[dict[str, Any]]
    if path.suffix == ".jsonl":
        raw_records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_records = payload if isinstance(payload, list) else payload["samples"]

    base_dir = path.parent
    samples: list[BenchmarkSample] = []
    for record in raw_records:
        doc_path = Path(record["document"])
        if not doc_path.is_absolute():
            doc_path = (base_dir / doc_path).resolve()
        samples.append(
            BenchmarkSample(
                sample_id=record["id"],
                document_path=doc_path,
                ground_truth=record["ground_truth"],
                instructions=record.get("instructions"),
            )
        )
    return samples


class EvaluationRunner:
    """Run systematic extraction evaluation against labeled documents."""

    def __init__(self, client: NuExtractClient | None = None) -> None:
        """Initialize the runner.

        Args:
            client: Optional NuExtract client; a default client is created if omitted.
        """
        self.client = client or NuExtractClient()
        self.template = default_extraction_template()

    def run(self, samples: list[BenchmarkSample]) -> BenchmarkReport:
        """Evaluate all samples and aggregate scores.

        Args:
            samples: Labeled benchmark documents.

        Returns:
            Benchmark report with per-document and aggregate metrics.
        """
        results: list[EvaluationResult] = []
        failures = 0

        for sample in samples:
            try:
                images = load_document_images(sample.document_path)
                predicted = self.client.extract_from_images(
                    images,
                    template=self.template,
                    instructions=sample.instructions,
                )
                result = score_extraction(
                    predicted,
                    sample.ground_truth,
                    self.template,
                    document_id=sample.sample_id,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                failures += 1
                logger.exception("Failed to evaluate sample %s", sample.sample_id)
                result = EvaluationResult(
                    document_id=sample.sample_id,
                    average_score=0.0,
                    leaf_scores=[],
                    parse_error=str(exc),
                )
            results.append(result)

        scores = [r.average_score for r in results if r.parse_error is None]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        failure_rate = failures / len(results) if results else 0.0
        return BenchmarkReport(results=results, mean_score=mean_score, failure_rate=failure_rate)

    def run_manifest(self, manifest_path: str | Path) -> BenchmarkReport:
        """Load a manifest and run evaluation.

        Args:
            manifest_path: Path to benchmark manifest file.

        Returns:
            Benchmark report.
        """
        return self.run(load_manifest(manifest_path))
