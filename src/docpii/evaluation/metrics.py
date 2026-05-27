"""Evaluation metrics aligned with NuExtract structured extraction scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _indel_similarity(a: str, b: str) -> float:
    """Compute normalized indel (Levenshtein without substitution) similarity.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Similarity in ``[0, 1]`` where 1 is an exact match.
    """
    if a == b:
        return 1.0
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    # Wagner-Fischer style indel distance on characters.
    len_a, len_b = len(a), len(b)
    prev = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        curr = [i] + [0] * len_b
        for j in range(1, len_b + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    distance = prev[len_b]
    return 1.0 - (distance / max(len(a), len(b)))


def _is_verbatim_type(node_type: str | list[Any] | None) -> bool:
    """Return whether a template leaf expects verbatim string matching."""
    return node_type == "verbatim-string"


def _leaf_score(
    predicted: Any,
    expected: Any,
    node_type: str | list[Any] | None,
) -> float:
    """Score a single aligned leaf value.

    Args:
        predicted: Predicted leaf value.
        expected: Ground-truth leaf value.
        node_type: NuExtract template type at this leaf.

    Returns:
        Score in ``[0, 1]``.
    """
    if predicted is None and expected is None:
        return 1.0
    if predicted is None or expected is None:
        return 0.0

    if isinstance(predicted, str) and isinstance(expected, str):
        if _is_verbatim_type(node_type):
            return _indel_similarity(predicted, expected)
        return _indel_similarity(predicted.lower(), expected.lower())

    return 1.0 if predicted == expected else 0.0


def _align_and_score(
    predicted: Any,
    expected: Any,
    template: Any,
    path: str = "",
) -> list[float]:
    """Recursively align JSON trees and collect leaf scores.

    Args:
        predicted: Predicted JSON subtree.
        expected: Ground-truth JSON subtree.
        template: NuExtract template subtree for type hints.
        path: Dot-separated path for debugging.

    Returns:
        List of leaf scores under this subtree.
    """
    scores: list[float] = []

    if isinstance(template, dict):
        for key, child_template in template.items():
            child_pred = predicted.get(key) if isinstance(predicted, dict) else None
            child_exp = expected.get(key) if isinstance(expected, dict) else None
            scores.extend(
                _align_and_score(child_pred, child_exp, child_template, f"{path}.{key}")
            )
        return scores

    if isinstance(template, list) and template:
        item_template = template[0]
        pred_list = predicted if isinstance(predicted, list) else []
        exp_list = expected if isinstance(expected, list) else []
        max_len = max(len(pred_list), len(exp_list))
        for idx in range(max_len):
            child_pred = pred_list[idx] if idx < len(pred_list) else None
            child_exp = exp_list[idx] if idx < len(exp_list) else None
            scores.extend(
                _align_and_score(child_pred, child_exp, item_template, f"{path}[{idx}]")
            )
        return scores

    scores.append(_leaf_score(predicted, expected, template))
    return scores


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate scores for one document."""

    document_id: str
    average_score: float
    leaf_scores: list[float]
    parse_error: str | None = None


def score_extraction(
    predicted: dict[str, Any],
    expected: dict[str, Any],
    template: dict[str, Any],
    *,
    document_id: str = "",
) -> EvaluationResult:
    """Score a predicted extraction against ground truth.

    Args:
        predicted: Model output JSON.
        expected: Ground-truth JSON.
        template: NuExtract template used for extraction.
        document_id: Optional document identifier for reporting.

    Returns:
        ``EvaluationResult`` with average leaf score.
    """
    leaf_scores = _align_and_score(predicted, expected, template)
    average = sum(leaf_scores) / len(leaf_scores) if leaf_scores else 0.0
    return EvaluationResult(
        document_id=document_id,
        average_score=average,
        leaf_scores=leaf_scores,
    )
