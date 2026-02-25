"""
Case Search Engine — search and filter previous analyses stored in
gemini_predictions.jsonl.

Supports:
  - Full-text search across classifications, clinical analysis, notes
  - Biomarker range filtering (e.g. entropy > 0.5)
  - Classification filtering (Normal, Sarnat I/II/III, Seizures)
  - Date range filtering
  - Similarity search: find cases with the most similar biomarker profile
"""

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

JSONL_FILE = os.path.join(os.path.dirname(__file__), "analysis_logs", "gemini_predictions.jsonl")

# Biomarker keys used for similarity comparisons and range filtering
BIOMARKER_KEYS = [
    "average_sample_entropy",
    "average_jerk",
    "average_fractal_dimension",
    "average_kinetic_energy",
    "average_root_stress",
    "bilateral_symmetry_index",
    "average_lower_limb_ke",
    "angular_jerk_index",
    "head_stability_index",
    "average_com_velocity",
    "variability_index",
    "cs_risk_score",
]


def _load_cases() -> List[Dict[str, Any]]:
    """Load all cases from the JSONL file."""
    if not os.path.exists(JSONL_FILE):
        return []
    cases = []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return cases


def _get_text_fields(case: Dict[str, Any]) -> str:
    """Extract all searchable text from a case into a single lowercase string."""
    parts = []
    pred = case.get("gemini_prediction") or {}
    parts.append(pred.get("classification", ""))
    parts.append(pred.get("clinicalAnalysis", ""))
    parts.append(pred.get("seizureType", ""))
    parts.append(pred.get("differentialAlert") or "")
    for rec in pred.get("recommendations", []):
        parts.append(rec)
    parts.append(case.get("ground_truth") or "")
    parts.append(case.get("doctor_notes") or "")
    meta = case.get("metadata") or {}
    parts.append(meta.get("filename") or "")
    correction = case.get("expert_correction") or {}
    parts.append(correction.get("correctClassification", ""))
    parts.append(correction.get("notes", ""))
    return " ".join(parts).lower()


def text_search(query: str, cases: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
    """Search cases by text query. Matches against classification,
    clinical analysis, recommendations, doctor notes, and filename."""
    if cases is None:
        cases = _load_cases()
    query_lower = query.lower().strip()
    if not query_lower:
        return cases
    tokens = query_lower.split()
    results = []
    for case in cases:
        text = _get_text_fields(case)
        if all(token in text for token in tokens):
            results.append(case)
    return results


def filter_by_classification(
    classifications: List[str], cases: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:
    """Filter cases by one or more classification labels."""
    if cases is None:
        cases = _load_cases()
    cls_lower = {c.lower() for c in classifications}
    results = []
    for case in cases:
        pred = case.get("gemini_prediction") or {}
        case_cls = pred.get("classification", "").lower()
        if case_cls in cls_lower:
            results.append(case)
    return results


def filter_by_date(
    start: Optional[str] = None,
    end: Optional[str] = None,
    cases: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """Filter cases by date range (ISO format strings)."""
    if cases is None:
        cases = _load_cases()
    results = []
    for case in cases:
        ts = case.get("timestamp", "")
        if not ts:
            continue
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        results.append(case)
    return results


def filter_by_biomarker(
    filters: Dict[str, Dict[str, float]],
    cases: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """Filter cases by biomarker ranges.

    Args:
        filters: Dict mapping biomarker name to {"min": float, "max": float}.
                 Either min or max can be omitted.
                 Example: {"average_sample_entropy": {"min": 0.3, "max": 0.6}}
        cases: Optional pre-loaded case list.
    """
    if cases is None:
        cases = _load_cases()
    results = []
    for case in cases:
        bio = case.get("biomarkers") or {}
        match = True
        for key, bounds in filters.items():
            value = bio.get(key)
            if value is None:
                match = False
                break
            if "min" in bounds and value < bounds["min"]:
                match = False
                break
            if "max" in bounds and value > bounds["max"]:
                match = False
                break
        if match:
            results.append(case)
    return results


def filter_validated(validated: bool = True, cases: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
    """Filter for cases that have (or lack) expert validation."""
    if cases is None:
        cases = _load_cases()
    results = []
    for case in cases:
        has_gt = bool(case.get("ground_truth") or case.get("expert_correction"))
        if has_gt == validated:
            results.append(case)
    return results


def _biomarker_vector(case: Dict[str, Any]) -> Optional[List[float]]:
    """Extract a normalized biomarker vector for similarity comparison."""
    bio = case.get("biomarkers") or {}
    values = []
    for key in BIOMARKER_KEYS:
        v = bio.get(key)
        if v is None:
            return None
        values.append(float(v))
    return values


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def find_similar(
    target_biomarkers: Dict[str, float],
    top_k: int = 5,
    method: str = "cosine",
    cases: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """Find the most similar cases by biomarker profile.

    Args:
        target_biomarkers: Biomarker dict to compare against.
        top_k: Number of results to return.
        method: "cosine" or "euclidean".
        cases: Optional pre-loaded case list.

    Returns:
        List of cases sorted by similarity, each with an added
        "_similarity_score" field.
    """
    if cases is None:
        cases = _load_cases()

    target_vec = [float(target_biomarkers.get(k, 0.0)) for k in BIOMARKER_KEYS]

    scored = []
    for case in cases:
        vec = _biomarker_vector(case)
        if vec is None:
            continue
        if method == "euclidean":
            dist = _euclidean_distance(target_vec, vec)
            score = 1.0 / (1.0 + dist)  # convert to 0-1 similarity
        else:
            score = _cosine_similarity(target_vec, vec)
        entry = dict(case)
        entry["_similarity_score"] = round(score, 6)
        scored.append(entry)

    scored.sort(key=lambda x: x["_similarity_score"], reverse=True)
    return scored[:top_k]


def search(
    query: Optional[str] = None,
    classifications: Optional[List[str]] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    biomarker_filters: Optional[Dict[str, Dict[str, float]]] = None,
    validated_only: Optional[bool] = None,
    similar_to: Optional[Dict[str, float]] = None,
    top_k: int = 50,
    method: str = "cosine",
) -> List[Dict[str, Any]]:
    """Unified search combining all filter types.

    All filters are AND-combined. Pass only the filters you need.

    Args:
        query: Free text search string.
        classifications: List of classification labels to include.
        date_start: ISO date string for range start.
        date_end: ISO date string for range end.
        biomarker_filters: Biomarker range filters.
        validated_only: If True, only validated cases; if False, only unvalidated.
        similar_to: Biomarker dict for similarity ranking.
        top_k: Max results (applies to similarity search or final output).
        method: Similarity method ("cosine" or "euclidean").

    Returns:
        Filtered and optionally similarity-ranked list of cases.
    """
    cases = _load_cases()

    if query:
        cases = text_search(query, cases)

    if classifications:
        cases = filter_by_classification(classifications, cases)

    if date_start or date_end:
        cases = filter_by_date(date_start, date_end, cases)

    if biomarker_filters:
        cases = filter_by_biomarker(biomarker_filters, cases)

    if validated_only is not None:
        cases = filter_validated(validated_only, cases)

    if similar_to:
        return find_similar(similar_to, top_k=top_k, method=method, cases=cases)

    # Sort by timestamp descending by default
    cases.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return cases[:top_k]
