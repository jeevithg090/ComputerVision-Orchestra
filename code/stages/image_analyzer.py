"""
Stage 2 — Image Analysis

Sends all images for a claim in a single Responses API vision call.
This allows the model to cross-reference images (e.g., wide shot vs. close-up)
and identify which specific image best supports or contradicts the claim.

Images that cannot be found on disk are not sent to the model; they receive a
conservative "unreadable" analysis so image IDs always stay aligned with the
pixels the model actually saw.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import (
    DATASET_DIR,
    IMAGE_ANALYSIS_MODEL,
    ISSUE_TYPE_VALUES,
    OBJECT_PARTS,
    SEVERITY_VALUES,
)
from llm_client import LLMClient
from models import ClaimExtraction, ClaimInput, ImageAnalysisResult, SingleImageAnalysis
from normalize import DAMAGE_SYNONYMS, normalize_label, normalize_severity
from prompts import IMAGE_ANALYSIS_SYSTEM, build_image_analysis_user_prompt

# Below this confidence the model's damage label is treated as unknown.
MIN_DAMAGE_CONFIDENCE = 0.70


def analyze_images(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    client: LLMClient,
) -> ImageAnalysisResult:
    """
    Run Stage 2: Analyze all images for a claim using a vision model.

    All readable images are sent in a single API call for cross-reference context.

    Args:
        claim: Raw claim input row.
        extraction: Stage 1 output with extracted claim details.
        client: LLM client for API calls.

    Returns:
        ImageAnalysisResult with exactly one analysis per submitted image ID.
    """
    present, _missing = resolve_images(claim.image_ids, claim.image_path_list)
    present_ids = [image_id for image_id, _ in present]

    raw_images: list[dict[str, Any]] = []
    if present:
        user_prompt = build_image_analysis_user_prompt(
            claim_object=claim.claim_object,
            claim_summary=extraction.claim_summary,
            claimed_object_part=extraction.claimed_object_part,
            image_ids=present_ids,
        )
        result = client.call_vision(
            model=IMAGE_ANALYSIS_MODEL,
            system_message=IMAGE_ANALYSIS_SYSTEM,
            user_text=user_prompt,
            image_paths=[path for _, path in present],
        )
        raw_images = result.get("images", []) or []
        # Handle a model that returns a single analysis instead of a list.
        if not raw_images and any(k in result for k in ("image_id", "object_visible")):
            raw_images = [result]

    parsed: dict[str, SingleImageAnalysis] = {}
    for i, raw in enumerate(raw_images):
        if not isinstance(raw, dict):
            continue
        img_id = str(raw.get("image_id", ""))
        if img_id not in present_ids:
            # Unknown/hallucinated ID: fall back to positional alignment.
            if i >= len(present_ids):
                continue
            img_id = present_ids[i]
        if img_id in parsed:
            continue
        parsed[img_id] = parse_image_analysis(raw, img_id, claim, extraction)

    analyses = [parsed.get(image_id) or unreadable_image_analysis(image_id) for image_id in claim.image_ids]
    return ImageAnalysisResult(images=analyses)


def parse_image_analysis(
    raw: dict[str, Any],
    img_id: str,
    claim: ClaimInput,
    extraction: ClaimExtraction,
) -> SingleImageAnalysis:
    """Normalize one raw per-image model result into a validated model."""
    # Normalize damage_type and check confidence
    damage_type = _normalize_damage_type(raw.get("damage_type") or raw.get("visible_damage_type", "unknown"))
    damage_confidence = _coerce_confidence(raw.get("damage_confidence", 1.0))
    if damage_confidence < MIN_DAMAGE_CONFIDENCE:
        damage_type = "unknown"
    visible_damage_type = _normalize_damage_type(raw.get("visible_damage_type", damage_type))

    # Normalize part_visible
    valid_parts = OBJECT_PARTS.get(claim.claim_object, frozenset(["unknown"]))
    part_visible = normalize_label(raw.get("part_visible", "unknown"), valid_parts)

    # Normalize quality issues
    quality_issues = _as_list(raw.get("quality_issues", []))
    for blocker in _as_list(raw.get("quality_blockers", [])):
        if blocker not in quality_issues:
            quality_issues.append(blocker)

    object_category = raw.get("object_category") or raw.get("object_visible", "unknown")
    object_category = str(object_category).lower().strip().replace(" ", "_")
    if object_category not in {"car", "laptop", "package"}:
        object_category = "other"
    object_matches = bool(
        raw.get(
            "object_matches_claim",
            raw.get("is_expected_object", object_category == claim.claim_object),
        )
    )
    part_matches = bool(raw.get("part_matches_claim", part_visible == extraction.claimed_object_part))
    text_content = raw.get("text_instruction_content") or raw.get("instructional_text", "")
    text_detected = bool(raw.get("text_instruction_detected", bool(text_content)))
    authenticity = raw.get("image_authenticity", "original_photo")
    appears_original = bool(raw.get("appears_original", authenticity == "original_photo"))

    return SingleImageAnalysis(
        image_id=img_id,
        object_category=object_category,
        object_visible=raw.get("object_visible", "unknown"),
        is_expected_object=object_matches,
        object_matches_claim=object_matches,
        part_visible=part_visible,
        part_matches_claim=part_matches,
        damage_visible=bool(raw.get("damage_visible", False)),
        claimed_damage_visible=bool(raw.get("claimed_damage_visible", False)),
        other_damage_visible=bool(raw.get("other_damage_visible", False)),
        damage_type=damage_type,
        visible_damage_type=visible_damage_type,
        damage_confidence=damage_confidence,
        damage_severity=normalize_severity(raw.get("damage_severity", "unknown"), SEVERITY_VALUES),
        quality_issues=quality_issues,
        text_instruction_detected=text_detected,
        text_instruction_content=text_content,
        instructional_text=text_content,
        appears_original=appears_original,
        image_authenticity=authenticity,
        support_role=raw.get("support_role", "unclear"),
        relevance_to_claim=raw.get("relevance_to_claim", "relevant"),
    )


def unreadable_image_analysis(image_id: str) -> SingleImageAnalysis:
    """Conservative analysis for an image the model did not (or could not) assess."""
    return SingleImageAnalysis(
        image_id=image_id,
        object_category="other",
        object_visible="unknown",
        is_expected_object=False,
        object_matches_claim=False,
        part_visible="unknown",
        part_matches_claim=False,
        damage_visible=False,
        claimed_damage_visible=False,
        other_damage_visible=False,
        damage_type="unknown",
        visible_damage_type="unknown",
        damage_confidence=0.0,
        damage_severity="unknown",
        quality_issues=["cropped_or_obstructed"],
        text_instruction_detected=False,
        text_instruction_content="",
        instructional_text="",
        appears_original=True,
        image_authenticity="unknown",
        support_role="insufficient",
        relevance_to_claim="not_relevant",
    )


def resolve_images(
    image_ids: list[str],
    relative_paths: list[str],
) -> tuple[list[tuple[str, Path]], list[str]]:
    """
    Resolve image paths (relative to ``dataset/``, or absolute).

    Returns ``(present, missing)`` where ``present`` is a list of
    ``(image_id, path)`` pairs for files that exist and ``missing`` lists the
    IDs of images that could not be found.
    """
    present: list[tuple[str, Path]] = []
    missing: list[str] = []
    for image_id, rel in zip(image_ids, relative_paths, strict=False):
        candidate = Path(rel.strip())
        full = candidate if candidate.is_absolute() else DATASET_DIR / candidate
        if full.is_file():
            present.append((image_id, full))
        else:
            print(f"  ⚠ Image not found: {full}")
            missing.append(image_id)
    return present, missing


def _normalize_damage_type(value: object) -> str:
    """Normalize a damage type string to an allowed value (exact or synonym match)."""
    return normalize_label(value, ISSUE_TYPE_VALUES, synonyms=DAMAGE_SYNONYMS, substring=False)


def _coerce_confidence(value: object) -> float:
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return min(max(conf, 0.0), 1.0)


def _as_list(value: object) -> list[str]:
    """Accept a list or comma-separated string; drop blanks and 'none'."""
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        items = []
    cleaned = [item.strip() for item in items]
    return [item for item in cleaned if item and item != "none"]
