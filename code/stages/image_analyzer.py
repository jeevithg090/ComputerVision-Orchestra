"""
Stage 2 — Image Analysis

Sends all images for a claim in a single Responses API vision call.
This allows the model to cross-reference images (e.g., wide shot vs. close-up)
and identify which specific image best supports or contradicts the claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from models import ClaimInput, ClaimExtraction, ImageAnalysisResult, SingleImageAnalysis
from llm_client import LLMClient
from config import (
    IMAGE_ANALYSIS_MODEL,
    IMAGE_ANALYSIS_TEMPERATURE,
    DATASET_DIR,
    ISSUE_TYPE_VALUES,
    OBJECT_PARTS,
    PROJECT_ROOT,
)
from prompts import IMAGE_ANALYSIS_SYSTEM, build_image_analysis_user_prompt


def analyze_images(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    client: LLMClient,
) -> ImageAnalysisResult:
    """
    Run Stage 2: Analyze all images for a claim using a vision model.

    All images are sent in a single API call for cross-reference context.

    Args:
        claim: Raw claim input row.
        extraction: Stage 1 output with extracted claim details.
        client: LLM client for API calls.

    Returns:
        ImageAnalysisResult with per-image analysis.
    """
    image_ids = claim.image_ids
    image_paths = _resolve_image_paths(claim.image_path_list)

    # Build the user prompt
    user_prompt = build_image_analysis_user_prompt(
        claim_object=claim.claim_object,
        claim_summary=extraction.claim_summary,
        claimed_object_part=extraction.claimed_object_part,
        image_ids=image_ids,
    )

    # Call vision model with all images
    result = client.call_vision(
        model=IMAGE_ANALYSIS_MODEL,
        system_message=IMAGE_ANALYSIS_SYSTEM,
        user_text=user_prompt,
        image_paths=image_paths,
        temperature=IMAGE_ANALYSIS_TEMPERATURE,
    )

    # Parse and validate per-image results
    raw_images = result.get("images", [])

    # Handle case where model returns a single image analysis not in a list
    if not raw_images and any(k in result for k in ("image_id", "object_visible")):
        raw_images = [result]

    analyses: List[SingleImageAnalysis] = []
    for i, raw in enumerate(raw_images):
        img_id = raw.get("image_id", image_ids[i] if i < len(image_ids) else f"img_{i+1}")

        # Normalize damage_type and check confidence
        damage_type = raw.get("damage_type") or raw.get("visible_damage_type", "unknown")
        if damage_type not in ISSUE_TYPE_VALUES:
            damage_type = _normalize_damage_type(damage_type)
            
        damage_confidence = float(raw.get("damage_confidence", 1.0))
        if damage_confidence < 0.70:
            damage_type = "unknown"
        visible_damage_type = raw.get("visible_damage_type", damage_type)
        if visible_damage_type not in ISSUE_TYPE_VALUES:
            visible_damage_type = _normalize_damage_type(visible_damage_type)

        # Normalize part_visible
        part_visible = raw.get("part_visible", "unknown")
        valid_parts = OBJECT_PARTS.get(claim.claim_object, frozenset(["unknown"]))
        if part_visible not in valid_parts:
            part_visible = _normalize_part(part_visible, valid_parts)

        # Normalize quality issues
        quality_issues = raw.get("quality_issues", [])
        if isinstance(quality_issues, str):
            quality_issues = [q.strip() for q in quality_issues.split(",") if q.strip()]
        quality_issues = [q for q in quality_issues if q and q != "none"]
        quality_blockers = raw.get("quality_blockers", [])
        if isinstance(quality_blockers, str):
            quality_blockers = [q.strip() for q in quality_blockers.split(",") if q.strip()]
        for blocker in quality_blockers:
            if blocker and blocker != "none" and blocker not in quality_issues:
                quality_issues.append(blocker)

        object_category = raw.get("object_category") or raw.get("object_visible", "unknown")
        object_category = str(object_category).lower().strip().replace(" ", "_")
        if object_category not in {"car", "laptop", "package"}:
            object_category = "other"
        object_matches = bool(raw.get(
            "object_matches_claim",
            raw.get("is_expected_object", object_category == claim.claim_object),
        ))
        part_matches = bool(raw.get("part_matches_claim", part_visible == extraction.claimed_object_part))
        text_content = raw.get("text_instruction_content") or raw.get("instructional_text", "")
        text_detected = bool(raw.get("text_instruction_detected", bool(text_content)))
        authenticity = raw.get("image_authenticity", "original_photo")
        appears_original = bool(raw.get("appears_original", authenticity == "original_photo"))

        analyses.append(SingleImageAnalysis(
            image_id=img_id,
            object_category=object_category,
            object_visible=raw.get("object_visible", "unknown"),
            is_expected_object=object_matches,
            object_matches_claim=object_matches,
            part_visible=part_visible,
            part_matches_claim=part_matches,
            damage_visible=raw.get("damage_visible", False),
            claimed_damage_visible=bool(raw.get("claimed_damage_visible", False)),
            other_damage_visible=bool(raw.get("other_damage_visible", False)),
            damage_type=damage_type,
            visible_damage_type=visible_damage_type,
            damage_confidence=damage_confidence,
            damage_severity=_normalize_severity(raw.get("damage_severity", "unknown")),
            quality_issues=quality_issues,
            text_instruction_detected=text_detected,
            text_instruction_content=text_content,
            instructional_text=text_content,
            appears_original=appears_original,
            image_authenticity=authenticity,
            support_role=raw.get("support_role", "unclear"),
            relevance_to_claim=raw.get("relevance_to_claim", "relevant"),
        ))

    # If model returned fewer analyses than images, fill in defaults
    while len(analyses) < len(image_ids):
        idx = len(analyses)
        analyses.append(SingleImageAnalysis(
            image_id=image_ids[idx],
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
        ))

    return ImageAnalysisResult(images=analyses)


def _resolve_image_paths(relative_paths: List[str]) -> List[Path]:
    """Resolve relative image paths from the dataset directory."""
    resolved = []
    for p in relative_paths:
        full = PROJECT_ROOT / "dataset" / p.strip()
        if full.exists():
            resolved.append(full)
        else:
            # Try without leading directory
            alt = DATASET_DIR / p.strip()
            if alt.exists():
                resolved.append(alt)
            else:
                print(f"  ⚠ Image not found: {full}")
    return resolved


def _normalize_damage_type(value: str) -> str:
    """Normalize a damage type string to an allowed value."""
    v = value.lower().strip().replace(" ", "_").replace("-", "_")
    from config import ISSUE_TYPE_VALUES
    if v in ISSUE_TYPE_VALUES:
        return v
    synonyms = {
        "cracked": "crack", "broken": "broken_part", "shattered": "glass_shatter",
        "torn": "torn_packaging", "crushed": "crushed_packaging",
        "water": "water_damage", "wet": "water_damage", "missing": "missing_part",
        "dented": "dent", "scratched": "scratch", "stained": "stain",
        "no_damage": "none", "no damage": "none", "not_visible": "unknown",
    }
    for syn, target in synonyms.items():
        if syn in v:
            return target
    return "unknown"


def _normalize_severity(value: str) -> str:
    """Normalize severity to allowed values."""
    v = value.lower().strip()
    from config import SEVERITY_VALUES
    if v in SEVERITY_VALUES:
        return v
    if "no" in v:
        return "none"
    return "unknown"


def _normalize_part(value: str, valid_parts: frozenset[str]) -> str:
    """Normalize an object part to the closest allowed value."""
    v = value.lower().strip().replace(" ", "_").replace("-", "_")
    if v in valid_parts:
        return v
    # Substring match
    for candidate in valid_parts:
        if candidate in v or v in candidate:
            return candidate
    return "unknown"
