"""
Stage 3 — Evidence Matching (Rule-based, no LLM)

Checks whether the submitted images satisfy the minimum evidence requirements
defined in evidence_requirements.csv.

"Evidence standard met" means the images are SUFFICIENT TO EVALUATE the claim,
not that they support it. Even if the images show no damage (contradicted),
the evidence standard is met if the relevant area is clearly visible.

Evidence standard is NOT met only when:
  - The claimed area is not visible (wrong angle, cropped, obstructed)
  - The image quality is too poor to make any assessment
  - The image doesn't show the claimed object at all AND nothing useful is visible
"""

from __future__ import annotations

from config import (
    ISSUE_TO_EVIDENCE_FAMILY,
    load_evidence_requirements,
)
from models import (
    ClaimExtraction,
    ClaimInput,
    EvidenceCheck,
    ImageAnalysisResult,
)

# Load requirements once at module level
_EVIDENCE_REQUIREMENTS = None


def _get_requirements() -> list[dict]:
    global _EVIDENCE_REQUIREMENTS
    if _EVIDENCE_REQUIREMENTS is None:
        _EVIDENCE_REQUIREMENTS = load_evidence_requirements()
    return _EVIDENCE_REQUIREMENTS


def match_evidence(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image_analysis: ImageAnalysisResult,
) -> EvidenceCheck:
    """
    Run Stage 3: Check if the image evidence meets the minimum requirements.

    This is a pure rule-based check — no LLM calls.

    Args:
        claim: Raw claim input.
        extraction: Stage 1 output.
        image_analysis: Stage 2 output.

    Returns:
        EvidenceCheck indicating whether evidence standards are met.
    """
    requirements = _get_requirements()
    applicable_req_ids: list[str] = []

    # ── Find applicable requirements ────────────────────────────────────
    claim_object = claim.claim_object
    damage_type = extraction.claimed_damage_type

    # Get the evidence family for this damage type + object
    family_mapping = ISSUE_TO_EVIDENCE_FAMILY.get(claim_object, {})
    evidence_family = family_mapping.get(damage_type)

    for req in requirements:
        obj = req["claim_object"]
        applies_to = req["applies_to"]

        # Check if this requirement applies
        is_applicable = False

        # Universal requirements (apply to all objects)
        if obj == "all":
            if (
                applies_to == "general claim review"
                or applies_to == "multi-image rows"
                and len(image_analysis.images) > 1
                or applies_to == "reviewability"
            ):
                is_applicable = True

        # Object-specific requirements
        elif obj == claim_object and evidence_family:
            if evidence_family.lower() in applies_to.lower():
                is_applicable = True

        if is_applicable:
            applicable_req_ids.append(req["requirement_id"])

    # ── Evaluate evidence against requirements ──────────────────────────

    # Check 1: Is at least one image relevant and showing the claimed object?
    any_object_match = any(img.object_matches_claim for img in image_analysis.images)

    # Check 2: Is the claimed part visible in at least one image?
    claimed_part = extraction.claimed_object_part
    any_part_visible = any(
        img.part_matches_claim or img.part_visible == claimed_part
        for img in image_analysis.images
        if img.object_matches_claim
    )

    # Check 3: Are there severe quality issues blocking evaluation?
    blocking_quality_issues = {"wrong_angle", "cropped_or_obstructed"}
    all_images_blocked = all(
        bool(set(img.quality_issues) & blocking_quality_issues) or not img.object_matches_claim
        for img in image_analysis.images
    )

    # Check 4: Is at least one image relevant?
    any_usable_image = any(img.relevance_to_claim != "not_relevant" for img in image_analysis.images)

    # ── Decision logic ──────────────────────────────────────────────────

    if not any_object_match:
        # No image shows the claimed object type
        if any(img.object_visible != "unknown" for img in image_analysis.images):
            # Images show something, just not the claimed object
            return EvidenceCheck(
                evidence_standard_met=True,  # Can still evaluate (might be wrong_object)
                evidence_standard_met_reason=(
                    "The images show an object but it may not match the claimed "
                    f"{claim_object}. The evidence is sufficient to evaluate the mismatch."
                ),
                applicable_requirements=applicable_req_ids,
            )
        else:
            return EvidenceCheck(
                evidence_standard_met=False,
                evidence_standard_met_reason=(
                    "The images do not clearly show the claimed object or any identifiable object."
                ),
                applicable_requirements=applicable_req_ids,
            )

    if all_images_blocked and not any_part_visible:
        return EvidenceCheck(
            evidence_standard_met=False,
            evidence_standard_met_reason=(
                f"The claimed {claimed_part} is not visible due to "
                "wrong angle, cropping, or obstruction in all submitted images."
            ),
            applicable_requirements=applicable_req_ids,
        )

    if not any_usable_image:
        return EvidenceCheck(
            evidence_standard_met=False,
            evidence_standard_met_reason=("None of the submitted images are relevant to the claim."),
            applicable_requirements=applicable_req_ids,
        )

    # If we reach here, at least one image shows the claimed object
    # and is of sufficient quality to evaluate

    # Build a positive reason
    matching_images = [img for img in image_analysis.images if img.object_matches_claim]
    part_match = any(img.part_visible == claimed_part for img in matching_images)

    if part_match:
        reason = f"The {claimed_part} is visible and the image quality is sufficient to evaluate the claim."
    else:
        visible_parts = sorted({img.part_visible for img in matching_images if img.part_visible != "unknown"})
        if visible_parts:
            reason = (
                f"The {claim_object} is visible and shows "
                f"{', '.join(visible_parts)}, providing enough context to evaluate."
            )
        else:
            reason = (
                f"The {claim_object} is visible in the submitted images, "
                "providing sufficient context for evaluation."
            )

    return EvidenceCheck(
        evidence_standard_met=True,
        evidence_standard_met_reason=reason,
        applicable_requirements=applicable_req_ids,
    )
