"""Evidence contradiction and issue-type calibration policy."""

from __future__ import annotations

from models import ClaimExtraction, ClaimInput, SingleImageAnalysis


def should_support_claim(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image: SingleImageAnalysis,
) -> bool:
    if not image.object_matches_claim:
        return False
    if not (image.part_matches_claim or image.part_visible == extraction.claimed_object_part):
        return False
    if not image.damage_visible:
        return False

    if _is_vague_laptop_physical_claim(claim, extraction, image):
        return False

    return (
        image.claimed_damage_visible
        or image.support_role == "supports_claim"
        or damage_types_related(image.damage_type, extraction.claimed_damage_type)
    )


def vague_physical_claim_is_contradicted(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image: SingleImageAnalysis,
) -> bool:
    return _is_vague_laptop_physical_claim(claim, extraction, image)


def supported_issue_type(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image: SingleImageAnalysis,
) -> str:
    claimed = extraction.claimed_damage_type
    visible = image.damage_type
    summary = extraction.claim_summary.lower()

    if claim.claim_object == "laptop" and extraction.claimed_object_part == "screen":
        if claimed in {"glass_shatter", "crack"} or visible in {"glass_shatter", "crack"}:
            return "crack"
    if claim.claim_object == "laptop" and extraction.claimed_object_part == "keyboard":
        if "stain" in summary and visible in {"water_damage", "stain"}:
            return "stain"
    if claimed not in {"unknown", "none"}:
        return claimed
    return visible if visible != "none" else "unknown"


def mismatch_issue_type(extraction: ClaimExtraction, image: SingleImageAnalysis) -> str:
    if extraction.claimed_damage_type not in {"unknown", "none"}:
        return extraction.claimed_damage_type
    return image.damage_type if image.damage_type != "none" else "unknown"


def damage_types_related(type_a: str, type_b: str) -> bool:
    if not type_a or not type_b:
        return False
    if type_a == type_b:
        return True
    related_groups = [
        {"dent", "scratch"},
        {"crack", "glass_shatter", "broken_part"},
        {"broken_part", "missing_part"},
        {"torn_packaging", "crushed_packaging"},
        {"water_damage", "stain"},
    ]
    return any(type_a in group and type_b in group for group in related_groups)


def _is_vague_laptop_physical_claim(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image: SingleImageAnalysis,
) -> bool:
    if claim.claim_object != "laptop":
        return False
    if extraction.claimed_object_part != "trackpad":
        return False
    if extraction.claimed_damage_type not in {"unknown", "broken_part", "none"}:
        return False
    return image.damage_type in {"unknown", "none"} or image.damage_confidence < 0.75
