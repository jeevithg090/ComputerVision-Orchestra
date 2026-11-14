"""Final risk-flag calibration.

The image-analysis stage can record diagnostics for every image. The final CSV
should only expose risk flags that affect the decision evidence.
"""

from __future__ import annotations

from collections.abc import Iterable

from config import RISK_FLAG_VALUES
from models import ImageAnalysisResult

QUALITY_FLAGS = {"blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle"}


def finalize_risk_flags(
    *,
    base_flags: Iterable[str],
    image_analysis: ImageAnalysisResult,
    claim_status: str,
    rule_id: str,
    supporting_ids: Iterable[str],
    user_history: dict | None = None,
) -> str:
    flags: set[str] = set()
    base = {f for f in base_flags if f and f != "none"}
    selected_ids = {sid for sid in supporting_ids if sid and sid != "none"}
    selected_images = [
        img for img in image_analysis.images if not selected_ids or img.image_id in selected_ids
    ]

    history_flags = _history_flags(user_history)
    flags.update(history_flags)

    if rule_id == "wrong_object":
        flags.update({"wrong_object", "claim_mismatch"})
    if rule_id == "wrong_part_damage":
        flags.update({"claim_mismatch", "wrong_object_part"})
    if rule_id == "image_text_instruction":
        flags.add("text_instruction_present")
    if rule_id in {"claimed_part_undamaged", "vague_laptop_physical_damage"}:
        flags.add("damage_not_visible")
    if rule_id == "claimed_part_not_visible":
        flags.add("wrong_object_part")

    for img in selected_images:
        if not img.object_matches_claim:
            flags.add("wrong_object")
        if img.text_instruction_detected:
            flags.add("text_instruction_present")
        if not img.appears_original and claim_status != "supported":
            flags.add("non_original_image")
        if claim_status != "supported":
            flags.update(q for q in img.quality_issues if q in QUALITY_FLAGS)

    # Preserve explicit severe evidence risks from earlier stages.
    for flag in base:
        if flag in {"possible_manipulation", "non_original_image"} and claim_status != "supported":
            flags.add(flag)

    evidence_risks = flags - {"user_history_risk", "manual_review_required"}
    if (
        "manual_review_required" in history_flags
        or "text_instruction_present" in flags
        or "wrong_object" in flags
        or "wrong_object_part" in flags
        or "non_original_image" in flags
        or ("user_history_risk" in flags and evidence_risks)
    ):
        flags.add("manual_review_required")

    valid = sorted(f for f in flags if f in RISK_FLAG_VALUES and f != "none")
    return ";".join(valid) if valid else "none"


def _history_flags(user_history: dict | None) -> set[str]:
    if not user_history:
        return set()
    raw = user_history.get("history_flags", "none") or "none"
    return {f.strip() for f in raw.split(";") if f.strip() in RISK_FLAG_VALUES and f.strip() != "none"}
