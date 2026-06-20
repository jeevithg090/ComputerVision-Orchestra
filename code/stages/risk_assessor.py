"""
Stage 4 — Risk Assessment (Rule-based, no LLM)

Aggregates risk flags from two sources:
  1. User history (user_history.csv) — past rejection rates, history flags
  2. Image analysis — quality issues, wrong objects, text instructions, non-original

The risk assessment ADDS CONTEXT but does not override visual evidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from models import (
    ClaimInput,
    ClaimExtraction,
    ImageAnalysisResult,
    RiskProfile,
)
from config import RISK_FLAG_VALUES, load_user_history


# Load user history once at module level
_USER_HISTORY = None


def _get_user_history() -> Dict[str, dict]:
    global _USER_HISTORY
    if _USER_HISTORY is None:
        _USER_HISTORY = load_user_history()
    return _USER_HISTORY


def assess_risk(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image_analysis: ImageAnalysisResult,
) -> RiskProfile:
    """
    Run Stage 4: Compute risk flags from user history and image analysis.

    This is a pure rule-based assessment — no LLM calls.

    Args:
        claim: Raw claim input.
        extraction: Stage 1 output.
        image_analysis: Stage 2 output.

    Returns:
        RiskProfile with aggregated risk flags.
    """
    user_history = _get_user_history()
    user_data = user_history.get(claim.user_id)

    flags: Set[str] = set()
    summary_parts: List[str] = []
    rejection_rate = 0.0

    # ── User History Flags ──────────────────────────────────────────────

    if user_data:
        # Propagate history_flags from the CSV
        history_flags = user_data.get("history_flags", "none")
        if history_flags and history_flags != "none":
            for flag in history_flags.split(";"):
                flag = flag.strip()
                if flag in RISK_FLAG_VALUES and flag != "none":
                    flags.add(flag)

        # Compute rejection rate
        total = user_data.get("past_claim_count", 0)
        rejected = user_data.get("rejected_claim", 0)
        if total > 0:
            rejection_rate = rejected / total
            # High rejection rate → user_history_risk
            if rejection_rate >= 0.3 and total >= 3:
                flags.add("user_history_risk")

        # High claim frequency in last 90 days
        recent = user_data.get("last_90_days_claim_count", 0)
        if recent >= 5:
            flags.add("user_history_risk")

        # If manual_review_claim count is high relative to total
        manual = user_data.get("manual_review_claim", 0)
        if total > 0 and (manual / total) >= 0.4 and total >= 3:
            flags.add("manual_review_required")

        summary_parts.append(user_data.get("history_summary", ""))
    else:
        summary_parts.append("No user history available.")

    # ── Image-Level Flags ───────────────────────────────────────────────

    has_claim_support = any(
        img.object_matches_claim and (
            img.claimed_damage_visible or
            img.support_role == "supports_claim" or
            (img.part_matches_claim and img.damage_visible)
        )
        for img in image_analysis.images
    )

    for img in image_analysis.images:
        # Wrong object
        if not img.object_matches_claim:
            flags.add("wrong_object")

        # Text instructions detected
        if img.text_instruction_detected:
            flags.add("text_instruction_present")

        # Non-original image (screenshot, printout)
        if not img.appears_original:
            flags.add("non_original_image")

        # Image quality issues
        for quality_issue in img.quality_issues:
            if quality_issue in RISK_FLAG_VALUES:
                flags.add(quality_issue)

        # Damage not visible when damage is claimed
        if not has_claim_support and not img.damage_visible and img.object_matches_claim:
            if img.relevance_to_claim == "relevant" and img.part_matches_claim:
                flags.add("damage_not_visible")

    # ── Adversarial Signals from Claim Extraction ───────────────────────

    if extraction.adversarial_signals:
        # Any adversarial signal in the conversation should trigger manual review
        flags.add("manual_review_required")

    # ── Check for Claim Mismatch ────────────────────────────────────────

    # If the image shows damage on a different part than claimed
    for img in image_analysis.images:
        if (img.object_matches_claim and
            img.damage_visible and
            not img.part_matches_claim and
            img.part_visible != extraction.claimed_object_part and
            img.part_visible != "unknown"):
            flags.add("claim_mismatch")
            flags.add("wrong_object_part")
            break

    # If the damage type visible is fundamentally different from claimed
    for img in image_analysis.images:
        if (img.object_matches_claim and
            img.damage_visible and
            img.damage_type != extraction.claimed_damage_type and
            img.damage_type not in ("unknown", "none")):
            # Only flag as mismatch if damage types are in different categories
            if not _damage_types_related(img.damage_type, extraction.claimed_damage_type):
                flags.add("claim_mismatch")
                break

    # ── If user has history risk, add manual review ─────────────────────

    if "user_history_risk" in flags:
        # High-risk users with any other flag should get manual review
        other_flags = flags - {"user_history_risk", "manual_review_required", "none"}
        if other_flags:
            flags.add("manual_review_required")

    # ── Finalize ────────────────────────────────────────────────────────

    # Remove "none" if other flags exist
    flags.discard("none")
    if not flags:
        flags.add("none")

    return RiskProfile(
        risk_flags=sorted(flags),
        user_history_summary=" ".join(s for s in summary_parts if s),
        rejection_rate=rejection_rate,
    )


def _damage_types_related(type_a: str, type_b: str) -> bool:
    """Check if two damage types are closely related (not a mismatch)."""
    related_groups = [
        {"dent", "scratch"},  # Both are surface damage
        {"crack", "glass_shatter"},  # Both involve breaking
        {"broken_part", "missing_part"},  # Both involve parts
        {"torn_packaging", "crushed_packaging"},  # Both are package exterior
        {"water_damage", "stain"},  # Both are surface discoloration
    ]
    for group in related_groups:
        if type_a in group and type_b in group:
            return True
    return False
