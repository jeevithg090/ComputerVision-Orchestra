"""
Stage 5 — Decision Synthesis

The final decision-maker. Receives gathered evidence from stages 1-4,
applies deterministic calibration policies first, and uses the configured
LLM only as a fallback for unresolved cases.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from models import (
    ClaimInput,
    ClaimExtraction,
    ImageAnalysisResult,
    EvidenceCheck,
    RiskProfile,
    ClaimDecision,
    AdjudicatedDecision,
)
from llm_client import LLMClient
from config import (
    DECISION_SYNTHESIS_MODEL,
    SYNTHESIS_REASONING_EFFORT,
    CLAIM_STATUS_VALUES,
    ISSUE_TYPE_VALUES,
    SEVERITY_VALUES,
    OBJECT_PARTS,
    RISK_FLAG_VALUES,
    load_evidence_requirements,
)
from prompts import (
    DECISION_SYNTHESIS_DEVELOPER,
    build_decision_synthesis_user_prompt,
)
from stages.policies.contradiction_policy import (
    damage_types_related,
    mismatch_issue_type,
    should_support_claim,
    supported_issue_type,
    vague_physical_claim_is_contradicted,
)
from stages.policies.risk_policy import finalize_risk_flags
from stages.policies.severity_policy import calibrate_severity


# Cache evidence requirements text
_REQUIREMENTS_TEXT = None


def _get_requirements_text(claim_object: str, issue_type: str) -> str:
    """Build a human-readable string of applicable evidence requirements."""
    global _REQUIREMENTS_TEXT
    reqs = load_evidence_requirements()
    lines = []
    for req in reqs:
        obj = req["claim_object"]
        if obj == "all" or obj == claim_object:
            lines.append(f"  [{req['requirement_id']}] ({req['applies_to']}): {req['minimum_image_evidence']}")
    return "\n".join(lines) if lines else "  No specific requirements found."


def synthesize_decision(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image_analysis: ImageAnalysisResult,
    evidence_check: EvidenceCheck,
    risk_profile: RiskProfile,
    user_history: Optional[dict],
    client: LLMClient,
) -> ClaimDecision:
    """
    Run Stage 5: Produce the final claim decision.

    Compiles all signals from stages 1-4 into a comprehensive prompt
    and uses deterministic policies before any model fallback.

    Args:
        claim: Raw claim input.
        extraction: Stage 1 output.
        image_analysis: Stage 2 output.
        evidence_check: Stage 3 output.
        risk_profile: Stage 4 output.
        user_history: Raw user history dict, or None.
        client: LLM client for API calls.

    Returns:
        ClaimDecision with all output fields.
    """
    adjudicated = adjudicate_decision(
        claim, extraction, image_analysis, evidence_check, risk_profile, user_history
    )
    if adjudicated.rule_id != "llm_fallback":
        return ClaimDecision(**adjudicated.model_dump())

    # Build the evidence requirements text
    requirements_text = _get_requirements_text(
        claim.claim_object,
        extraction.claimed_damage_type,
    )

    sanitized_images = []
    has_text_instruction = False
    for img in image_analysis.images:
        if img.text_instruction_detected:
            has_text_instruction = True
        img_dict = img.model_dump()
        img_dict.pop("text_instruction_detected", None)
        img_dict.pop("text_instruction_content", None)
        sanitized_images.append(img_dict)

    # Build the user prompt with all evidence
    user_prompt = build_decision_synthesis_user_prompt(
        claim_object=claim.claim_object,
        claim_extraction=extraction.model_dump(),
        image_analysis={"images": sanitized_images},
        evidence_check=evidence_check.model_dump(),
        risk_profile=risk_profile.model_dump(),
        evidence_requirements_text=requirements_text,
        user_history=user_history,
    )

    # Call configured fallback model for unresolved cases.
    result = client.call_reasoning(
        model=DECISION_SYNTHESIS_MODEL,
        developer_message=DECISION_SYNTHESIS_DEVELOPER,
        user_message=user_prompt,
        reasoning_effort=SYNTHESIS_REASONING_EFFORT,
    )

    # Validate and build the final decision
    return _build_decision(
        result,
        claim,
        extraction,
        evidence_check,
        risk_profile,
        image_analysis,
        user_history,
        has_text_instruction,
    )


def adjudicate_decision(
    claim: ClaimInput,
    extraction: ClaimExtraction,
    image_analysis: ImageAnalysisResult,
    evidence_check: EvidenceCheck,
    risk_profile: RiskProfile,
    user_history: Optional[dict] = None,
) -> AdjudicatedDecision:
    """Apply deterministic high-confidence adjudication rules."""
    images = image_analysis.images
    flags = _collect_flags(risk_profile, image_analysis)
    claimed_type = extraction.claimed_damage_type
    claimed_part = extraction.claimed_object_part

    object_images = [img for img in images if img.object_matches_claim]
    claimed_part_images = [
        img for img in object_images
        if img.part_matches_claim or img.part_visible == claimed_part
    ]
    support_images = [
        img for img in claimed_part_images
        if should_support_claim(claim, extraction, img)
    ]
    undamaged_claimed_part = [
        img for img in claimed_part_images
        if not img.damage_visible and img.relevance_to_claim != "not_relevant"
    ]
    mismatch_images = [
        img for img in object_images
        if img.damage_visible and not (img.part_matches_claim or img.part_visible == claimed_part)
    ]
    text_images = [img for img in images if img.text_instruction_detected]
    blocking_quality = {"wrong_angle", "cropped_or_obstructed", "blurry_image"}

    if images and not object_images and any(img.object_visible != "unknown" for img in images):
        flags.update({"wrong_object", "claim_mismatch", "manual_review_required"})
        return _adjudicated(
            rule_id="wrong_object",
            claim_status="contradicted",
            issue_type="unknown",
            object_part="unknown",
            severity="unknown",
            supporting_ids=[img.image_id for img in images],
            evidence_check=evidence_check,
            flags=flags,
            image_analysis=image_analysis,
            user_history=user_history,
            justification=(
                f"The submitted image evidence does not show a {claim.claim_object}; "
                "it shows a different object, so the claim is contradicted."
            ),
            valid_image=True,
        )

    if text_images and object_images:
        flags.update({"text_instruction_present", "manual_review_required"})
        return _adjudicated(
            rule_id="image_text_instruction",
            claim_status="contradicted",
            issue_type="none",
            object_part=claimed_part if claimed_part in OBJECT_PARTS.get(claim.claim_object, set()) else "unknown",
            severity="none",
            supporting_ids=[img.image_id for img in images],
            evidence_check=evidence_check,
            flags=flags,
            image_analysis=image_analysis,
            user_history=user_history,
            justification=(
                "The submitted evidence contains reviewer-directed text instructions "
                "attached to the claim image, so it is treated as contradictory evidence."
            ),
            valid_image=True,
        )

    is_missing_contents = claimed_type == "missing_part" and claimed_part in {"contents", "item"}
    if is_missing_contents:
        return _adjudicated(
            rule_id="missing_contents_not_provable",
            claim_status="not_enough_information",
            issue_type="unknown",
            object_part=claimed_part,
            severity="unknown",
            supporting_ids=[],
            evidence_check=EvidenceCheck(
                evidence_standard_met=False,
                evidence_standard_met_reason=(
                    "Images alone do not establish whether the expected contents are missing."
                ),
                applicable_requirements=evidence_check.applicable_requirements,
            ),
            flags=flags,
            image_analysis=image_analysis,
            user_history=user_history,
            justification=(
                "The images do not prove what contents should be present, so the missing-content "
                "claim cannot be verified or contradicted from the visual evidence."
            ),
            valid_image=True,
        )

    if support_images:
        best = _best_damage_image(support_images)
        issue_type = supported_issue_type(claim, extraction, best)
        return _adjudicated(
            rule_id="claimed_damage_visible",
            claim_status="supported",
            issue_type=issue_type,
            object_part=claimed_part if claimed_part != "unknown" else best.part_visible,
            severity=calibrate_severity(
                claim_status="supported",
                issue_type=issue_type,
                claim_object=claim.claim_object,
                object_part=claimed_part if claimed_part != "unknown" else best.part_visible,
                image_severity=best.damage_severity,
                rule_id="claimed_damage_visible",
                claimed_issue_type=claimed_type,
            ),
            supporting_ids=[img.image_id for img in support_images],
            evidence_check=evidence_check,
            flags=flags,
            image_analysis=image_analysis,
            user_history=user_history,
            justification=(
                f"The claimed {claimed_part} is visible and image "
                f"{best.image_id} shows damage consistent with the claim."
            ),
            valid_image=True,
        )

    if mismatch_images:
        best = _best_damage_image(mismatch_images)
        flags.update({"claim_mismatch", "wrong_object_part", "manual_review_required"})
        issue_type = mismatch_issue_type(extraction, best)
        return _adjudicated(
            rule_id="wrong_part_damage",
            claim_status="contradicted",
            issue_type=issue_type,
            object_part=best.part_visible if best.part_visible != "unknown" else claimed_part,
            severity=calibrate_severity(
                claim_status="contradicted",
                issue_type=issue_type,
                claim_object=claim.claim_object,
                object_part=best.part_visible if best.part_visible != "unknown" else claimed_part,
                image_severity=best.damage_severity,
                rule_id="wrong_part_damage",
                claimed_issue_type=claimed_type,
            ),
            supporting_ids=[img.image_id for img in mismatch_images],
            evidence_check=evidence_check,
            flags=flags,
            image_analysis=image_analysis,
            user_history=user_history,
            justification=(
                f"The submitted image shows damage on {best.part_visible}, not on the claimed "
                f"{claimed_part}, so the specific claim is contradicted."
            ),
            valid_image=True,
        )

    if undamaged_claimed_part:
        flags.add("damage_not_visible")
        return _adjudicated(
            rule_id="claimed_part_undamaged",
            claim_status="contradicted",
            issue_type="none",
            object_part=claimed_part,
            severity="none",
            supporting_ids=[img.image_id for img in undamaged_claimed_part],
            evidence_check=evidence_check,
            flags=flags,
            image_analysis=image_analysis,
            user_history=user_history,
            justification=(
                f"The claimed {claimed_part} is visible in the submitted image evidence, "
                "but no claimed damage is visible."
            ),
            valid_image=True,
        )

    if object_images and not claimed_part_images:
        flags.add("wrong_object_part")
        if all(set(img.quality_issues) & blocking_quality for img in object_images):
            flags.add("manual_review_required")
        return _adjudicated(
            rule_id="claimed_part_not_visible",
            claim_status="not_enough_information",
            issue_type="unknown",
            object_part=claimed_part,
            severity="unknown",
            supporting_ids=[],
            evidence_check=EvidenceCheck(
                evidence_standard_met=False,
                evidence_standard_met_reason=f"No submitted image clearly shows the claimed {claimed_part}.",
                applicable_requirements=evidence_check.applicable_requirements,
            ),
            flags=flags,
            image_analysis=image_analysis,
            user_history=user_history,
            justification=(
                f"The images do not clearly show the claimed {claimed_part}, so the claim "
                "cannot be verified or contradicted."
            ),
            valid_image=True,
        )

    vague_physical_images = [
        img for img in claimed_part_images
        if vague_physical_claim_is_contradicted(claim, extraction, img)
    ]
    if vague_physical_images:
        flags.add("damage_not_visible")
        best = vague_physical_images[0]
        return _adjudicated(
            rule_id="vague_laptop_physical_damage",
            claim_status="contradicted",
            issue_type="none",
            object_part=claimed_part,
            severity="none",
            supporting_ids=[img.image_id for img in vague_physical_images],
            evidence_check=evidence_check,
            flags=flags,
            image_analysis=image_analysis,
            user_history=user_history,
            justification=(
                f"The claimed {claimed_part} is visible in image {best.image_id}, "
                "but no clear allowed physical damage type is visible."
            ),
            valid_image=True,
        )

    return AdjudicatedDecision(
        evidence_standard_met=evidence_check.evidence_standard_met,
        evidence_standard_met_reason=evidence_check.evidence_standard_met_reason,
        risk_flags=_format_flags(flags),
        issue_type="unknown",
        object_part=claimed_part,
        claim_status="not_enough_information",
        claim_status_justification="No deterministic rule resolved the claim.",
        supporting_image_ids="none",
        valid_image=True,
        severity="unknown",
        rule_id="llm_fallback",
    )


def _collect_flags(risk_profile: RiskProfile, image_analysis: ImageAnalysisResult) -> set[str]:
    flags = {f for f in risk_profile.risk_flags if f and f != "none"}
    for img in image_analysis.images:
        if not img.object_matches_claim:
            flags.add("wrong_object")
        if img.text_instruction_detected:
            flags.add("text_instruction_present")
        if not img.appears_original:
            flags.add("non_original_image")
        for issue in img.quality_issues:
            if issue in RISK_FLAG_VALUES and issue != "none":
                flags.add(issue)
    return flags


def _format_flags(flags: set[str]) -> str:
    valid = sorted(f for f in flags if f in RISK_FLAG_VALUES and f != "none")
    return ";".join(valid) if valid else "none"


def _best_damage_image(images):
    return max(images, key=lambda img: (img.damage_confidence, img.damage_severity == "high"))


def _damage_types_related(type_a: str, type_b: str) -> bool:
    return damage_types_related(type_a, type_b)


def _calibrated_severity(issue_type: str, image_severity: str) -> str:
    return calibrate_severity(
        claim_status="supported",
        issue_type=issue_type,
        claim_object="unknown",
        object_part="unknown",
        image_severity=image_severity,
        rule_id="legacy",
    )


def _adjudicated(
    rule_id: str,
    claim_status: str,
    issue_type: str,
    object_part: str,
    severity: str,
    supporting_ids: List[str],
    evidence_check: EvidenceCheck,
    flags: set[str],
    image_analysis: ImageAnalysisResult,
    user_history: Optional[dict],
    justification: str,
    valid_image: bool,
) -> AdjudicatedDecision:
    if claim_status == "not_enough_information":
        supporting = "none"
    else:
        supporting = ";".join(dict.fromkeys(supporting_ids)) if supporting_ids else "none"
    risk_flags = finalize_risk_flags(
        base_flags=flags,
        image_analysis=image_analysis,
        claim_status=claim_status,
        rule_id=rule_id,
        supporting_ids=supporting.split(";") if supporting != "none" else [],
        user_history=user_history,
    )
    return AdjudicatedDecision(
        evidence_standard_met=evidence_check.evidence_standard_met,
        evidence_standard_met_reason=evidence_check.evidence_standard_met_reason,
        risk_flags=risk_flags,
        issue_type=issue_type if issue_type in ISSUE_TYPE_VALUES else "unknown",
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=supporting,
        valid_image=valid_image,
        severity=severity if severity in SEVERITY_VALUES else "unknown",
        rule_id=rule_id,
    )


def _build_decision(
    raw: dict,
    claim: ClaimInput,
    extraction: ClaimExtraction,
    evidence_check: EvidenceCheck,
    risk_profile: RiskProfile,
    image_analysis: ImageAnalysisResult,
    user_history: Optional[dict] = None,
    has_text_instruction: bool = False,
) -> ClaimDecision:
    """Validate and normalize the raw o3 response into a ClaimDecision."""

    # ── claim_status ────────────────────────────────────────────────────
    claim_status = raw.get("claim_status", "not_enough_information")
    if claim_status not in CLAIM_STATUS_VALUES:
        claim_status = _closest_status(claim_status)

    # ── CHANGE 2: FIX ALL MISSING-CONTENTS CLAIMS ──
    is_missing_claim = extraction.claimed_damage_type == "missing_part" and extraction.claimed_object_part in ["contents", "item"]
    if is_missing_claim:
        # Images generally cannot prove absence. Default to not_enough_information
        # unless we clearly see the expected item.
        expected_item_clearly_visible = False
        for img in image_analysis.images:
            if img.object_matches_claim and not img.damage_visible and img.relevance_to_claim != "not_relevant":
                expected_item_clearly_visible = True
        
        if expected_item_clearly_visible:
            claim_status = "contradicted"
        else:
            claim_status = "not_enough_information"

    # ── issue_type ──────────────────────────────────────────────────────
    issue_type = raw.get("issue_type", "unknown")
    if issue_type not in ISSUE_TYPE_VALUES:
        issue_type = _normalize_to_set(issue_type, ISSUE_TYPE_VALUES, "unknown")

    # ── object_part ─────────────────────────────────────────────────────
    object_part = raw.get("object_part", "unknown")
    valid_parts = OBJECT_PARTS.get(claim.claim_object, frozenset(["unknown"]))
    if object_part not in valid_parts:
        object_part = _normalize_to_set(object_part, valid_parts, "unknown")

    raw_severity = raw.get("severity", "unknown")
    severity = calibrate_severity(
        claim_status=claim_status,
        issue_type=issue_type,
        claim_object=claim.claim_object,
        object_part=object_part,
        image_severity=raw_severity,
        rule_id="llm_fallback",
        claimed_issue_type=extraction.claimed_damage_type,
    )

    # ── risk_flags ──────────────────────────────────────────────────────
    raw_flags = raw.get("risk_flags", "none")
    if isinstance(raw_flags, list):
        flags = raw_flags
    elif isinstance(raw_flags, str):
        flags = [f.strip() for f in raw_flags.split(";") if f.strip()]
    else:
        flags = ["none"]

    # Validate and calibrate each flag
    valid_flags = [f for f in flags if f in RISK_FLAG_VALUES]
    if has_text_instruction and "text_instruction_present" not in valid_flags:
        valid_flags.append("text_instruction_present")

    # ── supporting_image_ids ────────────────────────────────────────────
    supporting_ids = raw.get("supporting_image_ids", "none")
    if isinstance(supporting_ids, list):
        supporting_ids = ";".join(supporting_ids) if supporting_ids else "none"
    if not supporting_ids or supporting_ids.lower() == "none":
        if claim_status == "not_enough_information":
            supporting_ids = "none"
        # For supported/contradicted, the model should have provided IDs
        # If not, use all available image IDs
        elif not supporting_ids or supporting_ids.lower() == "none":
            all_ids = [img.image_id for img in []]
            supporting_ids = "none"

    risk_flags_str = finalize_risk_flags(
        base_flags=set(valid_flags) | set(risk_profile.risk_flags),
        image_analysis=image_analysis,
        claim_status=claim_status,
        rule_id="llm_fallback",
        supporting_ids=supporting_ids.split(";") if supporting_ids != "none" else [],
        user_history=user_history,
    )

    # ── valid_image ─────────────────────────────────────────────────────
    valid_image = raw.get("valid_image", True)
    if isinstance(valid_image, str):
        valid_image = valid_image.lower() in ("true", "yes", "1")

    # ── evidence_standard_met ───────────────────────────────────────────
    evidence_met = raw.get("evidence_standard_met", evidence_check.evidence_standard_met)
    if isinstance(evidence_met, str):
        evidence_met = evidence_met.lower() in ("true", "yes", "1")

    evidence_reason = raw.get(
        "evidence_standard_met_reason",
        evidence_check.evidence_standard_met_reason,
    )

    # ── justification ───────────────────────────────────────────────────
    justification = raw.get(
        "claim_status_justification",
        "Decision based on visual evidence analysis."
    )

    return ClaimDecision(
        evidence_standard_met=evidence_met,
        evidence_standard_met_reason=evidence_reason,
        risk_flags=risk_flags_str,
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=supporting_ids,
        valid_image=valid_image,
        severity=severity,
        rule_id="llm_fallback",
    )


def _closest_status(value: str) -> str:
    """Map a non-standard claim status to the closest valid one."""
    v = value.lower().strip().replace(" ", "_")
    if "support" in v:
        return "supported"
    if "contradict" in v:
        return "contradicted"
    return "not_enough_information"


def _normalize_to_set(value: str, allowed: frozenset[str], default: str) -> str:
    """Normalize a value to the closest match in an allowed set."""
    v = value.lower().strip().replace(" ", "_").replace("-", "_")
    if v in allowed:
        return v
    # Substring match
    for candidate in allowed:
        if candidate in v or v in candidate:
            return candidate
    return default
