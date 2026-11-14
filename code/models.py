"""
Pydantic data models for every stage of the claim verification pipeline.

Each model enforces type constraints and allowed values at the boundary,
making invalid state unrepresentable deeper in the codebase.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Stage 0: Raw CSV Input ──────────────────────────────────────────────────


class ClaimInput(BaseModel):
    """One row from claims.csv (input only, no labels)."""

    user_id: str
    image_paths: str  # semicolon-separated
    user_claim: str  # raw conversation transcript
    claim_object: Literal["car", "laptop", "package"]

    @property
    def image_path_list(self) -> list[str]:
        return [p.strip() for p in self.image_paths.split(";") if p.strip()]

    @property
    def image_ids(self) -> list[str]:
        """Extract image IDs (filename without extension)."""
        from pathlib import Path as _P

        return [_P(p).stem for p in self.image_path_list]


# ── Stage 1: Claim Extraction Output ───────────────────────────────────────


class ClaimExtraction(BaseModel):
    """Structured extraction from the user_claim conversation."""

    claimed_damage_type: str = Field(
        description="Damage type the customer is claiming (from allowed issue_type list)"
    )
    claimed_object_part: str = Field(description="Specific part of the object the customer says is damaged")
    claim_summary: str = Field(description="One-sentence summary of the customer's claim")
    adversarial_signals: list[str] = Field(
        default_factory=list, description="Any manipulation attempts detected in the conversation"
    )
    language_detected: str = Field(default="en", description="Primary language of the conversation")


# ── Stage 2: Image Analysis Output ─────────────────────────────────────────


class SingleImageAnalysis(BaseModel):
    """Analysis result for one image."""

    image_id: str
    object_category: str = Field(
        default="unknown", description="Normalized visible object category: car, laptop, package, or other"
    )
    object_visible: str = Field(description="What object is visible (car, laptop, package, other)")
    is_expected_object: bool = Field(
        default=True, description="Whether the visible object is the expected claim object"
    )
    object_matches_claim: bool = Field(description="Whether visible object matches the claim_object")
    part_visible: str = Field(description="Specific part visible in the image")
    part_matches_claim: bool = Field(default=False, description="Whether the claimed part is visible")
    damage_visible: bool = Field(description="Whether any damage is visible")
    claimed_damage_visible: bool = Field(
        default=False, description="Whether damage matching the claimed part/type is visible"
    )
    other_damage_visible: bool = Field(
        default=False, description="Whether damage is visible but not on the claimed part/type"
    )
    damage_type: str = Field(description="Type of visible damage (from allowed issue_type list)")
    visible_damage_type: str = Field(default="unknown", description="Richer alias for visible damage type")
    damage_confidence: float = Field(
        default=1.0, description="Confidence score for damage_type, between 0.0 and 1.0"
    )
    damage_severity: str = Field(description="Severity of visible damage: none, low, medium, high, unknown")
    quality_issues: list[str] = Field(
        default_factory=list, description="Image quality problems: blurry_image, cropped_or_obstructed, etc."
    )
    text_instruction_detected: bool = Field(
        default=False, description="Whether text instructions/notes are present in the image"
    )
    instructional_text: str = Field(
        default="", description="Manipulative or reviewer-directed text detected in the image"
    )
    text_instruction_content: str = Field(
        default="", description="Content of detected text instruction, if any"
    )
    appears_original: bool = Field(
        default=True, description="Whether the image appears to be an original photograph"
    )
    image_authenticity: str = Field(
        default="original_photo",
        description="original_photo, screenshot, printout, digitally_altered, or unknown",
    )
    support_role: str = Field(
        default="unclear", description="supports_claim, contradicts_claim, insufficient, mismatch, or unclear"
    )
    relevance_to_claim: str = Field(
        default="relevant",
        description="How relevant this image is: relevant, partially_relevant, not_relevant",
    )


class ImageAnalysisResult(BaseModel):
    """Combined analysis of all images for a single claim."""

    images: list[SingleImageAnalysis]


# ── Stage 3: Evidence Matching Output ───────────────────────────────────────


class EvidenceCheck(BaseModel):
    """Result of checking image evidence against requirements."""

    evidence_standard_met: bool
    evidence_standard_met_reason: str
    applicable_requirements: list[str] = Field(
        default_factory=list, description="IDs of evidence requirements that were checked"
    )


# ── Stage 4: Risk Assessment Output ────────────────────────────────────────


class RiskProfile(BaseModel):
    """Aggregated risk flags from user history + image analysis."""

    risk_flags: list[str] = Field(
        default_factory=lambda: ["none"], description="List of risk flags; ['none'] if no risks"
    )
    user_history_summary: str = Field(
        default="", description="Human-readable summary of relevant user history"
    )
    rejection_rate: float = Field(default=0.0, description="Historical claim rejection rate")


# ── Stage 5: Final Decision Output ─────────────────────────────────────────


class ClaimDecision(BaseModel):
    """Final structured decision — maps directly to output.csv columns."""

    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str = Field(description="Semicolon-separated risk flags, or 'none'")
    issue_type: str = Field(description="Visible issue type from allowed list")
    object_part: str = Field(description="Relevant object part from allowed list")
    claim_status: str = Field(description="supported, contradicted, or not_enough_information")
    claim_status_justification: str = Field(description="Image-grounded explanation with image IDs")
    supporting_image_ids: str = Field(description="Semicolon-separated image IDs, or 'none'")
    valid_image: bool = Field(description="Whether image set is usable for automated review")
    severity: str = Field(description="none, low, medium, high, or unknown")
    rule_id: str = Field(
        default="unknown", description="Internal adjudication rule; not written to output.csv"
    )


class AdjudicatedDecision(BaseModel):
    """Deterministic decision before optional LLM wording cleanup."""

    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: bool
    severity: str
    rule_id: str = Field(default="llm_fallback", description="Deterministic rule that selected this decision")


# ── Full Pipeline Result (per claim) ───────────────────────────────────────


class ClaimResult(BaseModel):
    """All intermediate + final results for one claim."""

    input: ClaimInput
    extraction: ClaimExtraction
    image_analysis: ImageAnalysisResult
    evidence_check: EvidenceCheck
    risk_profile: RiskProfile
    decision: ClaimDecision
