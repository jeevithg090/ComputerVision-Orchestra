"""Shared fixtures and builders for the test suite (no network, no API key)."""

from __future__ import annotations

from typing import Any

import pytest

from models import (
    ClaimExtraction,
    ClaimInput,
    EvidenceCheck,
    ImageAnalysisResult,
    RiskProfile,
    SingleImageAnalysis,
)


def make_claim(
    claim_object: str = "car",
    user_id: str = "test_user",
    image_paths: str = "images/sample/case_001/img_1.jpg",
    user_claim: str = "Customer: There is a dent on my rear bumper.",
) -> ClaimInput:
    return ClaimInput(
        user_id=user_id, image_paths=image_paths, user_claim=user_claim, claim_object=claim_object
    )


def make_extraction(
    damage: str = "dent", part: str = "rear_bumper", signals: list[str] | None = None
) -> ClaimExtraction:
    return ClaimExtraction(
        claimed_damage_type=damage,
        claimed_object_part=part,
        claim_summary=f"Customer claims {damage} on {part}.",
        adversarial_signals=signals or [],
        language_detected="en",
    )


def make_image(image_id: str = "img_1", **overrides: Any) -> SingleImageAnalysis:
    """A clean image showing the claimed damage on the claimed part; override as needed."""
    fields: dict[str, Any] = {
        "image_id": image_id,
        "object_category": "car",
        "object_visible": "car",
        "is_expected_object": True,
        "object_matches_claim": True,
        "part_visible": "rear_bumper",
        "part_matches_claim": True,
        "damage_visible": True,
        "claimed_damage_visible": True,
        "other_damage_visible": False,
        "damage_type": "dent",
        "visible_damage_type": "dent",
        "damage_confidence": 0.9,
        "damage_severity": "medium",
        "quality_issues": [],
        "text_instruction_detected": False,
        "text_instruction_content": "",
        "appears_original": True,
        "relevance_to_claim": "relevant",
        "support_role": "supports_claim",
    }
    fields.update(overrides)
    return SingleImageAnalysis(**fields)


def make_analysis(*images: SingleImageAnalysis) -> ImageAnalysisResult:
    return ImageAnalysisResult(images=list(images))


def make_evidence(met: bool = True) -> EvidenceCheck:
    return EvidenceCheck(evidence_standard_met=met, evidence_standard_met_reason="Test evidence.")


def make_risk(*flags: str) -> RiskProfile:
    return RiskProfile(risk_flags=list(flags) or ["none"])


class FakeClient:
    """Stand-in for LLMClient that returns canned structured outputs and records calls."""

    def __init__(
        self,
        text: dict | None = None,
        vision: dict | None = None,
        reasoning: dict | None = None,
    ) -> None:
        self.text = text or {}
        self.vision = vision or {"images": []}
        self.reasoning = reasoning or {}
        self.calls: list[tuple[str, dict]] = []

    def call_text(self, **kwargs: Any) -> dict:
        self.calls.append(("text", kwargs))
        return self.text

    def call_vision(self, **kwargs: Any) -> dict:
        self.calls.append(("vision", kwargs))
        return self.vision

    def call_reasoning(self, **kwargs: Any) -> dict:
        self.calls.append(("reasoning", kwargs))
        return self.reasoning


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()
