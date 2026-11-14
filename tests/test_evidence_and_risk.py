import pytest
from conftest import make_analysis, make_claim, make_extraction, make_image

from stages import risk_assessor
from stages.evidence_matcher import match_evidence
from stages.risk_assessor import assess_risk


@pytest.fixture
def history(monkeypatch):
    data = {
        "clean_user": {
            "past_claim_count": 2,
            "rejected_claim": 0,
            "manual_review_claim": 0,
            "last_90_days_claim_count": 1,
            "history_flags": "none",
            "history_summary": "Low risk.",
        },
        "risky_user": {
            "past_claim_count": 6,
            "rejected_claim": 3,
            "manual_review_claim": 1,
            "last_90_days_claim_count": 2,
            "history_flags": "none",
            "history_summary": "Several rejected claims.",
        },
    }
    monkeypatch.setattr(risk_assessor, "_USER_HISTORY", data)
    return data


class TestEvidenceMatcher:
    def test_visible_claimed_part_meets_standard(self):
        result = match_evidence(make_claim(), make_extraction(), make_analysis(make_image()))
        assert result.evidence_standard_met is True
        assert "REQ_GENERAL_OBJECT_PART" in result.applicable_requirements
        assert "REQ_CAR_BODY_PANEL" in result.applicable_requirements

    def test_multi_image_requirement_only_for_multiple_images(self):
        single = match_evidence(make_claim(), make_extraction(), make_analysis(make_image()))
        multi = match_evidence(
            make_claim(), make_extraction(), make_analysis(make_image("img_1"), make_image("img_2"))
        )
        assert "REQ_GENERAL_MULTI_IMAGE" not in single.applicable_requirements
        assert "REQ_GENERAL_MULTI_IMAGE" in multi.applicable_requirements

    def test_blocked_images_without_claimed_part_fail(self):
        image = make_image(
            part_visible="door", part_matches_claim=False, quality_issues=["cropped_or_obstructed"]
        )
        result = match_evidence(make_claim(), make_extraction(), make_analysis(image))
        assert result.evidence_standard_met is False

    def test_unidentifiable_object_fails(self):
        image = make_image(object_matches_claim=False, object_visible="unknown")
        result = match_evidence(make_claim(), make_extraction(), make_analysis(image))
        assert result.evidence_standard_met is False

    def test_reason_lists_visible_parts_in_stable_order(self):
        images = [
            make_image("img_1", part_visible="hood", part_matches_claim=False),
            make_image("img_2", part_visible="door", part_matches_claim=False),
        ]
        result = match_evidence(make_claim(), make_extraction(), make_analysis(*images))
        assert "door, hood" in result.evidence_standard_met_reason


class TestRiskAssessor:
    def test_clean_claim_has_no_flags(self, history):
        result = assess_risk(make_claim(user_id="clean_user"), make_extraction(), make_analysis(make_image()))
        assert result.risk_flags == ["none"]

    def test_high_rejection_rate_flags_user_history(self, history):
        result = assess_risk(make_claim(user_id="risky_user"), make_extraction(), make_analysis(make_image()))
        assert "user_history_risk" in result.risk_flags
        assert result.rejection_rate == pytest.approx(0.5)

    def test_history_risk_plus_evidence_risk_requires_manual_review(self, history):
        image = make_image(quality_issues=["blurry_image"])
        result = assess_risk(make_claim(user_id="risky_user"), make_extraction(), make_analysis(image))
        assert {"user_history_risk", "blurry_image", "manual_review_required"} <= set(result.risk_flags)

    def test_image_level_flags(self, history):
        image = make_image(object_matches_claim=False, text_instruction_detected=True, appears_original=False)
        result = assess_risk(make_claim(user_id="clean_user"), make_extraction(), make_analysis(image))
        assert {"wrong_object", "text_instruction_present", "non_original_image"} <= set(result.risk_flags)

    def test_damage_on_other_part_is_a_mismatch(self, history):
        image = make_image(part_visible="door", part_matches_claim=False)
        result = assess_risk(make_claim(user_id="clean_user"), make_extraction(), make_analysis(image))
        assert {"claim_mismatch", "wrong_object_part"} <= set(result.risk_flags)

    def test_adversarial_conversation_requires_manual_review(self, history):
        result = assess_risk(
            make_claim(user_id="clean_user"),
            make_extraction(signals=["asks reviewer to approve"]),
            make_analysis(make_image()),
        )
        assert "manual_review_required" in result.risk_flags

    def test_unknown_user_is_tolerated(self, history):
        result = assess_risk(make_claim(user_id="nobody"), make_extraction(), make_analysis(make_image()))
        assert result.user_history_summary == "No user history available."
