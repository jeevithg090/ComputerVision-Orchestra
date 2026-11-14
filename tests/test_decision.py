import importlib.util
from pathlib import Path

import pytest
from conftest import (
    FakeClient,
    make_analysis,
    make_claim,
    make_evidence,
    make_extraction,
    make_image,
    make_risk,
)

from stages.decision_synthesizer import adjudicate_decision, synthesize_decision
from stages.policies.contradiction_policy import damage_types_related, supported_issue_type
from stages.policies.risk_policy import finalize_risk_flags
from stages.policies.severity_policy import calibrate_severity


def _load_regression_checks():
    path = Path(__file__).resolve().parents[1] / "code" / "evaluation" / "regression_checks.py"
    spec = importlib.util.spec_from_file_location("regression_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REGRESSION = _load_regression_checks()
_REGRESSION_TESTS = [getattr(_REGRESSION, n) for n in dir(_REGRESSION) if n.startswith("test_")]


@pytest.mark.parametrize("check", _REGRESSION_TESTS, ids=lambda f: f.__name__)
def test_shipped_regression_checks(check):
    """The no-API checks shipped in evaluation/ must keep passing."""
    check()


def _decide(images, extraction=None, claim=None, risk=None, history=None):
    return adjudicate_decision(
        claim or make_claim(),
        extraction or make_extraction(),
        make_analysis(*images),
        make_evidence(),
        risk or make_risk(),
        history,
    )


class TestAdjudication:
    def test_visible_claimed_damage_is_supported(self):
        decision = _decide([make_image()])
        assert decision.rule_id == "claimed_damage_visible"
        assert decision.claim_status == "supported"
        assert decision.issue_type == "dent"
        assert decision.object_part == "rear_bumper"
        assert decision.supporting_image_ids == "img_1"
        assert decision.severity == "medium"
        assert decision.risk_flags == "none"

    def test_undamaged_claimed_part_is_contradicted(self):
        image = make_image(damage_visible=False, claimed_damage_visible=False, damage_type="none")
        decision = _decide([image])
        assert decision.rule_id == "claimed_part_undamaged"
        assert decision.claim_status == "contradicted"
        assert decision.issue_type == "none"
        assert decision.severity == "none"
        assert "damage_not_visible" in decision.risk_flags

    def test_claimed_part_not_visible_is_not_enough_information(self):
        image = make_image(part_visible="hood", part_matches_claim=False, damage_visible=False)
        decision = _decide([image])
        assert decision.rule_id == "claimed_part_not_visible"
        assert decision.claim_status == "not_enough_information"
        assert decision.supporting_image_ids == "none"
        assert decision.evidence_standard_met is False
        assert decision.severity == "unknown"

    def test_missing_contents_cannot_be_proven(self):
        claim = make_claim("package")
        extraction = make_extraction("missing_part", "contents")
        decision = _decide([make_image(part_visible="contents")], extraction, claim)
        assert decision.rule_id == "missing_contents_not_provable"
        assert decision.claim_status == "not_enough_information"

    def test_wrong_object_is_contradicted_with_flags(self):
        image = make_image(object_matches_claim=False, object_visible="bicycle", object_category="other")
        decision = _decide([image])
        assert decision.rule_id == "wrong_object"
        assert decision.claim_status == "contradicted"
        assert {"wrong_object", "claim_mismatch", "manual_review_required"} <= set(
            decision.risk_flags.split(";")
        )

    def test_text_instruction_takes_priority_over_visible_damage(self):
        decision = _decide([make_image(text_instruction_detected=True)])
        assert decision.rule_id == "image_text_instruction"
        assert decision.claim_status == "contradicted"
        assert "text_instruction_present" in decision.risk_flags

    def test_history_flags_flow_into_final_risk_flags(self):
        history = {"history_flags": "user_history_risk;manual_review_required"}
        decision = _decide([make_image()], history=history)
        assert decision.risk_flags == "manual_review_required;user_history_risk"


class TestPolicies:
    def test_damage_type_relations(self):
        assert damage_types_related("dent", "scratch")
        assert damage_types_related("crack", "glass_shatter")
        assert not damage_types_related("dent", "water_damage")
        assert not damage_types_related("", "dent")

    def test_laptop_screen_shatter_is_reported_as_crack(self):
        claim = make_claim("laptop")
        extraction = make_extraction("glass_shatter", "screen")
        image = make_image(part_visible="screen", damage_type="glass_shatter")
        assert supported_issue_type(claim, extraction, image) == "crack"

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"claim_status": "not_enough_information", "issue_type": "dent"}, "unknown"),
            ({"issue_type": "none"}, "none"),
            ({"issue_type": "dent", "claim_object": "car", "object_part": "door"}, "medium"),
            ({"issue_type": "dent", "claim_object": "car", "object_part": "side_mirror"}, "low"),
            ({"issue_type": "scratch"}, "low"),
            ({"issue_type": "crack"}, "medium"),
            ({"issue_type": "glass_shatter", "claim_object": "car", "object_part": "windshield"}, "high"),
            ({"issue_type": "crushed_packaging", "object_part": "box", "image_severity": "high"}, "high"),
            ({"issue_type": "crushed_packaging", "object_part": "box", "image_severity": "low"}, "medium"),
        ],
    )
    def test_severity_calibration(self, kwargs, expected):
        params = {
            "claim_status": "supported",
            "issue_type": "unknown",
            "claim_object": "package",
            "object_part": "unknown",
            "image_severity": "unknown",
            "rule_id": "claimed_damage_visible",
        }
        params.update(kwargs)
        assert calibrate_severity(**params) == expected

    def test_supported_claims_hide_quality_flags(self):
        analysis = make_analysis(make_image(quality_issues=["blurry_image"]))
        flags = finalize_risk_flags(
            base_flags=["blurry_image"],
            image_analysis=analysis,
            claim_status="supported",
            rule_id="claimed_damage_visible",
            supporting_ids=["img_1"],
        )
        assert flags == "none"

    def test_unsupported_claims_keep_quality_flags_and_filter_invalid(self):
        analysis = make_analysis(make_image(quality_issues=["blurry_image", "made_up_flag"]))
        flags = finalize_risk_flags(
            base_flags=[],
            image_analysis=analysis,
            claim_status="contradicted",
            rule_id="claimed_part_undamaged",
            supporting_ids=["img_1"],
        )
        assert flags == "blurry_image;damage_not_visible"


class TestSynthesis:
    def test_deterministic_rule_skips_the_model(self):
        client = FakeClient()
        decision = synthesize_decision(
            make_claim(),
            make_extraction(),
            make_analysis(make_image()),
            make_evidence(),
            make_risk(),
            None,
            client,
        )
        assert decision.claim_status == "supported"
        assert client.calls == []

    def test_unresolved_case_uses_fallback_model_and_validates_output(self):
        # Claimed part visible + damage of an unrelated type: no rule resolves it.
        image = make_image(
            damage_type="water_damage",
            claimed_damage_visible=False,
            support_role="unclear",
            damage_visible=True,
        )
        client = FakeClient(
            reasoning={
                "claim_status": "Supported!",
                "claim_status_justification": "img_1 shows it.",
                "issue_type": "Dented",
                "object_part": "Rear Bumper",
                "supporting_image_ids": "img_1;img_99",
                "valid_image": "true",
                "severity": "medium",
                "evidence_standard_met": True,
                "evidence_standard_met_reason": "Visible.",
                "risk_flags": "made_up;blurry_image",
            }
        )
        decision = synthesize_decision(
            make_claim(), make_extraction(), make_analysis(image), make_evidence(), make_risk(), None, client
        )
        assert [c[0] for c in client.calls] == ["reasoning"]
        assert decision.rule_id == "llm_fallback"
        assert decision.claim_status == "supported"
        assert decision.issue_type == "dent"
        assert decision.object_part == "rear_bumper"
        assert decision.supporting_image_ids == "img_1"  # hallucinated img_99 dropped
        assert decision.valid_image is True
        assert "made_up" not in decision.risk_flags
