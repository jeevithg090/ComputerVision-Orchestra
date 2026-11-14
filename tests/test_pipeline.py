import csv

from conftest import FakeClient, make_claim

from config import SAMPLE_CLAIMS_CSV, load_evidence_requirements, load_user_history
from pipeline import OUTPUT_COLUMNS, fallback_result, load_claims, process_claim, process_claims, write_output

GOOD_EXTRACTION = {
    "claimed_damage_type": "dent",
    "claimed_object_part": "rear_bumper",
    "claim_summary": "Dent on the rear bumper.",
    "adversarial_signals": [],
    "language_detected": "en",
}

GOOD_VISION = {
    "images": [
        {
            "image_id": "img_1",
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
            "quality_blockers": [],
            "text_instruction_detected": False,
            "text_instruction_content": "",
            "instructional_text": "",
            "appears_original": True,
            "image_authenticity": "original_photo",
            "relevance_to_claim": "relevant",
            "support_role": "supports_claim",
        }
    ]
}


def test_output_columns_match_problem_statement():
    assert OUTPUT_COLUMNS == [
        "user_id",
        "image_paths",
        "user_claim",
        "claim_object",
        "evidence_standard_met",
        "evidence_standard_met_reason",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "claim_status_justification",
        "supporting_image_ids",
        "valid_image",
        "severity",
    ]


def test_dataset_loaders():
    claims = load_claims(SAMPLE_CLAIMS_CSV)
    assert len(claims) == 20
    assert all(c.claim_object in {"car", "laptop", "package"} for c in claims)
    assert load_user_history()
    assert any(r["claim_object"] == "all" for r in load_evidence_requirements())


def test_end_to_end_claim_with_fake_model(capsys):
    client = FakeClient(text=GOOD_EXTRACTION, vision=GOOD_VISION)
    result = process_claim(make_claim(user_id="user_001"), load_user_history(), client, 0, 1)
    assert result.decision.claim_status == "supported"
    assert result.decision.supporting_image_ids == "img_1"
    assert [c[0] for c in client.calls] == ["text", "vision"]


def test_failing_claim_becomes_fallback_row(capsys):
    def boom(*_args, **_kwargs):
        raise RuntimeError("model exploded")

    results = process_claims([make_claim()], {}, FakeClient(), process=boom)
    assert len(results) == 1
    assert results[0].decision.claim_status == "not_enough_information"
    assert results[0].decision.risk_flags == "manual_review_required"


def test_write_output_schema(tmp_path, capsys):
    out = tmp_path / "nested" / "output.csv"
    write_output([fallback_result(make_claim())], out)
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == OUTPUT_COLUMNS
    assert rows[0]["valid_image"] == "false"
    assert rows[0]["evidence_standard_met"] == "false"


def test_cli_rejects_missing_input(capsys):
    import main

    assert main.main(["--input", "does-not-exist.csv"]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_runs_with_injected_client(monkeypatch, tmp_path, capsys):
    import main

    monkeypatch.setattr(
        main, "LLMClient", lambda tracker: FakeClient(text=GOOD_EXTRACTION, vision=GOOD_VISION)
    )
    out = tmp_path / "out.csv"
    assert main.main(["--input", str(SAMPLE_CLAIMS_CSV), "--output", str(out)]) == 0
    with open(out, encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 20
