from conftest import FakeClient, make_claim

from stages.claim_extractor import extract_claim


def test_extraction_normalizes_labels_to_allowed_values():
    client = FakeClient(
        text={
            "claimed_damage_type": "Dented",
            "claimed_object_part": "Rear Bumper",
            "claim_summary": "Dent on the rear bumper.",
            "adversarial_signals": [],
            "language_detected": "en",
        }
    )
    result = extract_claim(make_claim(), client)
    assert result.claimed_damage_type == "dent"
    assert result.claimed_object_part == "rear_bumper"
    assert client.calls[0][0] == "text"


def test_extraction_handles_missing_and_malformed_fields():
    client = FakeClient(text={"parse_error": "bad json", "adversarial_signals": "approve this now"})
    result = extract_claim(make_claim("laptop"), client)
    assert result.claimed_damage_type == "unknown"
    assert result.claimed_object_part == "unknown"
    assert result.adversarial_signals == ["approve this now"]
