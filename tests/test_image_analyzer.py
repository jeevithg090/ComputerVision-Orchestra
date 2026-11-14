from conftest import FakeClient, make_claim, make_extraction

from stages.image_analyzer import analyze_images, resolve_images

SAMPLE_TWO = "images/sample/case_002/img_1.jpg;images/sample/case_002/img_2.jpg"


def _raw(image_id: str, **overrides):
    raw = {
        "image_id": image_id,
        "object_category": "Car",
        "object_visible": "car",
        "is_expected_object": True,
        "object_matches_claim": True,
        "part_visible": "Rear Bumper",
        "part_matches_claim": True,
        "damage_visible": True,
        "claimed_damage_visible": True,
        "other_damage_visible": False,
        "damage_type": "dented",
        "visible_damage_type": "dent",
        "damage_confidence": 0.95,
        "damage_severity": "Moderate",
        "quality_issues": ["none"],
        "quality_blockers": "blurry_image",
        "text_instruction_detected": False,
        "text_instruction_content": "",
        "instructional_text": "",
        "appears_original": True,
        "image_authenticity": "original_photo",
        "relevance_to_claim": "relevant",
        "support_role": "supports_claim",
    }
    raw.update(overrides)
    return raw


def test_resolve_images_separates_missing_files():
    present, missing = resolve_images(
        ["img_1", "img_9"], ["images/sample/case_001/img_1.jpg", "images/sample/case_001/img_9.jpg"]
    )
    assert [image_id for image_id, _ in present] == ["img_1"]
    assert missing == ["img_9"]


def test_analysis_normalizes_model_output():
    client = FakeClient(vision={"images": [_raw("img_1")]})
    result = analyze_images(make_claim(), make_extraction(), client)
    img = result.images[0]
    assert img.part_visible == "rear_bumper"
    assert img.damage_type == "dent"
    assert img.damage_severity == "medium"
    assert img.object_category == "car"
    assert img.quality_issues == ["blurry_image"]


def test_low_confidence_damage_becomes_unknown():
    client = FakeClient(vision={"images": [_raw("img_1", damage_confidence=0.4)]})
    result = analyze_images(make_claim(), make_extraction(), client)
    assert result.images[0].damage_type == "unknown"


def test_every_submitted_image_gets_exactly_one_analysis():
    # Model omits img_1 and returns img_2 only: img_1 gets a conservative default.
    client = FakeClient(vision={"images": [_raw("img_2")]})
    result = analyze_images(make_claim(image_paths=SAMPLE_TWO), make_extraction(), client)
    assert [img.image_id for img in result.images] == ["img_1", "img_2"]
    assert result.images[0].relevance_to_claim == "not_relevant"
    assert result.images[1].damage_visible is True


def test_unknown_ids_fall_back_to_position():
    client = FakeClient(vision={"images": [_raw("image A"), _raw("image B", damage_visible=False)]})
    result = analyze_images(make_claim(image_paths=SAMPLE_TWO), make_extraction(), client)
    assert [img.image_id for img in result.images] == ["img_1", "img_2"]
    assert result.images[1].damage_visible is False


def test_missing_image_is_not_sent_and_ids_stay_aligned():
    paths = "images/sample/case_001/img_9.jpg;images/sample/case_001/img_1.jpg"
    claim = make_claim(image_paths=paths)
    client = FakeClient(vision={"images": [_raw("img_1")]})
    result = analyze_images(claim, make_extraction(), client)

    _, kwargs = client.calls[0]
    assert len(kwargs["image_paths"]) == 1
    assert kwargs["image_paths"][0].name == "img_1.jpg"
    assert "img_9" not in kwargs["user_text"]
    assert [img.image_id for img in result.images] == ["img_9", "img_1"]
    assert result.images[0].object_matches_claim is False
    assert result.images[1].object_matches_claim is True


def test_no_readable_images_skips_the_model_call():
    client = FakeClient()
    result = analyze_images(make_claim(image_paths="images/does/not/exist.jpg"), make_extraction(), client)
    assert client.calls == []
    assert len(result.images) == 1
    assert result.images[0].object_matches_claim is False
