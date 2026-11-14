import pytest

from config import CAR_OBJECT_PARTS, ISSUE_TYPE_VALUES, PACKAGE_OBJECT_PARTS, SEVERITY_VALUES
from normalize import DAMAGE_SYNONYMS, canonical, normalize_label, normalize_severity


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Rear Bumper", "rear_bumper"), ("  front-bumper ", "front_bumper"), (None, ""), (3, "3")],
)
def test_canonical(raw, expected):
    assert canonical(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("dent", "dent"),
        ("Glass Shatter", "glass_shatter"),
        ("cracked", "crack"),  # substring of an allowed value
        ("scratched paint", "scratch"),
        ("broken", "broken_part"),
        ("shattered", "glass_shatter"),
        ("wet box", "water_damage"),
        ("completely unrecognisable", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_normalize_issue_type(raw, expected):
    assert normalize_label(raw, ISSUE_TYPE_VALUES, synonyms=DAMAGE_SYNONYMS) == expected


def test_normalize_part_prefers_most_specific_candidate():
    assert normalize_label("corner", PACKAGE_OBJECT_PARTS) == "package_corner"
    assert normalize_label("left side mirror", CAR_OBJECT_PARTS) == "side_mirror"


def test_normalize_is_order_independent():
    """Result must not depend on set iteration order (hash randomization)."""
    allowed = list(CAR_OBJECT_PARTS)
    forward = normalize_label("bumper", allowed)
    backward = normalize_label("bumper", list(reversed(allowed)))
    assert forward == backward


def test_empty_value_never_matches_a_candidate():
    # "" is a substring of every candidate; it must still fall back to default.
    assert normalize_label("   ", CAR_OBJECT_PARTS, default="unknown") == "unknown"


def test_synonym_only_mode_skips_substring_matching():
    assert (
        normalize_label("no damage", ISSUE_TYPE_VALUES, synonyms=DAMAGE_SYNONYMS, substring=False) == "none"
    )
    assert normalize_label("dent-ish", ISSUE_TYPE_VALUES, substring=False) == "unknown"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("high", "high"),
        ("Minor", "low"),
        ("moderate", "medium"),
        ("severe", "high"),
        ("no visible damage", "none"),
        ("???", "unknown"),
    ],
)
def test_normalize_severity(raw, expected):
    assert normalize_severity(raw, SEVERITY_VALUES) == expected
