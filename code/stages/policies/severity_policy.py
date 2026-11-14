"""Final severity calibration.

The image model's visual severity is useful evidence, but the final label
should be stable across object types and final decision status.
"""

from __future__ import annotations


def calibrate_severity(
    *,
    claim_status: str,
    issue_type: str,
    claim_object: str,
    object_part: str,
    image_severity: str,
    rule_id: str,
    claimed_issue_type: str = "unknown",
) -> str:
    if claim_status == "not_enough_information":
        return "unknown"
    if issue_type == "none":
        return "none"
    if rule_id == "wrong_object":
        return "low" if claimed_issue_type in {"crushed_packaging", "dent", "scratch"} else "unknown"
    if issue_type == "unknown":
        return "unknown"

    if claim_object == "car" and issue_type == "dent":
        return (
            "medium"
            if object_part in {"rear_bumper", "door", "front_bumper", "quarter_panel", "body"}
            else "low"
        )

    if issue_type == "scratch":
        return "low"
    if issue_type in {"crack", "broken_part", "torn_packaging", "water_damage", "stain", "missing_part"}:
        return "medium"
    if issue_type == "crushed_packaging":
        return "high" if image_severity == "high" and object_part == "box" else "medium"
    if issue_type == "glass_shatter":
        return "medium" if claim_object == "laptop" and object_part == "screen" else "high"
    if issue_type == "dent":
        return "low"

    return image_severity if image_severity in {"none", "low", "medium", "high", "unknown"} else "unknown"
