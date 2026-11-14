"""Validate the committed predictions file against the problem-statement contract."""

import csv

import pytest

from config import (
    CLAIM_STATUS_VALUES,
    CLAIMS_CSV,
    ISSUE_TYPE_VALUES,
    OBJECT_PARTS,
    OUTPUT_CSV,
    RISK_FLAG_VALUES,
    SEVERITY_VALUES,
)
from pipeline import OUTPUT_COLUMNS


def _rows(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


pytestmark = pytest.mark.skipif(not OUTPUT_CSV.exists(), reason="output.csv not generated")


def test_output_has_one_row_per_claim_in_order():
    _, claims = _rows(CLAIMS_CSV)
    columns, rows = _rows(OUTPUT_CSV)
    assert columns == OUTPUT_COLUMNS
    assert len(rows) == len(claims)
    for claim, row in zip(claims, rows, strict=True):
        assert (row["user_id"], row["image_paths"]) == (claim["user_id"], claim["image_paths"])


def test_output_values_are_allowed():
    _, rows = _rows(OUTPUT_CSV)
    for row in rows:
        assert row["claim_status"] in CLAIM_STATUS_VALUES
        assert row["issue_type"] in ISSUE_TYPE_VALUES
        assert row["severity"] in SEVERITY_VALUES
        assert row["object_part"] in OBJECT_PARTS[row["claim_object"]]
        assert row["evidence_standard_met"] in {"true", "false"}
        assert row["valid_image"] in {"true", "false"}
        assert set(row["risk_flags"].split(";")) <= RISK_FLAG_VALUES
        assert row["claim_status_justification"].strip()
        ids = row["supporting_image_ids"]
        if ids != "none":
            submitted = {p.rsplit("/", 1)[-1].rsplit(".", 1)[0] for p in row["image_paths"].split(";")}
            assert set(ids.split(";")) <= submitted
