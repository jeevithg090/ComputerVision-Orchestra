import importlib.util
from pathlib import Path

import pytest
from conftest import make_claim

from llm_client import TokenTracker
from pipeline import fallback_result

_PATH = Path(__file__).resolve().parents[1] / "code" / "evaluation" / "main.py"
_spec = importlib.util.spec_from_file_location("evaluation_main", _PATH)
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)


def test_exact_match_accuracy():
    assert evaluation.exact_match_accuracy(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)
    assert evaluation.exact_match_accuracy([], []) == 0.0


def test_multi_label_f1():
    assert evaluation.multi_label_f1(["none"], ["none"]) == 1.0
    assert evaluation.multi_label_f1(["a;b"], ["a"]) == pytest.approx(2 / 3)
    assert evaluation.multi_label_f1(["a"], ["b"]) == 0.0


def test_set_overlap_score():
    assert evaluation.set_overlap_score(["img_1;img_2"], ["img_1"]) == pytest.approx(0.5)
    assert evaluation.set_overlap_score(["none"], ["none"]) == 1.0
    assert evaluation.set_overlap_score([""], ["img_1"]) == 0.0


def test_confusion_matrix():
    labels = ["supported", "contradicted"]
    cm = evaluation.confusion_matrix(["supported", "contradicted"], ["supported", "supported"], labels)
    assert cm["supported"] == {"supported": 1, "contradicted": 1}


def test_load_sample_claims_with_labels():
    rows = evaluation.load_sample_claims_with_labels(evaluation.SAMPLE_CLAIMS_CSV)
    assert len(rows) == 20
    claim, expected = rows[0]
    assert claim.user_id
    assert expected["claim_status"] in {"supported", "contradicted", "not_enough_information"}


def test_generate_report_writes_artifacts(tmp_path, capsys):
    results = [fallback_result(make_claim())]
    expected = [
        {
            "evidence_standard_met": "false",
            "evidence_standard_met_reason": "",
            "risk_flags": "manual_review_required",
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "",
            "supporting_image_ids": "none",
            "valid_image": "false",
            "severity": "unknown",
        }
    ]
    report = tmp_path / "evaluation_report.md"
    metrics = evaluation.generate_report(results, expected, TokenTracker(), 1.0, report)
    assert metrics["claim_status"] == 1.0
    assert "# Evaluation Report" in report.read_text(encoding="utf-8")
    assert (tmp_path / "predictions_sample.csv").exists()
    assert (tmp_path / "intermediate_sample.jsonl").exists()
