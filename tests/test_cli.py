"""CLI developer-experience checks: input validation, --check preflight, --limit, --quiet."""

import csv
import importlib.util
from pathlib import Path

import pytest
from conftest import FakeClient, make_claim
from test_pipeline import GOOD_EXTRACTION, GOOD_VISION

import main
from config import CLAIMS_CSV, SAMPLE_CLAIMS_CSV, load_user_history
from pipeline import InputError, load_claims, preflight

HEADER = "user_id,image_paths,user_claim,claim_object\n"

_EVAL_PATH = Path(__file__).resolve().parents[1] / "code" / "evaluation" / "main.py"
_spec = importlib.util.spec_from_file_location("evaluation_main_cli", _EVAL_PATH)
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)


def _write(tmp_path, text, name="claims.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_claims_reports_missing_columns(tmp_path):
    path = _write(tmp_path, "user_id,claim_object\nu1,car\n")
    with pytest.raises(InputError, match="missing required column.*image_paths, user_claim"):
        load_claims(path)


def test_load_claims_reports_bad_claim_object_with_line_number(tmp_path):
    path = _write(tmp_path, HEADER + "u1,images/a.jpg,hi,car\nu2,images/b.jpg,hi,boat\n")
    with pytest.raises(
        InputError, match=r"line 3: claim_object must be one of car, laptop, package \(got 'boat'\)"
    ):
        load_claims(path)


def test_load_claims_rejects_non_utf8(tmp_path):
    path = tmp_path / "claims.csv"
    path.write_bytes(HEADER.encode() + b"u1,images/a.jpg,caf\xe9,car\n")
    with pytest.raises(InputError, match="not valid UTF-8"):
        load_claims(path)


def test_preflight_passes_on_shipped_datasets():
    history = load_user_history()
    for csv_path in (CLAIMS_CSV, SAMPLE_CLAIMS_CSV):
        report = preflight(load_claims(csv_path), history)
        assert report.ok
        assert report.num_images >= report.num_claims


def test_preflight_flags_missing_images_and_unknown_users():
    claim = make_claim(user_id="nobody", image_paths="images/sample/case_001/img_1.jpg;images/nope/img_9.jpg")
    report = preflight([claim], load_user_history())
    assert not report.ok
    assert report.missing_images == [("nobody", "img_9")]
    assert report.users_without_history == ["nobody"]


def test_check_mode_needs_no_api_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "Claims:  44" in out
    assert "Preflight passed" in out


def test_check_mode_fails_on_missing_image(tmp_path, capsys):
    path = _write(tmp_path, HEADER + "u1,images/missing/img_1.jpg,hi,car\n")
    assert main.main(["--check", "--input", str(path)]) == 1
    assert "img_1 not found" in capsys.readouterr().out


def test_cli_reports_malformed_input_without_traceback(tmp_path, capsys):
    path = _write(tmp_path, "id,text\n1,hello\n")
    assert main.main(["--input", str(path)]) == 1
    err = capsys.readouterr().err
    assert "missing required column" in err
    assert "Traceback" not in err


def test_cli_rejects_empty_input(tmp_path, capsys):
    path = _write(tmp_path, HEADER)
    assert main.main(["--input", str(path)]) == 1
    assert "no claim rows" in capsys.readouterr().err


def test_cli_missing_key_points_to_env_example(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main.main(["--limit", "1", "--output", str(tmp_path / "out.csv")]) == 1
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert ".env.example" in err
    assert not (tmp_path / "out.csv").exists()


@pytest.mark.parametrize("value", ["0", "-2", "three"])
def test_cli_rejects_bad_limit(value, capsys):
    with pytest.raises(SystemExit) as exc:
        main.main(["--limit", value])
    assert exc.value.code == 2


def test_cli_limit_and_quiet(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        main, "LLMClient", lambda tracker: FakeClient(text=GOOD_EXTRACTION, vision=GOOD_VISION)
    )
    out = tmp_path / "out.csv"
    assert main.main(["-i", str(SAMPLE_CLAIMS_CSV), "-o", str(out), "--limit", "3", "--quiet"]) == 0
    stdout = capsys.readouterr().out
    assert "Stage 1" not in stdout  # per-stage lines suppressed
    assert stdout.count("[") >= 3 and "[3/3]" in stdout
    with open(out, encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 3


def test_quiet_flag_is_reset_after_run(monkeypatch, tmp_path, capsys):
    import pipeline

    monkeypatch.setattr(
        main, "LLMClient", lambda tracker: FakeClient(text=GOOD_EXTRACTION, vision=GOOD_VISION)
    )
    main.main(["-i", str(SAMPLE_CLAIMS_CSV), "-o", str(tmp_path / "o.csv"), "-n", "1", "-q"])
    assert pipeline._verbose is True


def test_evaluation_cli_missing_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert evaluation.main(["--report-dir", str(tmp_path), "--limit", "1"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "evaluation_report.md").exists()


def test_evaluation_cli_writes_report_to_custom_dir(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        evaluation, "LLMClient", lambda tracker: FakeClient(text=GOOD_EXTRACTION, vision=GOOD_VISION)
    )
    assert evaluation.main(["--report-dir", str(tmp_path), "--limit", "2", "--quiet"]) == 0
    assert (tmp_path / "evaluation_report.md").is_file()
    assert (tmp_path / "predictions_sample.csv").is_file()
    assert "EVALUATION SUMMARY" in capsys.readouterr().out
