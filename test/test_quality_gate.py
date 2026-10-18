import json

from src.models.quality_gate import check, main


def _write(tmp_path, **metrics):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"metrics": metrics}))
    return path


def test_check_passes_when_metrics_meet_thresholds():
    assert check({"cv_roc_auc_mean": 0.9, "test_recall": 0.9}, 0.85, 0.75) == []


def test_check_reports_every_failing_metric():
    failures = check({"cv_roc_auc_mean": 0.7, "test_recall": 0.5}, 0.85, 0.75)
    assert len(failures) == 2
    assert "cv_roc_auc_mean" in failures[0]


def test_check_flags_missing_metric():
    assert "missing" in check({"cv_roc_auc_mean": 0.9}, 0.85, 0.75)[0]


def test_main_exit_codes(tmp_path, capsys):
    good = _write(tmp_path, cv_roc_auc_mean=0.9, test_recall=0.9)
    assert main(["--metadata", str(good)]) == 0
    bad = _write(tmp_path, cv_roc_auc_mean=0.6, test_recall=0.9)
    assert main(["--metadata", str(bad)]) == 1
    assert "QUALITY GATE FAILED" in capsys.readouterr().err
