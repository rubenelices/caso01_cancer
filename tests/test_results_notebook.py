"""El cuaderno conserva resúmenes sin archivos locales y rechaza test/debug."""
import json
from pathlib import Path

import pytest

from src.results_notebook import load_results, load_histories, load_cohort_reports

ROOT = Path(__file__).resolve().parents[1]


def bare_registry(tmp_path):
    (tmp_path / "experimentos").mkdir()
    (tmp_path / "experimentos/resultados_registrados.json").write_bytes(
        (ROOT / "experimentos/resultados_registrados.json").read_bytes())
    return tmp_path


def test_registry_without_dataset_or_reports(tmp_path):
    root = bare_registry(tmp_path)
    results, warnings = load_results(root)
    assert len(results) == 13
    assert len(warnings) == 13
    assert results.loc[results.run_id == "E05_mac", "status"].item() == "complete"
    assert results.loc[results.run_id == "E05_mac", "roc_auc"].item() == pytest.approx(.5868951612903226)
    assert results.loc[results.run_id == "E05_universidad", "roc_auc"].item() == pytest.approx(.6140120967741935)
    assert load_histories(root, results) == {}
    assert results.loc[results.run_id == "E11_lr_0003_mac", "roc_auc"].item() == pytest.approx(.5502016129032258)
    assert results.loc[results.run_id == "E12_lr_0001_mac", "roc_auc"].item() == pytest.approx(.5579637096774194)
    assert results.loc[results.run_id == "E12_lr_0001_mac", "best_epoch"].item() == 3
    assert results.loc[results.run_id == "E13_phase_differences_mac", "status"].item() == "complete"
    assert results.loc[results.run_id == "E13_phase_differences_mac", "roc_auc"].item() == pytest.approx(.5724798387096774)
    assert load_cohort_reports(root) == []


def test_future_pending_result_is_not_zero(tmp_path):
    root = bare_registry(tmp_path)
    path = root / "experimentos/resultados_registrados.json"
    registry = json.loads(path.read_text())
    row = next(row for row in registry["runs"] if row["run_id"] == "E05_mac")
    row.update(status="pending", roc_auc=None, pr_auc=None, best_epoch=None)
    path.write_text(json.dumps(registry))
    results, _ = load_results(root)
    pending = results[results.run_id == "E05_mac"].iloc[0]
    assert pending.status == "pending"
    assert pending.roc_auc != pending.roc_auc  # NaN, no un AUC ficticio de cero.


def test_cohort_snapshots_survive_without_reports_or_patients(tmp_path):
    root = bare_registry(tmp_path)
    (root / "experimentos/cohortes_registradas.json").write_bytes(
        (ROOT / "experimentos/cohortes_registradas.json").read_bytes())
    reports = load_cohort_reports(root)
    assert len(reports) == 2
    latest = reports[-1]
    assert "Snapshot" in latest["notebook_source"]
    assert len(latest["experiments"]) == 3
    assert latest["paired_comparisons"][1]["auc_delta"]["ALL"]["difference"] == pytest.approx(.0037298387096774688)
    assert all("patient_id" not in json.dumps(report) for report in reports)


@pytest.mark.parametrize("invalid", [False, True])
def test_local_summary_updates_only_compatible_results(tmp_path, invalid):
    root = bare_registry(tmp_path)
    directory = root / "reports/experiments/E05_mac_reference/fold_0_seed_42"
    directory.mkdir(parents=True)
    config = json.loads((ROOT / "configs/experiments/E05_mac_reference.json").read_text())
    (directory / "config_resolved.json").write_text(json.dumps(config))
    summary = {"experiment_id": "E05_mac_reference", "status": "complete",
        "scientific_result": True, "test_evaluated": invalid, "best_epoch": 2,
        "epochs_completed": 10, "parameter_count": 294129, "total_seconds": 123,
        "environment": {"device": "mps"}, "validation_metrics": {
            "aggregation": "mean", "patient": {"n": 219, "roc_auc": .6,
                "pr_auc": .4, "sensitivity": 0., "specificity": 1., "balanced_accuracy": .5}}}
    (directory / "summary.json").write_text(json.dumps(summary))
    if invalid:
        with pytest.raises(ValueError, match="no científico"):
            load_results(root)
    else:
        results, _ = load_results(root)
        row = results[results.run_id == "E05_mac"].iloc[0]
        assert row.status == "complete"
        assert row.roc_auc == .6
        assert row.device == "mps"
