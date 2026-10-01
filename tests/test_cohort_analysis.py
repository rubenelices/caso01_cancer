"""Cohortes: unión por paciente, incertidumbre y rechazo de test/inconsistencias."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.analyze_cohorts import analyze_cohorts, load_validation, paired_auc
from src.experiment_config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def test_mac_reference_preserves_e05_protocol():
    original = load_experiment_config(ROOT / "configs/experiments/E05_pool_between_convs.json")
    replica = load_experiment_config(ROOT / "configs/experiments/E05_mac_reference.json")
    assert original.data == replica.data
    assert original.model == replica.model
    assert original.training == replica.training
    assert original.output_dir != replica.output_dir
    assert original.checkpoint_dir != replica.checkpoint_dir


@pytest.fixture
def cohort_case(tmp_path):
    root = tmp_path / "data"
    (root / "metadata").mkdir(parents=True)
    samples, metadata, predictions = [], [], []
    for index in range(10):
        patient = f"V{index:02d}"
        cohort = "duke" if index < 4 else "spy1" if index < 8 else "spy2"
        target = index % 2 if index < 8 else 0
        metadata.append({"pid": patient, "dataset": cohort, "pCR": target, "split": "train"})
        for cut in range(2):
            sample = f"{patient}_{cut}"
            samples.append({"sample_id": sample, "patient_id": patient, "pCR": target,
                "split": "train", "fold": 0, "slice_index": cut,
                "path_pre": "absent.png", "path_early": "absent.png", "path_late": "absent.png"})
            predictions.append({"sample_id": sample, "patient_id": patient, "target": target,
                "probability": .8 if target else .2, "split": "train", "fold": 0})
    for patient, split, fold in (("T", "train", 1), ("TEST", "test", -1)):
        metadata.append({"pid": patient, "dataset": "duke", "pCR": 1, "split": split})
        samples.append({"sample_id": patient, "patient_id": patient, "pCR": 1,
            "split": split, "fold": fold, "slice_index": 0,
            "path_pre": "absent.png", "path_early": "absent.png", "path_late": "absent.png"})
    pd.DataFrame(samples).to_csv(root / "metadata/samples.csv", index=False)
    pd.DataFrame(metadata).to_csv(root / "metadata/patients.csv", index=False)
    directories = []
    for name in ("LEFT", "RIGHT"):
        directory = tmp_path / name
        directory.mkdir()
        original = load_experiment_config(ROOT / "configs/experiments/E05_mac_reference.json")
        config = replace(original, experiment_id=name, data=replace(original.data, root=str(root)))
        (directory / "config_resolved.json").write_text(json.dumps(config.to_dict()))
        (directory / "summary.json").write_text(json.dumps({"status": "complete",
            "scientific_result": True, "test_evaluated": False, "experiment_id": name, "best_epoch": 1}))
        (directory / "evaluation_summary.json").write_text(json.dumps({
            "scope": "internal_validation_only", "aggregation": "mean", "threshold": .5}))
        cuts = pd.DataFrame(predictions)
        cuts.to_csv(directory / "predictions_cut.csv", index=False)
        patients = cuts.groupby("patient_id").agg(target=("target", "first"),
            probability=("probability", "mean"), cuts=("sample_id", "size")).reset_index()
        patients["split"], patients["fold"] = "validation", 0
        patients.to_csv(directory / "predictions_patient.csv", index=False)
        directories.append(directory)
    return root, directories


def test_report_uses_validation_without_images_and_handles_one_class(cohort_case, tmp_path, monkeypatch):
    root, directories = cohort_case
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("No leer imágenes"))
    before = [(path, path.read_bytes()) for directory in directories for path in directory.iterdir()]
    output = tmp_path / "report"
    report = analyze_cohorts(directories, output, repetitions=10)
    assert report["test_evaluated"] is False
    groups = report["experiments"][0]["groups"]
    assert groups["ALL"]["n"] == 10
    assert groups["duke"]["metrics"]["roc_auc"] == 1
    assert groups["spy2"]["metrics"]["roc_auc"] is None
    assert groups["spy2"]["metrics"]["pr_auc"] is None
    assert groups["spy2"]["small_group_warning"] is True
    assert report["paired_comparisons"][0]["auc_delta"]["ALL"] == {
        "difference": 0., "lower": 0., "upper": 0.}
    annotated = pd.read_csv(output / "LEFT_patients.csv")
    assert "TEST" not in annotated.patient_id.tolist()
    for name in ("cohort_summary.json", "cohort_metrics.csv", "cohort_dashboard.png", "README.md"):
        assert (output / name).is_file()
    assert all(path.read_bytes() == content for path, content in before)
    with pytest.raises(FileExistsError):
        analyze_cohorts(directories, output, repetitions=10)


@pytest.mark.parametrize("mutation", ["test", "incomplete", "label", "probability", "missing_cut", "patient_mean"])
def test_rejects_invalid_saved_predictions(cohort_case, mutation):
    _, directories = cohort_case
    directory = directories[0]
    if mutation in {"test", "incomplete"}:
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        summary["test_evaluated" if mutation == "test" else "scientific_result"] = mutation == "test"
        path.write_text(json.dumps(summary))
    elif mutation == "patient_mean":
        path = directory / "predictions_patient.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "probability"] = .99
        frame.to_csv(path, index=False)
    else:
        path = directory / "predictions_cut.csv"
        frame = pd.read_csv(path)
        if mutation == "label":
            frame.loc[0, "target"] = 1
        elif mutation == "probability":
            frame.loc[0, "probability"] = np.inf
        else:
            frame = frame.iloc[1:]
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_validation(directory)


def test_rejects_missing_cohort_and_misalignment(cohort_case):
    root, directories = cohort_case
    left, _ = load_validation(directories[0])
    with pytest.raises(ValueError, match="mismas pacientes"):
        paired_auc(left, left.iloc[:-1], 10, 42)
    path = root / "metadata/patients.csv"
    metadata = pd.read_csv(path)
    metadata.loc[0, "dataset"] = None
    metadata.to_csv(path, index=False)
    with pytest.raises(ValueError, match="cohorte"):
        load_validation(directories[0])
