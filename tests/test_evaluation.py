"""Pruebas del contrato de evaluacion por paciente y comparacion pareada."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.compare_experiments import compare_experiments, save_comparison
from src.evaluation import EvaluationConfig, save_evaluation_artifacts


def synthetic_cut_predictions(offset: float = 0.0) -> pd.DataFrame:
    records = []
    for patient_index in range(12):
        target = patient_index % 2
        base_probability = 0.25 if target == 0 else 0.70
        for cut in range(2):
            records.append(
                {
                    "sample_id": f"P{patient_index:02d}_z{cut:03d}",
                    "patient_id": f"P{patient_index:02d}",
                    "slice_index": cut,
                    "split": "train",
                    "fold": 0,
                    "target": target,
                    "probability": float(np.clip(base_probability + offset + cut * 0.02, 0, 1)),
                }
            )
    return pd.DataFrame(records)


def _tag_experiment(directory: Path, experiment_id: str) -> None:
    path = directory / "evaluation_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["experiment_id"] = experiment_id
    path.write_text(json.dumps(summary), encoding="utf-8")


def test_evaluation_saves_traceable_patient_artifacts(tmp_path: Path) -> None:
    config = EvaluationConfig(bootstrap_repetitions=100, seed=7)
    summary = save_evaluation_artifacts(synthetic_cut_predictions(), tmp_path, config)
    patients = pd.read_csv(tmp_path / "predictions_patient.csv")
    assert len(patients) == 12
    assert set(patients["split"]) == {"validation"}
    assert (patients["cuts"] == 2).all()
    assert summary["scope"] == "internal_validation_only"
    assert summary["statistical_unit"] == "patient"
    assert summary["patient_metrics_primary"]["roc_auc"] == 1.0
    assert summary["bootstrap"]["intervals"]["roc_auc"]["valid_repetitions"] > 0
    assert (tmp_path / "predictions_cut.csv").exists()
    assert (tmp_path / "evaluation_patient.png").exists()


def test_evaluation_rejects_public_test_predictions(tmp_path: Path) -> None:
    predictions = synthetic_cut_predictions()
    predictions["split"] = "test"
    with pytest.raises(ValueError, match="validacion interna"):
        save_evaluation_artifacts(
            predictions, tmp_path, EvaluationConfig(bootstrap_repetitions=10)
        )


def test_paired_comparison_requires_same_patients_and_writes_outputs(tmp_path: Path) -> None:
    left = tmp_path / "E02"
    right = tmp_path / "E03"
    config = EvaluationConfig(bootstrap_repetitions=30, seed=3)
    save_evaluation_artifacts(synthetic_cut_predictions(), left, config)
    save_evaluation_artifacts(synthetic_cut_predictions(offset=0.05), right, config)
    _tag_experiment(left, "E02_base_normal")
    _tag_experiment(right, "E03_base_weighted")

    result = compare_experiments(left, right, repetitions=100, seed=11)
    assert result["comparison_design"] == "paired_bootstrap_by_patient"
    assert result["patients"] == 12
    assert result["left_experiment"] == "E02_base_normal"
    assert result["right_experiment"] == "E03_base_weighted"

    output = tmp_path / "comparison"
    save_comparison(result, output)
    assert (output / "comparison.json").exists()
    assert (output / "comparison.csv").exists()
    assert (output / "comparison.png").exists()

    mismatched = pd.read_csv(right / "predictions_patient.csv")
    mismatched.iloc[:-1].to_csv(right / "predictions_patient.csv", index=False)
    with pytest.raises(ValueError, match="mismas pacientes"):
        compare_experiments(left, right, repetitions=10)
