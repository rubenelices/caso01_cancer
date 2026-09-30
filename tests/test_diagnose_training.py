"""El diagnóstico no entrena, no toca test y conserva el checkpoint original."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.architectures import BaseCNNConfig, BreastPCRNet
from src.data import BreastDCEDataset, load_samples, split_by_patient_fold
from src.diagnose_training import (file_sha256, paired_auc_interval,
    recalibrate_batchnorm, run_diagnosis, validate_internal_loader)
from src.experiment_config import (DataExperimentConfig, ExperimentConfig,
                                  TrainingConfig, experiment_config_from_dict)


@pytest.fixture
def diagnosis_case(tmp_path):
    root = tmp_path / "data"
    records = []
    for role, fold in (("T", 1), ("V", 0), ("TEST", -1)):
        for label in (0, 1):
            patient = f"{role}{label}"
            split = "test" if role == "TEST" else "train"
            for cut in range(2):
                sample_id = f"{patient}_z{cut:03d}"
                record = {"sample_id": sample_id, "patient_id": patient, "split": split,
                          "fold": fold, "pCR": label, "slice_index": cut}
                for phase, name in enumerate(("pre", "early", "late")):
                    path = Path("dataset") / split / patient / f"{sample_id}_{name}.png"
                    (root / path).parent.mkdir(parents=True, exist_ok=True)
                    image = np.zeros((256, 256), dtype=np.uint8)
                    image[80:150, 90:160] = 30 + 20 * phase + label * 40 + cut * 5
                    Image.fromarray(image).save(root / path)
                    record[f"path_{name}"] = str(path)
                records.append(record)
    (root / "metadata").mkdir()
    pd.DataFrame(records).to_csv(root / "metadata/samples.csv", index=False)
    config = ExperimentConfig("SYNTHETIC_DIAGNOSIS", "Prueba tecnica sintetica",
        DataExperimentConfig(root=str(root), validation_fold=0, batch_size=2,
                             num_workers=0, train_patients_per_class=1,
                             validation_patients_per_class=1),
        BaseCNNConfig(channels=(4, 8)), TrainingConfig(epochs=2),
        str(tmp_path / "training_report"), str(tmp_path / "checkpoints"))
    torch.manual_seed(42)
    model = BreastPCRNet(config.model).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save({"config": config.to_dict(), "experiment_id": config.experiment_id,
                "epoch": 1, "model_state_dict": model.state_dict()}, checkpoint)
    return root, config, model, checkpoint


def test_checkpoint_config_roundtrip(diagnosis_case):
    _, config, _, _ = diagnosis_case
    assert experiment_config_from_dict(config.to_dict()) == config


def test_bn_recalibration_preserves_original_weights_and_disables_dropout(diagnosis_case):
    root, config, model, _ = diagnosis_case
    rows = split_by_patient_fold(load_samples(root), 0).train
    loader = DataLoader(BreastDCEDataset(rows, root), batch_size=2)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    candidate, details = recalibrate_batchnorm(model, loader, torch.device("cpu"), 0)
    assert candidate is not model and details["source"] == "train_only"
    assert details["calibration_batches"] == 2
    assert not candidate.dropout.training
    assert all(not module.training for module in candidate.modules())
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())
    assert all(torch.equal(dict(model.named_parameters())[name], tensor)
               for name, tensor in candidate.named_parameters())
    assert all(parameter.grad is None for parameter in candidate.parameters())
    assert any(not torch.equal(before[name], tensor) for name, tensor in candidate.state_dict().items())


def test_reject_test_and_validation_for_bn_before_reading_pixels(diagnosis_case):
    root, _, model, _ = diagnosis_case
    splits = split_by_patient_fold(load_samples(root), 0)
    with pytest.raises(ValueError, match="test"):
        validate_internal_loader(DataLoader(BreastDCEDataset(splits.test, root)))
    with pytest.raises(ValueError, match="validación"):
        recalibrate_batchnorm(model, DataLoader(BreastDCEDataset(splits.validation, root)), torch.device("cpu"), 0)
    with pytest.raises(ValueError, match="aumentos"):
        validate_internal_loader(DataLoader(BreastDCEDataset(splits.train, root, transform=lambda x: x)))


def test_full_diagnosis_saves_visual_report_without_test_or_optimizer(diagnosis_case, tmp_path, monkeypatch):
    root, _, _, checkpoint = diagnosis_case
    import src.data as data
    original_decode = data.decode_phase_png
    def guard(source, **kwargs):
        assert "/dataset/test/" not in str(source)
        return original_decode(source, **kwargs)
    monkeypatch.setattr(data, "decode_phase_png", guard)
    def forbid_optimizer(*args, **kwargs):
        raise AssertionError("El diagnóstico no puede entrenar")
    monkeypatch.setattr(torch.optim, "AdamW", forbid_optimizer)
    digest = file_sha256(checkpoint)
    output = tmp_path / "diagnostic_report"
    report, actual_output = run_diagnosis(checkpoint, device_name="cpu", num_workers=0,
                                         output_dir=output, bootstrap_repetitions=20)
    assert actual_output == output
    assert file_sha256(checkpoint) == digest
    assert report["technical_subset"] is True
    assert report["test_evaluated"] is False
    assert report["checkpoint_unchanged"] and report["original_model_unchanged"]
    assert report["patients"] == {"train": 2, "validation": 2}
    assert set(report["metrics"]) == {"train_original", "validation_original",
                                     "train_bn_recalibrated", "validation_bn_recalibrated"}
    assert (output / "diagnostic_dashboard.png").is_file()
    assert (output / "README.md").is_file()
    assert json.loads((output / "diagnosis.json").read_text())["batchnorm"]["source"] == "train_only"
    assert len(list(output.glob("*_patients.csv"))) == 4
    assert not list(output.glob("*.pt"))
    with pytest.raises(FileExistsError):
        run_diagnosis(checkpoint, output_dir=output)


def test_paired_bootstrap_preserves_patient_pairing():
    left = pd.DataFrame({"patient_id": ["A", "B", "C", "D"],
                         "target": [0, 1, 0, 1], "probability": [.1, .8, .2, .9]})
    same = paired_auc_interval(left, left.copy(), 20, 42)
    assert same["lower"] == same["upper"] == 0
    with pytest.raises(ValueError, match="mismas"):
        paired_auc_interval(left, left.iloc[::-1].reset_index(drop=True), 20, 42)


def test_model_without_bn_is_not_retrained(diagnosis_case):
    root, config, _, _ = diagnosis_case
    model = BreastPCRNet(replace(config.model, use_batch_norm=False)).eval()
    rows = split_by_patient_fold(load_samples(root), 0).train
    candidate, details = recalibrate_batchnorm(model,
        DataLoader(BreastDCEDataset(rows, root), batch_size=2), torch.device("cpu"), 0)
    assert details["layers"] == [] and details["calibration_batches"] == 0
    assert all(torch.equal(model.state_dict()[name], tensor) for name, tensor in candidate.state_dict().items())
