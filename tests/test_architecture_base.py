"""Pruebas de la arquitectura base y sus metricas."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.architectures import BreastPCRNet, trainable_parameter_count
from src.experiment_config import load_experiment_config
from src.metrics import aggregate_by_patient, binary_metrics, cut_and_patient_metrics


def test_base_architecture_shapes_and_parameter_count() -> None:
    model = BreastPCRNet()
    logits = model(torch.zeros(2, 3, 256, 256))
    assert logits.shape == (2,)
    assert trainable_parameter_count(model) == 294_129
    assert [step.shape for step in model.trace_shapes(2)] == [
        (2, 3, 256, 256),
        (2, 16, 128, 128),
        (2, 32, 64, 64),
        (2, 64, 32, 32),
        (2, 128, 16, 16),
        (2, 128, 1, 1),
        (2, 128),
        (2,),
    ]


def test_base_architecture_backward_reaches_all_parameters() -> None:
    model = BreastPCRNet()
    images = torch.rand(2, 3, 256, 256)
    targets = torch.tensor([0.0, 1.0])
    loss = nn.BCEWithLogitsLoss()(model(images), targets)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_patient_aggregation_uses_mean_and_checks_labels() -> None:
    frame = aggregate_by_patient(
        ["A", "A", "B", "B"],
        [0, 0, 1, 1],
        [0.1, 0.3, 0.7, 0.9],
    )
    assert frame.set_index("patient_id").loc["A", "probability"] == pytest.approx(0.2)
    assert frame.set_index("patient_id").loc["B", "probability"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="etiqueta cambia"):
        aggregate_by_patient(["A", "A"], [0, 1], [0.2, 0.8])


def test_binary_metrics_and_patient_metrics() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    combined = cut_and_patient_metrics(
        ["A", "A", "B", "B"], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    )
    assert combined["patient"]["n"] == 2
    assert combined["patient"]["roc_auc"] == 1.0


def test_experiment_configs_differ_only_in_loss_and_identity() -> None:
    normal = load_experiment_config("configs/experiments/E02_base_normal.json")
    weighted = load_experiment_config("configs/experiments/E03_base_weighted.json")
    assert normal.model == weighted.model
    assert normal.data == weighted.data
    assert normal.training.loss == "normal"
    assert weighted.training.loss == "weighted"
    assert normal.data.train_patients_per_class is None
    assert normal.data.validation_patients_per_class is None


def test_smoke_config_is_explicitly_small() -> None:
    smoke = load_experiment_config(
        "configs/experiments/SMOKE_base_end_to_end.json"
    )
    assert smoke.training.epochs == 2
    assert smoke.data.train_patients_per_class == 4
    assert smoke.data.validation_patients_per_class == 2
