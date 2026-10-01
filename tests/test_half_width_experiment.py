"""E10 reduce solo anchura; conserva profundidad, resolucion y protocolo."""

from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from src.architectures import BreastPCRNet, trainable_parameter_count
from src.document_architectures import trace
from src.experiment_config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return load_experiment_config(ROOT / "configs/experiments" / name)


def test_only_width_changes_against_e05():
    base = config("E05_pool_between_convs.json")
    candidate = config("E10_half_width.json")
    assert candidate.data == base.data
    assert candidate.training == base.training
    assert candidate.model == replace(base.model, channels=(8, 16, 32, 64))
    assert candidate.output_dir != base.output_dir
    assert candidate.checkpoint_dir != base.checkpoint_dir
    assert candidate.data.augmentation.enabled is False


def test_shapes_parameters_receptive_field_and_gradients():
    model = BreastPCRNet(config("E10_half_width.json").model)
    assert trainable_parameter_count(model) == 73913
    assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) == 8
    assert sum(isinstance(layer, nn.MaxPool2d) for layer in model.modules()) == 4
    steps = trace(model)
    pools = [step for step in steps if step["kind"] == "pool"]
    assert [step["shape"] for step in pools] == [
        (8, 128, 128), (16, 64, 64), (32, 32, 32), (64, 16, 16)]
    assert [step for step in steps if step["kind"] == "conv"][-1]["rf"] == 106
    logits = model.train()(torch.rand(2, 3, 256, 256))
    assert logits.shape == (2,)
    nn.BCEWithLogitsLoss()(logits, torch.tensor([0., 1.])).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_smoke_has_same_architecture_and_separate_outputs():
    real = config("E10_half_width.json")
    smoke = config("SMOKE_half_width.json")
    assert smoke.model == real.model
    assert smoke.training.epochs == 2
    assert smoke.data.train_patients_per_class == 4
    assert smoke.data.validation_patients_per_class == 2
    assert smoke.output_dir != real.output_dir
    assert smoke.checkpoint_dir != real.checkpoint_dir
