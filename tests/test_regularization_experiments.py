"""Cada experimento cambia una variable de E05 y conserva sus resultados."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from src.architectures import BreastPCRNet, trainable_parameter_count
from src.experiment_config import load_experiment_config


EXPERIMENTS = ["E07_dropout_050", "E08_weight_decay_001"]


@pytest.mark.parametrize("experiment_id", EXPERIMENTS)
def test_regularization_changes_only_one_variable(experiment_id: str) -> None:
    base = load_experiment_config("configs/experiments/E05_pool_between_convs.json")
    candidate = load_experiment_config(f"configs/experiments/{experiment_id}.json")
    assert candidate.data == base.data
    assert candidate.experiment_id == experiment_id
    if experiment_id == "E07_dropout_050":
        assert candidate.model == replace(base.model, dropout=0.5)
        assert candidate.training == base.training
    else:
        assert candidate.model == base.model
        assert candidate.training == replace(base.training, weight_decay=0.001)
    assert candidate.data.train_patients_per_class is None
    assert candidate.data.validation_patients_per_class is None


def test_regularization_output_paths_are_unique() -> None:
    ids = ["E05_pool_between_convs", "E06_five_blocks_normal", *EXPERIMENTS,
           "SMOKE_dropout_050", "SMOKE_weight_decay_001"]
    configs = [load_experiment_config(f"configs/experiments/{name}.json") for name in ids]
    assert len({c.output_dir for c in configs}) == len(configs)
    assert len({c.checkpoint_dir for c in configs}) == len(configs)


@pytest.mark.parametrize("experiment_id", EXPERIMENTS)
def test_regularization_model_and_optimizer_use_config(experiment_id: str) -> None:
    config = load_experiment_config(f"configs/experiments/{experiment_id}.json")
    torch.manual_seed(config.training.seed)
    model = BreastPCRNet(config.model)
    assert model.dropout.p == config.model.dropout
    assert trainable_parameter_count(model) == 294_129
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate,
                                  weight_decay=config.training.weight_decay)
    assert optimizer.param_groups[0]["weight_decay"] == config.training.weight_decay
    logits = model(torch.rand(2, 3, 256, 256))
    assert logits.shape == (2,)
    loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        assert torch.isfinite(model(torch.rand(2, 3, 256, 256))).all()


@pytest.mark.parametrize("name", ["dropout_050", "weight_decay_001"])
def test_regularization_smoke_preserves_effective_settings(name: str) -> None:
    experiment_id = "E07_dropout_050" if name == "dropout_050" else "E08_weight_decay_001"
    real = load_experiment_config(f"configs/experiments/{experiment_id}.json")
    smoke = load_experiment_config(f"configs/experiments/SMOKE_{name}.json")
    assert smoke.model == real.model
    assert smoke.training.weight_decay == real.training.weight_decay
    assert smoke.training.learning_rate == real.training.learning_rate
    assert smoke.data.train_patients_per_class == 4
    assert smoke.data.validation_patients_per_class == 2
    assert smoke.training.epochs == 2
    assert smoke.output_dir != real.output_dir
    assert smoke.checkpoint_dir != real.checkpoint_dir
