"""E11/E12 conservan E05 Mac y cambian solo el learning rate inicial."""
from dataclasses import replace

import pytest
import torch

from src.architectures import BreastPCRNet, trainable_parameter_count
from src.experiment_config import load_experiment_config

CASES = [("E11_lr_0003", "SMOKE_lr_0003", .0003),
         ("E12_lr_0001", "SMOKE_lr_0001", .0001)]


def config(name):
    return load_experiment_config(f"configs/experiments/{name}.json")


@pytest.mark.parametrize("name,smoke_name,rate", CASES)
def test_only_initial_learning_rate_changes(name, smoke_name, rate):
    base, candidate = config("E05_mac_reference"), config(name)
    assert candidate.data == base.data
    assert candidate.model == base.model
    assert candidate.training == replace(base.training, learning_rate=rate)
    assert candidate.training.epochs == 10
    assert candidate.data.augmentation.enabled is False


def test_unique_output_and_checkpoint_paths():
    names = ["E05_mac_reference"] + [name for real, smoke, _ in CASES for name in (real, smoke)]
    configs = [config(name) for name in names]
    assert len({c.output_dir for c in configs}) == len(configs)
    assert len({c.checkpoint_dir for c in configs}) == len(configs)


@pytest.mark.parametrize("name,smoke_name,rate", CASES)
def test_optimizer_uses_actual_rate_and_scheduler(name, smoke_name, rate):
    candidate = config(name)
    torch.manual_seed(candidate.training.seed)
    model = BreastPCRNet(candidate.model)
    assert trainable_parameter_count(model) == 294129
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate.training.learning_rate,
                                  weight_decay=candidate.training.weight_decay)
    assert optimizer.param_groups[0]["lr"] == rate
    assert optimizer.param_groups[0]["weight_decay"] == .0001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=.5)
    for score in [.6, .59, .59, .59, .59]:
        scheduler.step(score)
    assert optimizer.param_groups[0]["lr"] == rate * .5


@pytest.mark.parametrize("name,smoke_name,rate", CASES)
def test_smoke_preserves_learning_rate_and_model(name, smoke_name, rate):
    real, smoke = config(name), config(smoke_name)
    assert smoke.model == real.model
    assert smoke.training.learning_rate == real.training.learning_rate == rate
    assert smoke.training.scheduler_patience == real.training.scheduler_patience
    assert smoke.data.train_patients_per_class == 4
    assert smoke.data.validation_patients_per_class == 2
    assert smoke.training.epochs == 2
