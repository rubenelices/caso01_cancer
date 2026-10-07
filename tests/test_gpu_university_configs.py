"""University repeats preserve the Mac scientific protocol; no real data."""
from dataclasses import replace
from pathlib import Path

import pytest

from src.architectures import BreastPCRNet, trainable_parameter_count
from src.experiment_config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ('E19_weight_decay_001_mac', 'E19_reference_cuda'),
    ('E48_ema_mac', 'E48_ema_cuda'),
    ('E51_label_smoothing_mac', 'E51_label_smoothing_cuda'),
]


@pytest.mark.parametrize('mac,gpu', PAIRS)
def test_gpu_repeat_changes_only_identity_and_storage(mac, gpu):
    old = load_experiment_config(ROOT / 'configs/experiments' / f'{mac}.json')
    new = load_experiment_config(ROOT / 'configs/gpu' / f'{gpu}.json')
    assert old.model == new.model
    assert old.data == new.data
    assert old.training == new.training
    assert new.output_dir == f'reports/gpu/{gpu}/fold_0_seed_42'
    assert new.checkpoint_dir == f'checkpoints/{gpu}/fold_0_seed_42'
    assert new.output_dir != old.output_dir and new.checkpoint_dir != old.checkpoint_dir
    assert not new.training.mixed_precision
    assert new.data.batch_size == 16 and new.training.seed == 42
    assert new.training.epochs == 50 and new.training.early_stopping_patience == 10
    assert trainable_parameter_count(BreastPCRNet(new.model)) == 294129


@pytest.mark.parametrize('mac,gpu', PAIRS)
def test_gpu_smoke_has_separate_storage_and_tiny_patient_subset(mac, gpu):
    full = load_experiment_config(ROOT / 'configs/gpu' / f'{gpu}.json')
    tiny = load_experiment_config(ROOT / 'configs/gpu' / f'SMOKE_{gpu}.json')
    assert full.model == tiny.model
    assert tiny.data == replace(full.data, train_patients_per_class=4,
                               validation_patients_per_class=2)
    assert tiny.training == replace(full.training, epochs=2, early_stopping_patience=3)
    assert tiny.output_dir.startswith('reports/smoke/')
    assert tiny.checkpoint_dir != full.checkpoint_dir


def test_only_the_intended_policy_differs_between_gpu_candidates():
    base, ema, smooth = [load_experiment_config(ROOT / 'configs/gpu' / f'{p[1]}.json')
                         for p in PAIRS]
    assert base.data == ema.data == smooth.data
    assert base.model == ema.model == smooth.model
    assert ema.training == replace(base.training, ema=replace(base.training.ema, enabled=True))
    assert smooth.training == replace(base.training,
                                     label_smoothing=replace(base.training.label_smoothing, enabled=True))
    assert ema.training.ema.decay == 0.99
    assert smooth.training.label_smoothing.epsilon == 0.1
