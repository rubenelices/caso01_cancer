"""Alineación de fases, reproducibilidad y ausencia de aumentos en evaluación."""

from dataclasses import replace

import pytest
import torch

from src.augmentation import (AffineParameters, AugmentationConfig, SharedRandomAffine,
                              apply_shared_affine, build_train_transform)
from src.data import LoaderConfig, create_dataloaders, split_by_patient_fold
from src.experiment_config import load_experiment_config
from test_data import create_tiny_dataset


def pattern():
    image = torch.zeros(3, 256, 256)
    image[:, 90:150, 110:160] = torch.tensor([0.2, 0.4, 0.6])[:, None, None]
    return image


def test_shared_grid_preserves_phase_alignment_range_and_original():
    original = pattern()
    before = original.clone()
    output = apply_shared_affine(original, AffineParameters(5, 7.68, -7.68))
    assert torch.equal(original, before)
    assert output.shape == original.shape and output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert 0 <= float(output.min()) <= float(output.max()) <= 1
    assert torch.allclose(output[1], output[0] * 2, atol=1e-6)
    assert torch.allclose(output[2], output[0] * 3, atol=1e-6)
    assert not torch.equal(output, original)


def test_affine_identity_and_translation_direction():
    image = pattern()
    assert torch.equal(apply_shared_affine(image, AffineParameters(0, 0, 0)), image)
    translated = apply_shared_affine(image, AffineParameters(0, 5, 3))
    assert torch.allclose(translated[:, 3:, 5:], image[:, :-3, :-5], atol=1e-5)


def test_random_parameters_are_bounded_reproducible_and_not_fixed():
    transform = SharedRandomAffine(AugmentationConfig(enabled=True))
    first = torch.Generator().manual_seed(42)
    second = torch.Generator().manual_seed(42)
    values = [transform.sample_parameters(generator=first) for _ in range(20)]
    assert values == [transform.sample_parameters(generator=second) for _ in range(20)]
    assert len(set(values)) > 1
    assert all(abs(p.angle_degrees) <= 5 and abs(p.translation_x_pixels) <= 7.68
               and abs(p.translation_y_pixels) <= 7.68 for p in values)
    torch.manual_seed(42)
    output = transform(pattern())
    torch.manual_seed(42)
    assert torch.equal(output, transform(pattern()))


@pytest.mark.parametrize("kwargs", [{"max_rotation_degrees": -1},
    {"max_rotation_degrees": float("nan")}, {"max_translation_fraction": 0.9},
    {"enabled": "yes"}])
def test_invalid_augmentation_rejected(kwargs):
    with pytest.raises(ValueError):
        AugmentationConfig(**kwargs)


def test_old_experiments_disabled_and_e09_changes_only_augmentation():
    base = load_experiment_config("configs/experiments/E05_pool_between_convs.json")
    e09 = load_experiment_config("configs/experiments/E09_shared_affine.json")
    assert not base.data.augmentation.enabled
    assert build_train_transform(base.data.augmentation) is None
    assert e09.model == base.model and e09.training == base.training
    assert replace(e09.data, augmentation=base.data.augmentation) == base.data
    assert e09.output_dir != base.output_dir and e09.checkpoint_dir != base.checkpoint_dir
    assert e09.to_dict()["data"]["augmentation"]["enabled"] is True


def test_loaders_transform_train_only_and_preserve_labels(tmp_path):
    rows = create_tiny_dataset(tmp_path)
    splits = split_by_patient_fold(rows, validation_fold=1)
    transform = build_train_transform(AugmentationConfig(enabled=True))
    loaders = create_dataloaders(splits, tmp_path, LoaderConfig(batch_size=1),
                                train_transform=transform)
    assert loaders.train.dataset.transform is transform
    assert loaders.validation.dataset.transform is None
    assert loaders.test.dataset.transform is None
    assert torch.equal(loaders.validation.dataset[0]["image"], loaders.validation.dataset[0]["image"])
    # Solo imágenes sintéticas de test; no se leen píxeles del test real.
    assert torch.equal(loaders.test.dataset[0]["image"], loaders.test.dataset[0]["image"])
    item = loaders.train.dataset[0]
    assert item["patient_id"] == "P1" and item["target"].item() == 0


def test_smoke_has_same_augmentation_and_separate_outputs():
    real = load_experiment_config("configs/experiments/E09_shared_affine.json")
    smoke = load_experiment_config("configs/experiments/SMOKE_shared_affine.json")
    assert smoke.data.augmentation == real.data.augmentation
    assert smoke.model == real.model
    assert smoke.training.epochs == 2
    assert smoke.data.train_patients_per_class == 4
    assert smoke.data.validation_patients_per_class == 2
    assert smoke.output_dir != real.output_dir
