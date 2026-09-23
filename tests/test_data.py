"""Pruebas del pipeline unico PRE/EARLY/LATE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from src.data import (
    BreastDCEDataset,
    DataValidationError,
    LoaderConfig,
    create_dataloaders,
    balanced_patient_subset,
    decode_phase_png,
    load_dce_triplet,
    load_samples,
    resolve_inside_root,
    split_by_patient_fold,
)


def save_gray(path: Path, value: int, size: tuple[int, int] = (256, 256)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(size, value, dtype=np.uint8)).save(path)


def create_tiny_dataset(root: Path) -> pd.DataFrame:
    rows = []
    definitions = [
        ("P1", "train", 0, 0),
        ("P2", "train", 1, 1),
        ("P3", "test", -1, 0),
    ]
    for patient, split, fold, label in definitions:
        sample = f"{patient}_z001"
        base = Path("dataset") / split / patient
        paths = {}
        for phase, value in zip(("PRE", "EARLY", "LATE"), (10, 20, 30)):
            relative = base / f"{sample}_{phase}.png"
            save_gray(root / relative, value)
            paths[phase] = str(relative)
        rows.append(
            {
                "sample_id": sample,
                "patient_id": patient,
                "split": split,
                "path_pre": paths["PRE"],
                "path_early": paths["EARLY"],
                "path_late": paths["LATE"],
                "slice_index": 1,
                "pCR": label,
                "fold": fold,
            }
        )
    return pd.DataFrame(rows)


def phase_paths(root: Path) -> dict[str, Path]:
    paths = {}
    for phase, value in zip(("PRE", "EARLY", "LATE"), (10, 20, 30)):
        path = root / f"sample_{phase}.png"
        save_gray(path, value)
        paths[phase] = path
    return paths


def test_triplet_has_fixed_phase_order_shape_dtype_and_range(tmp_path: Path) -> None:
    image = load_dce_triplet(phase_paths(tmp_path))
    assert image.shape == (3, 256, 256)
    assert image.dtype == torch.float32
    assert image.is_contiguous()
    assert torch.allclose(image[:, 0, 0], torch.tensor([10, 20, 30]) / 255)
    assert 0 <= float(image.min()) <= float(image.max()) <= 1


def test_file_paths_and_uploaded_bytes_produce_identical_tensor(tmp_path: Path) -> None:
    paths = phase_paths(tmp_path)
    uploaded = {phase: path.read_bytes() for phase, path in paths.items()}
    assert torch.equal(load_dce_triplet(paths), load_dce_triplet(uploaded))


def test_invalid_image_mode_and_size_are_rejected(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(rgb)
    with pytest.raises(DataValidationError, match="Modo invalido"):
        decode_phase_png(rgb)

    wrong_size = tmp_path / "wrong.png"
    save_gray(wrong_size, 0, size=(128, 128))
    with pytest.raises(DataValidationError, match="Tamano invalido"):
        decode_phase_png(wrong_size)


def test_missing_or_extra_phases_are_rejected(tmp_path: Path) -> None:
    paths = phase_paths(tmp_path)
    paths.pop("LATE")
    with pytest.raises(DataValidationError, match="ausentes"):
        load_dce_triplet(paths)


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="sale de la raiz"):
        resolve_inside_root(tmp_path / "safe", "../outside.png")


def test_dataset_applies_transform_once_to_complete_triplet(tmp_path: Path) -> None:
    rows = create_tiny_dataset(tmp_path).iloc[[0]]
    observed_shapes: list[tuple[int, ...]] = []

    def transform(image: torch.Tensor) -> torch.Tensor:
        observed_shapes.append(tuple(image.shape))
        return image

    item = BreastDCEDataset(rows, tmp_path, transform=transform)[0]
    assert observed_shapes == [(3, 256, 256)]
    assert item["image"].shape == (3, 256, 256)
    assert item["patient_id"] == "P1"
    assert item["target"].dtype == torch.float32


def test_splits_and_loader_sampling_are_correct(tmp_path: Path) -> None:
    rows = create_tiny_dataset(tmp_path)
    splits = split_by_patient_fold(rows, validation_fold=1)
    assert set(splits.train.patient_id) == {"P1"}
    assert set(splits.validation.patient_id) == {"P2"}
    assert set(splits.test.patient_id) == {"P3"}

    loaders = create_dataloaders(
        splits,
        tmp_path,
        LoaderConfig(batch_size=1, num_workers=0, seed=42),
    )
    assert isinstance(loaders.train.sampler, RandomSampler)
    assert isinstance(loaders.validation.sampler, SequentialSampler)
    assert isinstance(loaders.test.sampler, SequentialSampler)
    assert next(iter(loaders.train))["image"].shape == (1, 3, 256, 256)


def test_real_public_split_is_disjoint_by_patient() -> None:
    samples = load_samples("breastdcedl")
    splits = split_by_patient_fold(samples, validation_fold=0)
    train = set(splits.train.patient_id)
    validation = set(splits.validation.patient_id)
    test = set(splits.test.patient_id)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_balanced_subset_selects_complete_patients_reproducibly() -> None:
    rows = pd.DataFrame(
        [
            {"patient_id": f"P{label}{patient}", "pCR": label, "cut": cut}
            for label in (0, 1)
            for patient in range(3)
            for cut in range(2)
        ]
    )
    first = balanced_patient_subset(rows, patients_per_class=2, seed=42)
    second = balanced_patient_subset(rows, patients_per_class=2, seed=42)
    assert first.equals(second)
    assert first["patient_id"].nunique() == 4
    assert first.groupby("pCR")["patient_id"].nunique().to_dict() == {0: 2, 1: 2}
    assert (first.groupby("patient_id").size() == 2).all()
