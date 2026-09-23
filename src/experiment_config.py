"""Configuracion tipada y serializable de experimentos."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.architectures import BaseCNNConfig


@dataclass(frozen=True)
class DataExperimentConfig:
    root: str = "breastdcedl"
    validation_fold: int = 0
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    train_patients_per_class: int | None = None
    validation_patients_per_class: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("train_patients_per_class", self.train_patients_per_class),
            ("validation_patients_per_class", self.validation_patients_per_class),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} debe ser positivo o null")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "normal"
    threshold: float = 0.5
    seed: int = 42
    early_stopping_patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    monitor: str = "patient_roc_auc"
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        if self.loss not in {"normal", "weighted"}:
            raise ValueError("loss debe ser 'normal' o 'weighted'")
        if self.monitor not in {"patient_roc_auc", "patient_pr_auc"}:
            raise ValueError("monitor no reconocido")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    description: str
    data: DataExperimentConfig
    model: BaseCNNConfig
    training: TrainingConfig
    output_dir: str
    checkpoint_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = DataExperimentConfig(**raw["data"])
    model_raw = dict(raw["model"])
    model_raw["channels"] = tuple(model_raw["channels"])
    model = BaseCNNConfig(**model_raw)
    training = TrainingConfig(**raw["training"])
    return ExperimentConfig(
        experiment_id=raw["experiment_id"],
        description=raw["description"],
        data=data,
        model=model,
        training=training,
        output_dir=raw["output_dir"],
        checkpoint_dir=raw["checkpoint_dir"],
    )
