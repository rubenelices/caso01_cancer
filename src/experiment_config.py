"""Configuracion tipada y serializable de experimentos."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.architectures import BaseCNNConfig
from src.augmentation import AugmentationConfig, IntensityAugmentationConfig
from src.learning_rate_schedule import LRScheduleConfig
from src.mixup import MixupConfig
from src.sharpness_step import SAMConfig
from src.weight_average import EMAConfig
from src.horizontal_flip import HorizontalFlipConfig
from src.label_smoothing import LabelSmoothingConfig


@dataclass(frozen=True)
class DataExperimentConfig:
    root: str = "breastdcedl"
    validation_fold: int = 0
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    train_patients_per_batch: int | None = None
    train_patients_per_class: int | None = None
    validation_patients_per_class: int | None = None
    # Fracción científica por paciente, distinta del subconjunto técnico equilibrado.
    train_patient_fraction: float = 1.0
    patient_subset_seed: int = 42
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    intensity_augmentation: IntensityAugmentationConfig = field(default_factory=IntensityAugmentationConfig)
    horizontal_flip: HorizontalFlipConfig = field(default_factory=HorizontalFlipConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.horizontal_flip, HorizontalFlipConfig):
            raise ValueError("horizontal_flip debe ser HorizontalFlipConfig")
        if (type(self.train_patient_fraction) not in (int, float)
                or not math.isfinite(self.train_patient_fraction)
                or not 0 < self.train_patient_fraction <= 1):
            raise ValueError("train_patient_fraction debe estar en (0, 1]")
        if type(self.patient_subset_seed) is not int or self.patient_subset_seed < 0:
            raise ValueError("patient_subset_seed debe ser un entero >= 0")
        if not isinstance(self.root, str) or not self.root.strip():
            raise ValueError("root debe ser una ruta no vacía")
        if type(self.validation_fold) is not int or not 0 <= self.validation_fold <= 4:
            raise ValueError("validation_fold debe ser un entero entre 0 y 4")
        for name, minimum in (("batch_size", 1), ("num_workers", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} debe ser un entero >= {minimum}")
        for name, value in (
            ("train_patients_per_batch", self.train_patients_per_batch),
            ("train_patients_per_class", self.train_patients_per_class),
            ("validation_patients_per_class", self.validation_patients_per_class),
        ):
            if value is not None and (type(value) is not int or value < 1):
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
    objective: str = "cut"
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    mixup: MixupConfig = field(default_factory=MixupConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    label_smoothing: LabelSmoothingConfig = field(default_factory=LabelSmoothingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.label_smoothing, LabelSmoothingConfig):
            raise ValueError("label_smoothing debe ser LabelSmoothingConfig")
        if not isinstance(self.ema, EMAConfig):
            raise ValueError("ema debe ser EMAConfig")
        if not isinstance(self.sam, SAMConfig):
            raise ValueError("sam debe ser SAMConfig")
        if not isinstance(self.mixup, MixupConfig):
            raise ValueError("mixup debe ser MixupConfig")
        if self.label_smoothing.enabled and (self.mixed_precision or self.objective != "cut"
                or self.loss != "normal" or self.mixup.enabled or self.sam.enabled or self.ema.enabled):
            raise ValueError("Suavizado requiere BCE normal por corte, float32 sin AMP/Mixup/SAM/EMA")
        if self.ema.enabled and (self.mixed_precision or self.sam.enabled
                or self.mixup.enabled or self.objective != "cut"):
            raise ValueError("EMA requiere float32 sin AMP/SAM/Mixup y objetivo cut")
        if self.sam.enabled and (self.mixed_precision or self.objective != "cut" or self.mixup.enabled):
            raise ValueError("SAM requiere float32 sin AMP, objetivo cut y sin Mixup")
        if not isinstance(self.lr_schedule, LRScheduleConfig):
            raise ValueError("lr_schedule debe ser LRScheduleConfig")
        for name, minimum in (("epochs", 1), ("early_stopping_patience", 1),
                              ("scheduler_patience", 0), ("seed", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} debe ser un entero >= {minimum}")
        for name in ("learning_rate", "weight_decay", "threshold", "scheduler_factor"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} debe ser un número finito")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo")
        if self.weight_decay < 0:
            raise ValueError("weight_decay no puede ser negativo")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold debe estar en [0, 1]")
        if not 0 < self.scheduler_factor < 1:
            raise ValueError("scheduler_factor debe estar en (0, 1)")
        if self.loss not in {"normal", "weighted"}:
            raise ValueError("loss debe ser 'normal' o 'weighted'")
        if self.monitor not in {"patient_roc_auc", "patient_pr_auc"}:
            raise ValueError("monitor no reconocido")
        if self.objective not in {"cut", "patient_mean_probability"}:
            raise ValueError("objective no reconocido")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    description: str
    data: DataExperimentConfig
    model: BaseCNNConfig
    training: TrainingConfig
    output_dir: str
    checkpoint_dir: str

    def __post_init__(self) -> None:
        if self.training.label_smoothing.enabled and self.data.train_patients_per_batch is not None:
            raise ValueError("Suavizado solo está comprobado con lotes de cortes")
        if self.training.ema.enabled and self.data.train_patients_per_batch is not None:
            raise ValueError("EMA solo está comprobada con lotes de cortes")
        if self.training.sam.enabled and self.data.train_patients_per_batch is not None:
            raise ValueError("SAM solo está comprobado con lotes de cortes, no bolsas de pacientes")
        if self.training.mixup.enabled and (
                self.training.objective != "cut"
                or self.data.train_patients_per_batch is not None):
            raise ValueError("Mixup requiere BCE por corte y lotes de cortes, no bolsas de pacientes")
        if (self.training.objective == "patient_mean_probability"
                and self.data.train_patients_per_batch is None):
            raise ValueError("El objetivo por paciente requiere lotes de pacientes completas")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return experiment_config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def experiment_config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Mismo contrato para JSON y configuraciones guardadas en checkpoints."""
    data_raw = dict(raw["data"])
    augmentation = AugmentationConfig(**data_raw.pop("augmentation", {}))
    intensity = IntensityAugmentationConfig(**data_raw.pop("intensity_augmentation", {}))
    horizontal = HorizontalFlipConfig(**data_raw.pop("horizontal_flip", {}))
    data = DataExperimentConfig(**data_raw, augmentation=augmentation, intensity_augmentation=intensity, horizontal_flip=horizontal)
    model_raw = dict(raw["model"])
    model_raw["channels"] = tuple(model_raw["channels"])
    model = BaseCNNConfig(**model_raw)
    training_raw = dict(raw["training"])
    schedule = LRScheduleConfig(**training_raw.pop("lr_schedule", {}))
    mixup = MixupConfig(**training_raw.pop("mixup", {}))
    sam = SAMConfig(**training_raw.pop("sam", {}))
    ema = EMAConfig(**training_raw.pop("ema", {}))
    smoothing = LabelSmoothingConfig(**training_raw.pop("label_smoothing", {}))
    training = TrainingConfig(**training_raw, lr_schedule=schedule, mixup=mixup, sam=sam, ema=ema,
                              label_smoothing=smoothing)
    return ExperimentConfig(
        experiment_id=raw["experiment_id"],
        description=raw["description"],
        data=data,
        model=model,
        training=training,
        output_dir=raw["output_dir"],
        checkpoint_dir=raw["checkpoint_dir"],
    )
