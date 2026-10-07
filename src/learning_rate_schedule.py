"""Calendario por época; no modifica arquitectura, datos ni función de pérdida."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class LRScheduleConfig:
    policy: str = "plateau"
    warmup_epochs: int = 0
    cosine_epochs: int = 0
    start_factor: float = 0.1
    min_factor: float = 0.01

    def __post_init__(self) -> None:
        if self.policy not in {"plateau", "warmup_cosine"}:
            raise ValueError("lr_schedule.policy no reconocida")
        for name in ("warmup_epochs", "cosine_epochs"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"lr_schedule.{name} debe ser un entero >= 0")
        for name in ("start_factor", "min_factor"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"lr_schedule.{name} debe ser finito y estar en (0, 1)")
        if self.policy == "plateau" and (self.warmup_epochs or self.cosine_epochs):
            raise ValueError("lr_schedule plateau no usa warmup_epochs/cosine_epochs")
        if self.policy == "warmup_cosine" and (self.warmup_epochs < 2 or self.cosine_epochs < 1):
            raise ValueError("lr_schedule requiere warmup_epochs >= 2 y cosine_epochs >= 1")


def rate_for_epoch(base_rate: float, schedule: LRScheduleConfig, epoch: int) -> float:
    """LR realmente usado: warmup lineal, descenso coseno, luego suelo fijo.

    Épocas 1..W: start_factor→1, ambos extremos incluidos.
    Épocas W+1..W+C: coseno con progreso 1/C..1, sin reinicios.
    Épocas posteriores: min_factor. No lee métricas ni aumenta LR de nuevo.
    """
    if type(base_rate) not in (int, float) or not math.isfinite(base_rate) or base_rate <= 0:
        raise ValueError("base_rate debe ser positivo y finito")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch debe ser un entero >= 1")
    if schedule.policy != "warmup_cosine":
        raise ValueError("plateau depende de métricas, no tiene calendario prefijado")
    if epoch <= schedule.warmup_epochs:
        progress = (epoch - 1) / (schedule.warmup_epochs - 1)
        factor = schedule.start_factor + (1 - schedule.start_factor) * progress
    else:
        progress = min((epoch - schedule.warmup_epochs) / schedule.cosine_epochs, 1)
        factor = schedule.min_factor + (1 - schedule.min_factor) * (1 + math.cos(math.pi * progress)) / 2
    return base_rate * factor


class LearningRateSchedule:
    """Conserva exactamente ReduceLROnPlateau para configuraciones antiguas."""

    def __init__(self, optimizer: torch.optim.Optimizer, training) -> None:
        self.optimizer = optimizer
        self.training = training
        self.schedule = training.lr_schedule
        self.epoch = 0
        self.plateau = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=training.scheduler_factor,
                patience=training.scheduler_patience,
            ) if self.schedule.policy == "plateau" else None
        )

    def start_epoch(self, epoch: int) -> float:
        if type(epoch) is not int or epoch != self.epoch + 1:
            raise ValueError("El calendario exige épocas consecutivas desde 1")
        self.epoch = epoch
        if self.plateau is None:
            rate = rate_for_epoch(self.training.learning_rate, self.schedule, epoch)
            for group in self.optimizer.param_groups:
                group["lr"] = rate
        return float(self.optimizer.param_groups[0]["lr"])

    def finish_epoch(self, metric: float) -> None:
        if self.plateau is not None:
            self.plateau.step(metric)

    def state_dict(self) -> dict:
        return {"schedule": asdict(self.schedule), "epoch": self.epoch,
                "plateau": self.plateau.state_dict() if self.plateau is not None else None}


def main() -> None:
    """Dibuja el calendario previsto sin datos, inferencia ni entrenamiento."""
    import argparse
    from pathlib import Path
    import matplotlib.pyplot as plt
    from src.experiment_config import load_experiment_config

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Prefijo para PNG/SVG")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    epochs = list(range(1, cfg.training.epochs + 1))
    rates = [rate_for_epoch(cfg.training.learning_rate, cfg.training.lr_schedule, e) for e in epochs]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(epochs, rates, marker=".", color="#345d8c")
    ax.set(xlabel="Época", ylabel="Learning rate usado", title=f"{cfg.experiment_id} · Calendario previsto (no resultado)")
    ax.axvline(cfg.training.lr_schedule.warmup_epochs, linestyle="--", color="#888888", label="Fin warmup")
    ax.axvline(cfg.training.lr_schedule.warmup_epochs + cfg.training.lr_schedule.cosine_epochs,
               linestyle=":", color="#888888", label="Inicio del suelo fijo")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=.2)
    ax.legend()
    fig.text(.02, .01, "Early stopping puede acortar la ejecución. Uso educativo, no clínico.", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=140)
    fig.savefig(args.output.with_suffix(".svg"), metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
