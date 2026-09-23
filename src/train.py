"""Entrenamiento reproducible de experimentos CNN sobre validacion interna."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.architectures import BreastPCRNet, parameter_breakdown, trainable_parameter_count
from src.data import (
    LoaderConfig,
    create_dataloaders,
    cut_level_pos_weight,
    limit_splits_for_debug,
    load_samples,
    split_by_patient_fold,
)
from src.experiment_config import ExperimentConfig, load_experiment_config
from src.metrics import cut_and_patient_metrics
from src.training_report import (
    build_training_diagnostics,
    early_learning_signal,
    save_diagnostics_json,
    save_training_dashboard,
)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_criterion(
    config: ExperimentConfig,
    train_rows: Any,
    device: torch.device,
) -> tuple[nn.Module, float | None]:
    if config.training.loss == "normal":
        return nn.BCEWithLogitsLoss(), None
    weight = cut_level_pos_weight(train_rows)
    tensor = torch.tensor(weight, dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=tensor), weight


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    loss_sum = 0.0
    samples_seen = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch_size
        samples_seen += batch_size
    return loss_sum / samples_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    use_amp: bool,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    loss_sum = 0.0
    samples_seen = 0
    probabilities: list[float] = []
    targets_all: list[float] = []
    patient_ids: list[str] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        batch_probabilities = torch.sigmoid(logits)
        batch_size = images.shape[0]
        loss_sum += float(loss.detach().cpu()) * batch_size
        samples_seen += batch_size
        probabilities.extend(batch_probabilities.float().cpu().tolist())
        targets_all.extend(targets.float().cpu().tolist())
        patient_ids.extend(list(batch["patient_id"]))

    metrics = cut_and_patient_metrics(
        patient_ids,
        targets_all,
        probabilities,
        threshold=threshold,
        aggregation="mean",
    )
    return loss_sum / samples_seen, metrics


def _history_row(
    epoch: int,
    train_loss: float,
    validation_loss: float,
    learning_rate: float,
    elapsed_seconds: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "learning_rate": learning_rate,
        "elapsed_seconds": elapsed_seconds,
    }
    for level in ("cut", "patient"):
        for key, value in metrics[level].items():
            if key != "confusion_matrix":
                row[f"{level}_{key}"] = value
        for key, value in metrics[level]["confusion_matrix"].items():
            row[f"{level}_{key}"] = value
    return row


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return result


def run_experiment(
    config: ExperimentConfig,
    *,
    device_name: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Entrena y valida; nunca itera el DataLoader de test."""

    output_dir = Path(config.output_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    history_path = output_dir / "history.csv"
    if history_path.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {history_path}. Usa --overwrite solo si quieres reemplazarlo."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_reproducibility(config.training.seed)
    device = choose_device(device_name)
    use_amp = config.training.mixed_precision and device.type == "cuda"
    samples = load_samples(config.data.root)
    splits = split_by_patient_fold(samples, config.data.validation_fold)
    splits = limit_splits_for_debug(
        splits,
        config.data.train_patients_per_class,
        config.data.validation_patients_per_class,
        config.training.seed,
    )
    loader_config = LoaderConfig(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
        seed=config.training.seed,
    )
    loaders = create_dataloaders(splits, config.data.root, loader_config)

    model = BreastPCRNet(config.model).to(device)
    criterion, positive_weight = build_criterion(config, splits.train, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    resolved = config.to_dict()
    (output_dir / "config_resolved.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    environment = _environment(device)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.training.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model, loaders.train, criterion, optimizer, scaler, device, use_amp
        )
        validation_loss, metrics = evaluate(
            model,
            loaders.validation,
            criterion,
            device,
            config.training.threshold,
            use_amp,
        )
        monitor_key = config.training.monitor.removeprefix("patient_")
        monitor_value = float(metrics["patient"][monitor_key])
        scheduler.step(monitor_value)
        epoch_seconds = time.perf_counter() - epoch_start
        row = _history_row(
            epoch,
            train_loss,
            validation_loss,
            optimizer.param_groups[0]["lr"],
            epoch_seconds,
            metrics,
        )
        history.append(row)
        _write_history(history_path, history)

        checkpoint = {
            "experiment_id": config.experiment_id,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": resolved,
            "validation_metrics": metrics,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if monitor_value > best_score:
            best_score = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} patient_roc_auc="
            f"{metrics['patient']['roc_auc']:.4f} patient_pr_auc="
            f"{metrics['patient']['pr_auc']:.4f} time={epoch_seconds:.1f}s"
        )
        if epoch == 5:
            print(f"Diagnostico epoca 5: {early_learning_signal(history)['message']}")
        if epochs_without_improvement >= config.training.early_stopping_patience:
            print("Early stopping: validacion sin mejora.")
            break

    total_seconds = time.perf_counter() - training_start
    best_checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_validation_loss, final_metrics = evaluate(
        model,
        loaders.validation,
        criterion,
        device,
        config.training.threshold,
        use_amp,
    )
    training_diagnostics = build_training_diagnostics(history, final_metrics, best_epoch)
    save_training_dashboard(
        history,
        final_metrics,
        best_epoch,
        output_dir / "training_curves.png",
    )
    save_diagnostics_json(
        training_diagnostics,
        output_dir / "training_diagnostics.json",
    )

    summary = {
        "status": "complete",
        "experiment_id": config.experiment_id,
        "description": config.description,
        "debug_patient_subset": (
            config.data.train_patients_per_class is not None
            or config.data.validation_patients_per_class is not None
        ),
        "scientific_result": (
            config.data.train_patients_per_class is None
            and config.data.validation_patients_per_class is None
        ),
        "best_epoch": best_epoch,
        "best_monitor_value": best_score,
        "monitor": config.training.monitor,
        "epochs_completed": len(history),
        "total_seconds": total_seconds,
        "seconds_per_epoch_mean": total_seconds / len(history),
        "gpu_peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "parameter_count": trainable_parameter_count(model),
        "parameter_breakdown": parameter_breakdown(model),
        "positive_weight": positive_weight,
        "validation_loss": final_validation_loss,
        "validation_metrics": final_metrics,
        "training_diagnostics": training_diagnostics,
        "patients": splits.patient_counts(),
        "samples": splits.sample_counts(),
        "test_evaluated": False,
        "environment": environment,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    summary = run_experiment(config, device_name=args.device, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
