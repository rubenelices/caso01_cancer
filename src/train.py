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
from src.augmentation import build_train_transform
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
from src.patient_training import objective_loss, patient_level_pos_weight
from src.learning_rate_schedule import LearningRateSchedule
from src.mixup import MixupConfig, apply_mixup
from src.sharpness_step import SAMConfig, sam_step
from src.weight_average import ExponentialModelAverage
from src.label_smoothing import LabelSmoothingConfig, smooth_binary_targets
from src.patient_subsets import select_training_patients, selection_hash
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
    weight = (patient_level_pos_weight(train_rows)
              if config.training.objective == "patient_mean_probability"
              else cut_level_pos_weight(train_rows))
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
    objective: str = "cut",
    mixup: MixupConfig | None = None,
    sam: SAMConfig | None = None,
    ema: ExponentialModelAverage | None = None,
    label_smoothing: LabelSmoothingConfig | None = None,
) -> float:
    use_sam = sam is not None and sam.enabled
    if label_smoothing is not None and label_smoothing.enabled:
        if (use_amp or use_sam or ema is not None or objective != "cut"
                or (mixup is not None and mixup.enabled)
                or not isinstance(criterion, nn.BCEWithLogitsLoss)
                or criterion.pos_weight is not None or criterion.weight is not None):
            raise ValueError("Suavizado requiere BCE normal por corte, float32 sin AMP/Mixup/SAM/EMA")
    if ema is not None and (use_amp or use_sam or objective != "cut"
            or (mixup is not None and mixup.enabled)):
        raise ValueError("EMA requiere float32 sin AMP/SAM/Mixup y objetivo cut")
    if use_sam and (use_amp or objective != "cut" or (mixup is not None and mixup.enabled)):
        raise ValueError("SAM requiere float32 sin AMP, objetivo cut y sin Mixup")
    if mixup is not None and mixup.enabled and objective != "cut":
        raise ValueError("Mixup solo está implementado para la pérdida por corte")
    model.train()
    loss_sum = 0.0
    samples_seen = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        images, targets = apply_mixup(images, targets, mixup)
        if label_smoothing is not None:
            targets = smooth_binary_targets(targets, label_smoothing)
        if use_sam:
            # Dos pasadas del MISMO lote, pero una única actualización AdamW.
            # Registrar la primera loss, no la calculada en pesos perturbados.
            loss = sam_step(model, optimizer, lambda: criterion(model(images), targets), sam.rho)
            units = targets.numel()
        else:
            # Camino anterior intacto: sin snapshots ni consumo RNG adicional.
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss, units = objective_loss(logits, targets, batch["patient_id"], criterion, objective)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
        loss_sum += float(loss.detach().cpu()) * units
        samples_seen += units
    return loss_sum / samples_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    use_amp: bool,
    objective: str = "cut",
) -> tuple[float, dict[str, Any]]:
    model.eval()
    loss_sum = 0.0
    samples_seen = 0
    probabilities: list[float] = []
    targets_all: list[float] = []
    patient_ids: list[str] = []
    logits_all: list[float] = []

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
        if objective == "patient_mean_probability":
            logits_all.extend(logits.float().cpu().tolist())

    metrics = cut_and_patient_metrics(
        patient_ids,
        targets_all,
        probabilities,
        threshold=threshold,
        aggregation="mean",
    )
    if objective == "patient_mean_probability":
        # Los lotes secuenciales de evaluación pueden cortar una paciente.
        # Agregar al terminar la pasada, nunca calcular pérdidas de bolsas parciales.
        loss, _ = objective_loss(
            torch.tensor(logits_all, dtype=torch.float32, device=device),
            torch.tensor(targets_all, dtype=torch.float32, device=device),
            patient_ids, criterion, objective,
        )
        return float(loss.cpu()), metrics
    if objective != "cut":
        raise ValueError("Objetivo de evaluación no reconocido")
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
    splits, patient_selection = select_training_patients(
        splits, config.data.root, config.data.train_patient_fraction, config.data.patient_subset_seed)
    splits = limit_splits_for_debug(
        splits,
        config.data.train_patients_per_class,
        config.data.validation_patients_per_class,
        config.training.seed,
    )
    patient_selection.update(effective_train_patients=splits.patient_counts()['train'],
                             effective_train_cuts=len(splits.train),
                             effective_selection_sha256=selection_hash(splits.train))
    loader_config = LoaderConfig(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and device.type == "cuda",
        persistent_workers=config.data.num_workers > 0,
        seed=config.training.seed,
        train_patients_per_batch=config.data.train_patients_per_batch,
    )
    loaders = create_dataloaders(
        splits, config.data.root, loader_config,
        train_transform=build_train_transform(config.data.augmentation, config.data.intensity_augmentation, config.data.horizontal_flip),
    )

    model = BreastPCRNet(config.model).to(device)
    ema = (ExponentialModelAverage(model, config.training.ema.decay)
           if config.training.ema.enabled else None)
    criterion, positive_weight = build_criterion(config, splits.train, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = LearningRateSchedule(optimizer, config.training)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    resolved = config.to_dict()
    (output_dir / "config_resolved.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    environment = _environment(device)
    (output_dir / "train_patient_selection.json").write_text(
        json.dumps(patient_selection, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.perf_counter()
    loss_unit = "patient" if config.training.objective == "patient_mean_probability" else "cut"
    if config.data.train_patients_per_batch is not None:
        loss_unit_label = "paciente" if loss_unit == "patient" else "corte"
        print(f"Train: {config.data.train_patients_per_batch} pacientes completas/lote; "
              f"{len(loaders.train)} pasos/época; pérdida por {loss_unit_label}.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.training.epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate_used = scheduler.start_epoch(epoch)
        train_loss = train_one_epoch(
            model, loaders.train, criterion, optimizer, scaler, device, use_amp,
            objective=config.training.objective,
            mixup=config.training.mixup,
            sam=config.training.sam,
            ema=ema,
            label_smoothing=config.training.label_smoothing,
        )
        validation_loss, metrics = evaluate(
            ema.model if ema is not None else model,
            loaders.validation,
            criterion,
            device,
            config.training.threshold,
            use_amp,
            objective=config.training.objective,
        )
        monitor_key = config.training.monitor.removeprefix("patient_")
        monitor_value = float(metrics["patient"][monitor_key])
        scheduler.finish_epoch(monitor_value)
        epoch_seconds = time.perf_counter() - epoch_start
        row = _history_row(
            epoch,
            train_loss,
            validation_loss,
            optimizer.param_groups[0]["lr"],
            epoch_seconds,
            metrics,
        )
        # La columna histórica learning_rate sigue indicando LR tras scheduler.
        # Esta nueva columna distingue el LR realmente aplicado en la época.
        row["learning_rate_used"] = learning_rate_used
        history.append(row)
        _write_history(history_path, history)

        checkpoint = {
            "experiment_id": config.experiment_id,
            "epoch": epoch,
            "model_state_dict": (ema.model if ema is not None else model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": resolved,
            "validation_metrics": metrics,
        }
        if ema is not None:
            checkpoint.update(trainable_model_state_dict=model.state_dict(),
                              ema_state_dict=ema.state_dict(),
                              evaluated_weights="ema")
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
            f"{metrics['patient']['pr_auc']:.4f} lr={learning_rate_used:.6g} time={epoch_seconds:.1f}s"
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
        objective=config.training.objective,
    )
    # Se hace una segunda pasada secuencial para conservar sample_id y
    # patient_id. Test permanece cerrado y nunca se itera en este runner.
    from src.evaluation import (
        EvaluationConfig,
        collect_cut_predictions,
        save_evaluation_artifacts,
    )

    validation_predictions = collect_cut_predictions(
        model,
        loaders.validation,
        device,
        use_amp=use_amp,
    )
    evaluation_summary = save_evaluation_artifacts(
        validation_predictions,
        output_dir,
        EvaluationConfig(
            threshold=config.training.threshold,
            aggregation="mean",
            seed=config.training.seed,
        ),
    )
    evaluation_summary["experiment_id"] = config.experiment_id
    evaluation_summary["validation_fold"] = config.data.validation_fold
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    training_diagnostics = build_training_diagnostics(history, final_metrics, best_epoch)
    training_target_mode = ("smoothed_binary" if config.training.label_smoothing.enabled
                            else "mixup_soft" if config.training.mixup.enabled else "binary")
    if config.training.label_smoothing.enabled:
        training_diagnostics["loss_comparison_note"] = (
            "Train BCE con objetivos suavizados; val BCE con etiquetas binarias reales. "
            "Las pérdidas no son el mismo objetivo; las métricas conservan etiquetas reales.")
    if ema is not None:
        training_diagnostics["loss_comparison_note"] = (
            "Train loss: pesos entrenables con dropout/BN de train; "
            "val loss y métricas: copia EMA en eval. No son los mismos pesos.")
    save_training_dashboard(
        history,
        final_metrics,
        best_epoch,
        output_dir / "training_curves.png",
        evaluated_weights="ema" if ema is not None else "trainable",
        training_target_mode=training_target_mode,
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
        "loss_unit": loss_unit,
        "training_target_mode": training_target_mode,
        "training_steps_per_epoch": len(loaders.train),
        "train_patients_per_batch": config.data.train_patients_per_batch,
        "train_patient_selection": patient_selection,
        "validation_loss": final_validation_loss,
        "validation_metrics": final_metrics,
        "evaluation_artifacts": {
            "cut_predictions": "predictions_cut.csv",
            "patient_predictions": "predictions_patient.csv",
            "summary": "evaluation_summary.json",
            "figure": "evaluation_patient.png",
        },
        "training_diagnostics": training_diagnostics,
        "patients": splits.patient_counts(),
        "samples": splits.sample_counts(),
        "test_evaluated": False,
        "environment": environment,
    }
    if ema is not None:
        summary.update(evaluated_weights="ema", ema_decay=ema.config.decay,
                       ema_updates=ema.updates,
                       train_loss_weights="trainable", validation_loss_weights="ema",
                       best_ema_updates=best_checkpoint["ema_state_dict"]["updates"])
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
