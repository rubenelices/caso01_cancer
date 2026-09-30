"""Diagnóstico de train/validación y BatchNorm, sin entrenamiento ni test."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from src.architectures import BreastPCRNet
from src.data import (BreastDCEDataset, load_samples, split_by_patient_fold,
                      limit_splits_for_debug, seed_worker)
from src.experiment_config import experiment_config_from_dict
from src.metrics import aggregate_by_patient, binary_metrics
from src.train import build_criterion, choose_device, set_reproducibility


DEFAULT_CHECKPOINT = Path("checkpoints/E05_pool_between_convs/fold_0_seed_42/best.pt")
NOTICE = "Diagnóstico educativo y de investigación, sin validez clínica. No selecciona un modelo automáticamente."


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_internal_loader(loader: DataLoader, *, validation_fold: int | None = None) -> None:
    """Rechaza test y aumentos antes de leer cualquier píxel."""
    if not isinstance(loader.dataset, BreastDCEDataset):
        raise ValueError("Se exige el Dataset compartido del proyecto")
    rows = loader.dataset.rows
    if rows.empty or set(rows["split"]) != {"train"}:
        raise ValueError("Solo se permiten pacientes del train público; test está cerrado")
    if loader.dataset.transform is not None or loader.drop_last:
        raise ValueError("Diagnóstico sin aumentos ni drop_last")
    if validation_fold is not None and (rows["fold"] == validation_fold).any():
        raise ValueError("No se permite recalibrar BatchNorm con validación")


@torch.no_grad()
def collect_predictions(model, loader, criterion, device, *, use_amp=False):
    validate_internal_loader(loader)
    model.eval()  # Dropout apagado y BatchNorm con estadísticas fijas.
    records, loss_sum, count = [], 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        if logits.shape != targets.shape or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise ValueError("Logits/loss no finitos o con forma inválida")
        probabilities = torch.sigmoid(logits.float()).cpu().tolist()
        loss_sum += loss.item() * len(targets)
        count += len(targets)
        for index, probability in enumerate(probabilities):
            records.append({"sample_id": str(batch["sample_id"][index]),
                "patient_id": str(batch["patient_id"][index]),
                "target": int(targets[index].item()), "probability": probability})
    cuts = pd.DataFrame(records)
    if cuts.sample_id.duplicated().any():
        raise ValueError("Cortes duplicados")
    patients = aggregate_by_patient(cuts.patient_id, cuts.target, cuts.probability, method="mean")
    return cuts, patients, loss_sum / count


@torch.no_grad()
def recalibrate_batchnorm(model, loader, device, validation_fold: int):
    """Solo copia de buffers BN; pesos idénticos y ninguna imagen de validación.

    Usa promedios acumulados por minibatch (momentum=None), no momentos exactos
    de toda la población. El resultado puede depender del batch y su composición.
    """
    validate_internal_loader(loader, validation_fold=validation_fold)
    candidate = copy.deepcopy(model).eval()
    layers = {name: module for name, module in candidate.named_modules()
              if isinstance(module, nn.BatchNorm2d)}
    weights = {name: value.detach().clone() for name, value in candidate.named_parameters()}
    for layer in layers.values():
        if not layer.track_running_stats:
            raise ValueError("El diagnóstico exige BatchNorm con running stats")
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()  # Solo BN en train; Dropout sigue apagado.
    batches = 0
    if layers:
        for batch in loader:
            candidate(batch["image"].to(device))  # FP32, sin etiquetas, gradientes ni optimizador.
            batches += 1
    candidate.eval()
    if not all(torch.equal(weights[name], value) for name, value in candidate.named_parameters()):
        raise RuntimeError("La recalibración ha cambiado parámetros aprendidos")
    before = dict(model.named_modules())
    changes = []
    for name, layer in layers.items():
        original = before[name]
        if not torch.isfinite(layer.running_mean).all() or not torch.isfinite(layer.running_var).all():
            raise ValueError("BatchNorm produjo estadísticas no finitas")
        changes.append({"layer": name, "batches": int(layer.num_batches_tracked),
            "mean_absolute_change": float((layer.running_mean - original.running_mean).abs().mean()),
            "variance_absolute_change": float((layer.running_var - original.running_var).abs().mean()),
            "old_variance_mean": float(original.running_var.mean()),
            "new_variance_mean": float(layer.running_var.mean())})
        layer.momentum = original.momentum
    return candidate, {"layers": changes, "calibration_batches": batches,
                       "weights_unchanged": True, "source": "train_only"}


def class_distributions(frame):
    result = {}
    for label in (0, 1):
        values = frame.loc[frame.target == label, "probability"]
        result[str(label)] = {"n": len(values), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "min": float(values.min()),
            "median": float(values.median()), "max": float(values.max()),
            "fraction_at_least_0_5": float((values >= 0.5).mean()) if len(values) else None}
    return result


def paired_auc_interval(original, candidate, repetitions, seed):
    """Bootstrap estratificado y pareado por paciente; no por corte."""
    if not original.patient_id.equals(candidate.patient_id) or not original.target.equals(candidate.target):
        raise ValueError("El bootstrap exige las mismas pacientes y etiquetas")
    y = original.target.to_numpy()
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    if any(len(group) == 0 for group in groups):
        return {"status": "requires_two_classes"}
    rng, differences = np.random.default_rng(seed), []
    left, right = original.probability.to_numpy(), candidate.probability.to_numpy()
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        differences.append(roc_auc_score(y[indices], right[indices]) - roc_auc_score(y[indices], left[indices]))
    return {"method": "stratified_paired_patient_bootstrap", "repetitions": repetitions,
            "confidence_level": 0.95, "lower": float(np.quantile(differences, .025)),
            "upper": float(np.quantile(differences, .975)),
            "warning": "Exploratorio en el fold de desarrollo; no corrige selección de checkpoint ni múltiples ensayos"}


def save_dashboard(frames, metrics, output, technical):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for column, split in enumerate(("train", "validation")):
        for mode, color in (("original", "#2464a0"), ("bn_recalibrated", "#bf6a22")):
            key = f"{split}_{mode}"
            frame = frames[key]
            if frame.target.nunique() == 2:
                fpr, tpr, _ = roc_curve(frame.target, frame.probability)
                axes[0, column].plot(fpr, tpr, color=color,
                    label=f"{mode}: AUC {metrics[key]['patient']['roc_auc']:.3f}")
        axes[0, column].plot([0, 1], [0, 1], "--", color="gray")
        axes[0, column].set(title=f"ROC por paciente · {split}", xlabel="1 − especificidad",
                           ylabel="Sensibilidad", xlim=(0, 1), ylim=(0, 1))
        axes[0, column].legend(fontsize=9)
        frame = frames[f"{split}_original"]
        low, high = float(frame.probability.min()), float(frame.probability.max())
        margin = max((high - low) * .1, .005)
        low, high = max(0, low - margin), min(1, high + margin)
        for label, color in ((0, "#2464a0"), (1, "#bf6a22")):
            axes[1, column].hist(frame.loc[frame.target == label, "probability"],
                bins=np.linspace(low, high, 21), alpha=.5, color=color, label=f"pCR={label}", density=True)
        if low <= .5 <= high:
            axes[1, column].axvline(.5, color="black", linestyle="--", label="Umbral 0,5")
        else:
            axes[1, column].text(.03, .93, "Umbral 0,5 fuera del rango mostrado",
                                transform=axes[1, column].transAxes, fontsize=9)
        axes[1, column].set(title=f"Probabilidades originales · {split}",
                           xlabel="Probabilidad media por paciente (rango ampliado)", ylabel="Densidad", xlim=(low, high))
        axes[1, column].legend(fontsize=9)
    title = "DIAGNÓSTICO TÉCNICO: no estima generalización" if technical else "Diagnóstico interno: no es evaluación final"
    fig.suptitle(title + "\nSin dropout/aumentos en evaluación. Uso educativo, no clínico.", fontsize=14)
    fig.savefig(output / "diagnostic_dashboard.png", dpi=130)
    plt.close(fig)


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def run_diagnosis(checkpoint_path: Path = DEFAULT_CHECKPOINT, *, device_name="auto",
                  batch_size=None, num_workers=None, data_root=None, output_dir=None,
                  bootstrap_repetitions=500, float32=False):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No existe {checkpoint_path}; copia nuestro best.pt, no entrenes de nuevo")
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions debe ser positivo")
    digest = file_sha256(checkpoint_path)
    # Solo checkpoints propios y confiables: contienen configuración, no solo pesos.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = experiment_config_from_dict(checkpoint["config"])
    if checkpoint.get("experiment_id", config.experiment_id) != config.experiment_id:
        raise ValueError("Identidad del checkpoint inconsistente")
    output = Path(output_dir) if output_dir is not None else Path("reports/diagnostics") / config.experiment_id / digest[:12]
    if output.exists():
        raise FileExistsError(f"Ya existe {output}; usa otra carpeta para conservar el informe")
    set_reproducibility(config.training.seed)
    device = choose_device(device_name)
    use_amp = config.training.mixed_precision and device.type == "cuda" and not float32
    batch_size = config.data.batch_size if batch_size is None else batch_size
    num_workers = config.data.num_workers if num_workers is None else num_workers
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Batch positivo y workers no negativos")
    root = config.data.root if data_root is None else data_root
    splits = split_by_patient_fold(load_samples(root), config.data.validation_fold)
    splits = limit_splits_for_debug(splits, config.data.train_patients_per_class,
                                   config.data.validation_patients_per_class, config.training.seed)
    datasets = {"train": BreastDCEDataset(splits.train, root),
                "validation": BreastDCEDataset(splits.validation, root)}
    def loader(dataset, shuffle=False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config.training.seed))
    loaders = {name: loader(dataset) for name, dataset in datasets.items()}
    calibration_loader = loader(datasets["train"], shuffle=True)
    model = BreastPCRNet(config.model).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    initial_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    criterion, positive_weight = build_criterion(config, splits.train, device)
    frames, metrics = {}, {}
    output.mkdir(parents=True)
    start = time.perf_counter()
    def measure(current, mode):
        for split, current_loader in loaders.items():
            key = f"{split}_{mode}"
            print(f"Evaluando {key} sin aumentos...", flush=True)
            cuts, patients, loss = collect_predictions(current, current_loader, criterion, device, use_amp=use_amp)
            cuts.to_csv(output / f"{key}_cuts.csv", index=False)
            patients.to_csv(output / f"{key}_patients.csv", index=False)
            frames[key] = patients
            metrics[key] = {"loss_per_cut": loss,
                "cut": binary_metrics(cuts.target, cuts.probability, config.training.threshold),
                "patient": binary_metrics(patients.target, patients.probability, config.training.threshold),
                "probability_by_class": class_distributions(patients)}
            metrics[key]["patient"]["brier_score"] = float(brier_score_loss(patients.target, patients.probability))
    measure(model, "original")
    print("Recalculando estadísticas BN en una copia, solo con train...", flush=True)
    candidate, bn = recalibrate_batchnorm(model, calibration_loader, device, config.data.validation_fold)
    measure(candidate, "bn_recalibrated")
    if not all(torch.equal(initial_state[name], tensor) for name, tensor in model.state_dict().items()):
        raise RuntimeError("El diagnóstico alteró el modelo original")
    if file_sha256(checkpoint_path) != digest:
        raise RuntimeError("El checkpoint original cambió durante el diagnóstico")
    left, right = frames["validation_original"], frames["validation_bn_recalibrated"]
    delta = metrics["validation_bn_recalibrated"]["patient"]["roc_auc"] - metrics["validation_original"]["patient"]["roc_auc"]
    technical = config.data.train_patients_per_class is not None or config.data.validation_patients_per_class is not None
    comparison = {"train_minus_validation_auc_original": metrics["train_original"]["patient"]["roc_auc"] - metrics["validation_original"]["patient"]["roc_auc"],
        "validation_auc_change_after_bn": delta,
        "paired_delta_interval": paired_auc_interval(left, right, bootstrap_repetitions, config.training.seed),
        "mean_absolute_patient_probability_change": float(np.abs(left.probability - right.probability).mean()),
        "decision": "No elegir automáticamente; revisar métricas, distribuciones y estabilidad en otras semillas/folds"}
    summary = clean_json({"status": "complete", "scope": "train_and_internal_validation_only",
        "notice": NOTICE, "technical_subset": technical, "test_evaluated": False,
        "checkpoint_unchanged": True, "original_model_unchanged": True,
        "checkpoint": str(checkpoint_path), "sha256": digest, "checkpoint_epoch": checkpoint.get("epoch"),
        "experiment_id": config.experiment_id, "validation_fold": config.data.validation_fold,
        "patients": {name: dataset.rows.patient_id.nunique() for name, dataset in datasets.items()},
        "config": config.to_dict(), "data_root_used": str(root), "batch_size": batch_size,
        "num_workers": num_workers, "evaluation_amp": use_amp, "bn_calibration_precision": "float32",
        "positive_weight": positive_weight, "metrics": metrics, "batchnorm": bn,
        "comparison": comparison, "seconds": time.perf_counter() - start,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": str(device), "rocm_hip": getattr(torch.version, "hip", None),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}})
    (output / "diagnosis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    save_dashboard(frames, metrics, output, technical)
    lines = ["# Diagnóstico interno", "", NOTICE, "",
             f"Checkpoint: `{checkpoint_path}` · época {checkpoint.get('epoch')} · `{digest}`", "",
             "**Ensayo técnico, no científico.**" if technical else "**Resultado exploratorio; test cerrado.**", "",
             "| Condición | ROC-AUC paciente | PR-AUC paciente | Loss por corte | Sensibilidad |",
             "|---|---:|---:|---:|---:|"]
    for key, values in metrics.items():
        p = values["patient"]
        lines.append(f"| {key} | {p['roc_auc']:.4f} | {p['pr_auc']:.4f} | {values['loss_per_cut']:.4f} | {p['sensitivity']:.4f} |")
    lines += ["", "![Curvas y probabilidades](diagnostic_dashboard.png)", "",
        "La copia BN no se guarda como checkpoint ni se incorpora a la web. Solo cambia estadísticas",
        "con train; no aprende pesos ni usa validación para recalibrarse. Su resultado depende del",
        "batch y la composición de pacientes. Una mejora no demuestra por sí sola un fallo de BN.", "",
        "Comparar train/validación en eval orienta el siguiente ensayo, pero el mejor checkpoint",
        "temprano no revela por sí solo cuánto memorizó el último. Puede diagnosticarse last.pt",
        "por separado para investigar la evolución, sin cambiar la selección del modelo."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--float32", action="store_true", help="Desactivar AMP solo en este diagnóstico")
    args = parser.parse_args()
    summary, output = run_diagnosis(args.checkpoint, device_name=args.device,
        batch_size=args.batch_size, num_workers=args.num_workers, data_root=args.data_root,
        output_dir=args.output_dir, bootstrap_repetitions=args.bootstrap_repetitions, float32=args.float32)
    compact = {"experiment_id": summary["experiment_id"], "checkpoint_epoch": summary["checkpoint_epoch"],
               "technical_subset": summary["technical_subset"], "test_evaluated": False,
               "comparison": summary["comparison"], "report_directory": str(output),
               "patient_metrics": {key: values["patient"] for key, values in summary["metrics"].items()}}
    print(json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
