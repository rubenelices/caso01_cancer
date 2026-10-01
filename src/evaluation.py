"""Evaluacion reproducible en validacion interna, con paciente como unidad.

Este modulo no ofrece una opcion para evaluar test. La seleccion de agregacion,
umbral, calibracion y arquitectura debe cerrarse usando exclusivamente los
folds internos de train antes de abrir el test una sola vez.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from torch import nn

from src.architectures import BaseCNNConfig, BreastPCRNet
from src.data import LoaderConfig, create_dataloaders, load_samples, split_by_patient_fold
from src.experiment_config import load_experiment_config
from src.metrics import aggregate_by_patient, binary_metrics
from src.train import choose_device, set_reproducibility


EVALUATION_NOTICE = (
    "Resultado educativo y de investigacion; no tiene validez clinica ni debe "
    "usarse para diagnostico o decisiones terapeuticas."
)
BOOTSTRAP_METRICS = (
    "roc_auc",
    "pr_auc",
    "brier_score",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "balanced_accuracy",
)


@dataclass(frozen=True)
class EvaluationConfig:
    threshold: float = 0.5
    aggregation: str = "mean"
    bootstrap_repetitions: int = 2_000
    confidence_level: float = 0.95
    calibration_bins: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold debe estar en [0, 1]")
        if self.aggregation not in {"mean", "median", "max", "vote"}:
            raise ValueError("aggregation no reconocida")
        if self.bootstrap_repetitions < 1:
            raise ValueError("bootstrap_repetitions debe ser positivo")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level debe estar en (0, 1)")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins debe ser al menos 2")


@torch.no_grad()
def collect_cut_predictions(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> pd.DataFrame:
    """Conserva una fila trazable por corte en el orden secuencial del loader."""

    if getattr(loader, "drop_last", False):
        raise ValueError("El loader de evaluacion no puede usar drop_last=True")
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            probabilities = torch.sigmoid(model(images)).float().cpu().numpy()
        targets = batch["target"].float().cpu().numpy()
        for index, probability in enumerate(probabilities):
            records.append(
                {
                    "sample_id": str(batch["sample_id"][index]),
                    "patient_id": str(batch["patient_id"][index]),
                    "slice_index": int(batch["slice_index"][index]),
                    "split": str(batch["split"][index]),
                    "fold": int(batch["fold"][index]),
                    "target": int(targets[index]),
                    "probability": float(probability),
                }
            )
    if not records:
        raise ValueError("El loader de evaluacion no produjo predicciones")
    frame = pd.DataFrame.from_records(records)
    if set(frame["split"]) != {"train"}:
        raise ValueError("Solo se permite evaluar el fold interno del train publico")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Hay sample_id duplicados en las predicciones")
    return frame


def patient_predictions(cut_predictions: pd.DataFrame, config: EvaluationConfig) -> pd.DataFrame:
    required = {"sample_id", "patient_id", "target", "probability", "split", "fold"}
    missing = sorted(required - set(cut_predictions.columns))
    if missing:
        raise ValueError(f"Faltan columnas de prediccion: {missing}")
    if set(cut_predictions["split"].astype(str)) != {"train"}:
        raise ValueError("La evaluacion de desarrollo solo acepta validacion interna")
    aggregated = aggregate_by_patient(
        cut_predictions["patient_id"],
        cut_predictions["target"],
        cut_predictions["probability"],
        method=config.aggregation,
        slice_threshold=config.threshold,
    )
    patient_fold_counts = cut_predictions.groupby("patient_id")["fold"].nunique()
    if (patient_fold_counts != 1).any():
        raise ValueError("Una paciente aparece en mas de un fold")
    folds = cut_predictions.groupby("patient_id")["fold"].first()
    aggregated["fold"] = aggregated["patient_id"].map(folds).astype(int)
    aggregated["split"] = "validation"
    aggregated["prediction"] = (
        aggregated["probability"] >= config.threshold
    ).astype(int)
    return aggregated[
        ["patient_id", "fold", "split", "target", "probability", "prediction", "cuts"]
    ]


def bootstrap_confidence_intervals(
    predictions: pd.DataFrame,
    config: EvaluationConfig,
) -> dict[str, dict[str, float | int]]:
    """Bootstrap no parametrico; cada remuestreo toma pacientes completas."""

    rng = np.random.default_rng(config.seed)
    targets = predictions["target"].to_numpy(dtype=int)
    probabilities = predictions["probability"].to_numpy(dtype=float)
    estimates: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(config.bootstrap_repetitions):
        indices = rng.integers(0, len(predictions), size=len(predictions))
        sampled = binary_metrics(
            targets[indices], probabilities[indices], threshold=config.threshold
        )
        for name in BOOTSTRAP_METRICS:
            value = float(sampled[name])
            if np.isfinite(value):
                estimates[name].append(value)

    alpha = 1 - config.confidence_level
    result: dict[str, dict[str, float | int]] = {}
    point_metrics = binary_metrics(targets, probabilities, config.threshold)
    for name, values in estimates.items():
        array = np.asarray(values, dtype=float)
        lower = float(np.quantile(array, alpha / 2)) if len(array) else float("nan")
        upper = (
            float(np.quantile(array, 1 - alpha / 2)) if len(array) else float("nan")
        )
        result[name] = {
            "estimate": float(point_metrics[name]),
            "lower": lower,
            "upper": upper,
            "valid_repetitions": int(len(array)),
            "requested_repetitions": config.bootstrap_repetitions,
        }
    return result


def expected_calibration_error(
    targets: np.ndarray, probabilities: np.ndarray, bins: int
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for bin_index in range(bins):
        mask = assignments == bin_index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(targets[mask].mean()) - float(probabilities[mask].mean())
            )
    return error


def build_evaluation_summary(
    cut_predictions: pd.DataFrame,
    patient_frame: pd.DataFrame,
    config: EvaluationConfig,
) -> dict[str, Any]:
    patient_metrics = binary_metrics(
        patient_frame["target"], patient_frame["probability"], config.threshold
    )
    cut_metrics = binary_metrics(
        cut_predictions["target"], cut_predictions["probability"], config.threshold
    )
    targets = patient_frame["target"].to_numpy(dtype=int)
    probabilities = patient_frame["probability"].to_numpy(dtype=float)
    return {
        "scope": "internal_validation_only",
        "statistical_unit": "patient",
        "aggregation": config.aggregation,
        "threshold": config.threshold,
        "bootstrap": {
            "method": "nonparametric_percentile_by_patient",
            "confidence_level": config.confidence_level,
            "seed": config.seed,
            "intervals": bootstrap_confidence_intervals(patient_frame, config),
        },
        "calibration": {
            "brier_score": patient_metrics["brier_score"],
            "expected_calibration_error": expected_calibration_error(
                targets, probabilities, config.calibration_bins
            ),
            "bins": config.calibration_bins,
        },
        "cut_metrics_secondary": cut_metrics,
        "patient_metrics_primary": patient_metrics,
        "notice": EVALUATION_NOTICE,
    }


def save_evaluation_figure(
    patient_frame: pd.DataFrame,
    summary: dict[str, Any],
    config: EvaluationConfig,
    path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-caso-cancer")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = patient_frame["target"].to_numpy(dtype=int)
    probabilities = patient_frame["probability"].to_numpy(dtype=float)
    false_positive_rate, true_positive_rate, _ = roc_curve(targets, probabilities)
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    observed, predicted = calibration_curve(
        targets,
        probabilities,
        n_bins=config.calibration_bins,
        strategy="quantile",
    )
    matrix = summary["patient_metrics_primary"]["confusion_matrix"]
    matrix_values = np.asarray([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])

    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].plot(false_positive_rate, true_positive_rate, color="#126E82", linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], color="0.6", linestyle="--")
    axes[0, 0].set(
        xlabel="1 - especificidad",
        ylabel="Sensibilidad",
        title=f"ROC por paciente (AUC={summary['patient_metrics_primary']['roc_auc']:.3f})",
    )

    prevalence = float(targets.mean())
    axes[0, 1].plot(recall, precision, color="#A44A3F", linewidth=2)
    axes[0, 1].axhline(prevalence, color="0.6", linestyle="--", label="Prevalencia")
    axes[0, 1].set(
        xlabel="Sensibilidad",
        ylabel="Precision",
        title=f"Precision-recall (AP={summary['patient_metrics_primary']['pr_auc']:.3f})",
    )
    axes[0, 1].legend()

    axes[1, 0].plot([0, 1], [0, 1], color="0.6", linestyle="--", label="Ideal")
    axes[1, 0].plot(predicted, observed, marker="o", color="#5C6B2F", linewidth=2)
    axes[1, 0].set(
        xlabel="Probabilidad media predicha",
        ylabel="Fraccion pCR observada",
        title=f"Calibracion (Brier={summary['calibration']['brier_score']:.3f})",
    )

    axes[1, 1].imshow(matrix_values, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1, 1].text(
                column,
                row,
                str(matrix_values[row, column]),
                ha="center",
                va="center",
                fontsize=16,
            )
    axes[1, 1].set_xticks([0, 1], labels=["Predice 0", "Predice 1"])
    axes[1, 1].set_yticks([0, 1], labels=["Real 0", "Real 1"])
    axes[1, 1].set_title(f"Matriz de confusion (umbral={config.threshold:.2f})")

    for axis in axes.flat:
        axis.grid(alpha=0.15)
    figure.suptitle("Evaluacion interna - unidad estadistica: paciente", fontsize=14)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_evaluation_artifacts(
    cut_predictions: pd.DataFrame,
    output_dir: str | Path,
    config: EvaluationConfig,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    patients = patient_predictions(cut_predictions, config)
    summary = build_evaluation_summary(cut_predictions, patients, config)
    cut_predictions.to_csv(output / "predictions_cut.csv", index=False)
    patients.to_csv(output / "predictions_patient.csv", index=False)
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_evaluation_figure(patients, summary, config, output / "evaluation_patient.png")
    return summary


def evaluate_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    evaluation: EvaluationConfig,
    device_name: str = "auto",
) -> dict[str, Any]:
    experiment = load_experiment_config(config_path)
    set_reproducibility(evaluation.seed)
    device = choose_device(device_name)
    samples = load_samples(experiment.data.root)
    splits = split_by_patient_fold(samples, experiment.data.validation_fold)
    loaders = create_dataloaders(
        splits,
        experiment.data.root,
        LoaderConfig(
            batch_size=experiment.data.batch_size,
            num_workers=experiment.data.num_workers,
            pin_memory=experiment.data.pin_memory and device.type == "cuda",
            persistent_workers=experiment.data.num_workers > 0,
            seed=evaluation.seed,
        ),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = BaseCNNConfig(**checkpoint["config"]["model"])
    model = BreastPCRNet(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = collect_cut_predictions(
        model,
        loaders.validation,
        device,
        use_amp=experiment.training.mixed_precision and device.type == "cuda",
    )
    summary = save_evaluation_artifacts(predictions, output_dir, evaluation)
    summary["experiment_id"] = experiment.experiment_id
    summary["validation_fold"] = experiment.data.validation_fold
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aggregation", choices=("mean", "median", "max", "vote"), default="mean")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = evaluate_checkpoint(
        args.config,
        args.checkpoint,
        args.output_dir,
        EvaluationConfig(
            threshold=args.threshold,
            aggregation=args.aggregation,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        ),
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
