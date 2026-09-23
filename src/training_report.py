"""Diagnostico y figura final para interpretar un entrenamiento."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def early_learning_signal(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Compara la loss de train de la epoca 1 con la epoca 5."""

    if len(history) < 5:
        return {
            "status": "insufficient_epochs",
            "epoch_checked": len(history),
            "message": "Se necesitan 5 epocas para este diagnostico.",
        }
    initial = float(history[0]["train_loss"])
    fifth = float(history[4]["train_loss"])
    relative_drop = (initial - fifth) / max(abs(initial), 1e-12)
    decreasing = fifth < initial
    return {
        "status": "decreasing" if decreasing else "not_decreasing",
        "epoch_checked": 5,
        "epoch_1_train_loss": initial,
        "epoch_5_train_loss": fifth,
        "relative_drop": relative_drop,
        "message": (
            f"La loss de train ha bajado un {relative_drop:.1%} entre las epocas 1 y 5."
            if decreasing
            else "La loss de train no ha bajado entre las epocas 1 y 5; revisar el entrenamiento."
        ),
    }


def build_training_diagnostics(
    history: list[dict[str, Any]],
    final_metrics: dict[str, Any],
    best_epoch: int,
) -> dict[str, Any]:
    if not history:
        raise ValueError("No hay historial de entrenamiento")
    patient = final_metrics["patient"]
    matrix = patient["confusion_matrix"]
    positives = matrix["tp"] + matrix["fn"]
    negatives = matrix["tn"] + matrix["fp"]
    prevalence = positives / max(positives + negatives, 1)
    predicted_positives = matrix["tp"] + matrix["fp"]
    predicted_negatives = matrix["tn"] + matrix["fn"]
    if predicted_positives == 0:
        collapse = "all_negative"
    elif predicted_negatives == 0:
        collapse = "all_positive"
    else:
        collapse = "none"

    validation_losses = [float(row["validation_loss"]) for row in history]
    train_losses = [float(row["train_loss"]) for row in history]
    minimum_validation = min(validation_losses)
    minimum_validation_epoch = int(np.argmin(validation_losses)) + 1
    overfitting_warning = (
        len(history) >= 5
        and validation_losses[-1] > minimum_validation * 1.10
        and train_losses[-1] < train_losses[minimum_validation_epoch - 1]
    )
    early = early_learning_signal(history)
    needs_review = early["status"] == "not_decreasing" or collapse != "none"
    return {
        "interpretation": "review" if needs_review else "learning_signal_present",
        "early_epoch_5_check": early,
        "prediction_collapse": collapse,
        "overfitting_warning": overfitting_warning,
        "best_epoch_by_monitor": best_epoch,
        "minimum_validation_loss_epoch": minimum_validation_epoch,
        "baselines": {
            "majority_accuracy": max(prevalence, 1.0 - prevalence),
            "random_roc_auc": 0.5,
            "prevalence_pr_auc": prevalence,
            "balanced_accuracy": 0.5,
        },
        "final_best_checkpoint_patient_metrics": {
            key: patient[key]
            for key in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "sensitivity",
                "specificity",
                "f1",
                "roc_auc",
                "pr_auc",
            )
        },
        "notes": [
            "Las metricas corresponden a validacion por paciente.",
            "Una loss de validacion creciente con loss de train decreciente sugiere sobreajuste.",
            "E02 y E03 usan losses con escalas distintas; no comparar solo su valor absoluto.",
            "Este diagnostico no usa el conjunto test.",
        ],
    }


def save_training_dashboard(
    history: list[dict[str, Any]],
    final_metrics: dict[str, Any],
    best_epoch: int,
    path: Path,
) -> None:
    """Guarda cuatro paneles para diagnosticar el entrenamiento de un vistazo."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-caso-cancer")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.asarray([row["epoch"] for row in history])
    diagnostics = build_training_diagnostics(history, final_metrics, best_epoch)
    baselines = diagnostics["baselines"]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    loss_axis = axes[0, 0]
    loss_axis.plot(epochs, [row["train_loss"] for row in history], label="Train", marker="o")
    loss_axis.plot(
        epochs,
        [row["validation_loss"] for row in history],
        label="Validacion",
        marker="o",
    )
    if len(history) >= 5:
        loss_axis.axvline(5, color="0.5", linestyle=":", label="Chequeo epoca 5")
    loss_axis.set(xlabel="Epoca", ylabel="Loss", title="1. ¿Esta aprendiendo?")
    loss_axis.grid(alpha=0.2)
    loss_axis.legend()

    auc_axis = axes[0, 1]
    auc_axis.plot(
        epochs,
        [row["patient_roc_auc"] for row in history],
        label="ROC-AUC paciente",
        marker="o",
    )
    auc_axis.plot(
        epochs,
        [row["patient_pr_auc"] for row in history],
        label="PR-AUC paciente",
        marker="o",
    )
    auc_axis.axhline(0.5, color="0.4", linestyle="--", label="ROC azar = 0,5")
    auc_axis.axhline(
        baselines["prevalence_pr_auc"],
        color="0.6",
        linestyle=":",
        label=f"PR base = {baselines['prevalence_pr_auc']:.2f}",
    )
    auc_axis.axvline(
        best_epoch,
        color="tab:green",
        alpha=0.5,
        linestyle="--",
        label=f"Mejor epoca = {best_epoch}",
    )
    auc_axis.set(xlabel="Epoca", ylabel="AUC", ylim=(0, 1), title="2. ¿Separa las clases?")
    auc_axis.grid(alpha=0.2)
    auc_axis.legend(fontsize=8)

    metric_axis = axes[1, 0]
    for key, label in (
        ("patient_balanced_accuracy", "Balanced accuracy"),
        ("patient_sensitivity", "Sensibilidad pCR=1"),
        ("patient_specificity", "Especificidad pCR=0"),
    ):
        metric_axis.plot(epochs, [row[key] for row in history], label=label, marker="o")
    metric_axis.axhline(0.5, color="0.5", linestyle="--", label="Balanced base = 0,5")
    metric_axis.set(
        xlabel="Epoca",
        ylabel="Valor",
        ylim=(0, 1),
        title="3. ¿Predice ambas clases?",
    )
    metric_axis.grid(alpha=0.2)
    metric_axis.legend(fontsize=8)

    matrix = final_metrics["patient"]["confusion_matrix"]
    values = np.asarray([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
    matrix_axis = axes[1, 1]
    matrix_axis.imshow(values, cmap="Blues")
    contrast_threshold = values.max() / 2 if values.max() else 0
    for row in range(2):
        for column in range(2):
            matrix_axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                fontsize=16,
                color="white" if values[row, column] > contrast_threshold else "black",
            )
    matrix_axis.set_xticks([0, 1], labels=["Predice 0", "Predice 1"])
    matrix_axis.set_yticks([0, 1], labels=["Real 0", "Real 1"])
    matrix_axis.set_title("4. Matriz de confusion del mejor checkpoint")

    early = diagnostics["early_epoch_5_check"]
    subtitle = early["message"]
    if diagnostics["prediction_collapse"] != "none":
        subtitle += " Advertencia: el modelo predice una sola clase."
    figure.suptitle(f"Diagnostico del entrenamiento\n{subtitle}", fontsize=13)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_diagnostics_json(diagnostics: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
