"""Metricas binarias por corte y por paciente."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


def aggregate_by_patient(
    patient_ids: Sequence[str],
    targets: Sequence[float],
    probabilities: Sequence[float],
    method: str = "mean",
) -> pd.DataFrame:
    """Agrega cortes manteniendo una etiqueta unica por paciente."""

    if method not in {"mean", "median", "max"}:
        raise ValueError("method debe ser mean, median o max")
    frame = pd.DataFrame(
        {
            "patient_id": list(patient_ids),
            "target": np.asarray(targets, dtype=int),
            "probability": np.asarray(probabilities, dtype=float),
        }
    )
    if frame.empty:
        raise ValueError("No hay predicciones para agregar")
    inconsistent = frame.groupby("patient_id")["target"].nunique()
    if (inconsistent > 1).any():
        raise ValueError("La etiqueta cambia dentro de una paciente")
    grouped = frame.groupby("patient_id")
    aggregated = getattr(grouped["probability"], method)().to_frame()
    aggregated["target"] = grouped["target"].first()
    aggregated["cuts"] = grouped.size()
    return aggregated.reset_index()


def binary_metrics(
    targets: Sequence[float],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(targets, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def safe_divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": safe_divide(int(tp), int(tp + fn)),
        "specificity": safe_divide(int(tn), int(tn + fp)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def cut_and_patient_metrics(
    patient_ids: Sequence[str],
    targets: Sequence[float],
    probabilities: Sequence[float],
    threshold: float = 0.5,
    aggregation: str = "mean",
) -> dict[str, Any]:
    patient_frame = aggregate_by_patient(
        patient_ids, targets, probabilities, method=aggregation
    )
    return {
        "cut": binary_metrics(targets, probabilities, threshold),
        "patient": binary_metrics(
            patient_frame["target"], patient_frame["probability"], threshold
        ),
        "aggregation": aggregation,
    }
