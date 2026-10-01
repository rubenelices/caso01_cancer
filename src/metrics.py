"""Metricas binarias por corte y por paciente."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def aggregate_by_patient(
    patient_ids: Sequence[str],
    targets: Sequence[float],
    probabilities: Sequence[float],
    method: str = "mean",
    slice_threshold: float = 0.5,
) -> pd.DataFrame:
    """Agrega cortes manteniendo una etiqueta unica por paciente."""

    if method not in {"mean", "median", "max", "vote"}:
        raise ValueError("method debe ser mean, median, max o vote")
    if not 0 <= slice_threshold <= 1:
        raise ValueError("slice_threshold debe estar en [0, 1]")
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
    if method == "vote":
        aggregated = grouped["probability"].apply(
            lambda values: float((values >= slice_threshold).mean())
        ).to_frame()
    else:
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
    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
        raise ValueError("targets y probabilities deben ser vectores de igual longitud")
    if len(y_true) == 0:
        raise ValueError("No hay predicciones para evaluar")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("targets debe contener solo 0 y 1")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("probabilities debe contener valores finitos en [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar en [0, 1]")
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def safe_divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    sensitivity = safe_divide(int(tp), int(tp + fn))
    specificity = safe_divide(int(tn), int(tn + fp))
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else float("nan")
    )

    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": safe_divide(int(tp + tn), int(len(y_true))),
        "balanced_accuracy": balanced_accuracy,
        "precision": (
            safe_divide(int(tp), int(tp + fp)) if tp + fp else 0.0
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": (
            safe_divide(int(2 * tp), int(2 * tp + fp + fn))
            if 2 * tp + fp + fn
            else 0.0
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if np.unique(y_true).size == 2
            else float("nan")
        ),
        "pr_auc": (
            float(average_precision_score(y_true, y_prob))
            if np.unique(y_true).size == 2
            else float("nan")
        ),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
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
    slice_threshold: float = 0.5,
) -> dict[str, Any]:
    patient_frame = aggregate_by_patient(
        patient_ids,
        targets,
        probabilities,
        method=aggregation,
        slice_threshold=slice_threshold,
    )
    return {
        "cut": binary_metrics(targets, probabilities, threshold),
        "patient": binary_metrics(
            patient_frame["target"], patient_frame["probability"], threshold
        ),
        "aggregation": aggregation,
    }
