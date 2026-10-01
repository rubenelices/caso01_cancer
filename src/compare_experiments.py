"""Comparacion pareada y reproducible de dos experimentos en validacion interna."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import binary_metrics


COMPARISON_METRICS = (
    "roc_auc",
    "pr_auc",
    "brier_score",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "balanced_accuracy",
)


def _load_experiment(directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions_path = directory / "predictions_patient.csv"
    summary_path = directory / "evaluation_summary.json"
    predictions = pd.read_csv(predictions_path, dtype={"patient_id": str})
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(predictions["split"].astype(str)) != {"validation"}:
        raise ValueError(f"{directory} no contiene solo validacion interna")
    if predictions["patient_id"].duplicated().any():
        raise ValueError(f"{directory} contiene pacientes duplicadas")
    return predictions, summary


def _aligned_predictions(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = left.sort_values("patient_id").reset_index(drop=True)
    right = right.sort_values("patient_id").reset_index(drop=True)
    if left["patient_id"].tolist() != right["patient_id"].tolist():
        raise ValueError("Los experimentos no evaluaron exactamente las mismas pacientes")
    if not np.array_equal(left["target"].to_numpy(), right["target"].to_numpy()):
        raise ValueError("Las etiquetas no coinciden entre experimentos")
    if not np.array_equal(left["fold"].to_numpy(), right["fold"].to_numpy()):
        raise ValueError("Los folds no coinciden entre experimentos")
    return left, right


def compare_experiments(
    left_dir: str | Path,
    right_dir: str | Path,
    *,
    repetitions: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Estima diferencias right-left remuestreando las mismas pacientes."""

    if repetitions < 1:
        raise ValueError("repetitions debe ser positivo")
    left, left_summary = _load_experiment(Path(left_dir))
    right, right_summary = _load_experiment(Path(right_dir))
    left, right = _aligned_predictions(left, right)
    for field in ("aggregation", "threshold"):
        if left_summary[field] != right_summary[field]:
            raise ValueError(f"La comparacion exige el mismo {field}")

    threshold = float(left_summary["threshold"])
    targets = left["target"].to_numpy(dtype=int)
    left_probabilities = left["probability"].to_numpy(dtype=float)
    right_probabilities = right["probability"].to_numpy(dtype=float)
    left_metrics = binary_metrics(targets, left_probabilities, threshold)
    right_metrics = binary_metrics(targets, right_probabilities, threshold)
    rng = np.random.default_rng(seed)
    differences: dict[str, list[float]] = {metric: [] for metric in COMPARISON_METRICS}
    for _ in range(repetitions):
        indices = rng.integers(0, len(targets), size=len(targets))
        sampled_left = binary_metrics(targets[indices], left_probabilities[indices], threshold)
        sampled_right = binary_metrics(targets[indices], right_probabilities[indices], threshold)
        for metric in COMPARISON_METRICS:
            difference = float(sampled_right[metric]) - float(sampled_left[metric])
            if np.isfinite(difference):
                differences[metric].append(difference)

    alpha = 1 - confidence_level
    comparison: dict[str, dict[str, float | int]] = {}
    for metric, values in differences.items():
        array = np.asarray(values, dtype=float)
        lower = float(np.quantile(array, alpha / 2)) if len(array) else float("nan")
        upper = (
            float(np.quantile(array, 1 - alpha / 2)) if len(array) else float("nan")
        )
        comparison[metric] = {
            "left": float(left_metrics[metric]),
            "right": float(right_metrics[metric]),
            "difference_right_minus_left": float(right_metrics[metric] - left_metrics[metric]),
            "difference_ci_lower": lower,
            "difference_ci_upper": upper,
            "probability_difference_positive": (
                float((array > 0).mean()) if len(array) else float("nan")
            ),
            "valid_repetitions": int(len(array)),
        }
    return {
        "scope": "internal_validation_only",
        "comparison_design": "paired_bootstrap_by_patient",
        "left_experiment": left_summary.get("experiment_id", Path(left_dir).name),
        "right_experiment": right_summary.get("experiment_id", Path(right_dir).name),
        "patients": int(len(targets)),
        "folds": sorted(left["fold"].astype(int).unique().tolist()),
        "aggregation": left_summary["aggregation"],
        "threshold": threshold,
        "confidence_level": confidence_level,
        "repetitions": repetitions,
        "seed": seed,
        "metrics": comparison,
        "interpretation": (
            "Un intervalo de la diferencia que cruza 0 no aporta evidencia clara de "
            "ventaja. Para Brier, una diferencia negativa favorece al experimento derecho; "
            "para el resto, una diferencia positiva lo favorece."
        ),
    }


def save_comparison(result: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = [{"metric": metric, **values} for metric, values in result["metrics"].items()]
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-caso-cancer")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    positions = np.arange(len(frame))
    values = frame["difference_right_minus_left"].to_numpy()
    errors = np.vstack(
        [
            values - frame["difference_ci_lower"].to_numpy(),
            frame["difference_ci_upper"].to_numpy() - values,
        ]
    )
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.errorbar(values, positions, xerr=errors, fmt="o", color="#126E82", capsize=4)
    axis.axvline(0, color="0.5", linestyle="--")
    axis.set_yticks(positions, labels=frame["metric"])
    axis.set_xlabel("Diferencia: experimento derecho - izquierdo")
    axis.set_title("Comparacion pareada por paciente en validacion interna")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_experiments(
        args.left, args.right, repetitions=args.repetitions, seed=args.seed
    )
    save_comparison(result, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
