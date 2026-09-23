"""Experimento E01: memorizar un subconjunto pequeno con la CNN minima.

Memorizar 20-50 muestras no mide generalizacion. Es una prueba de cableado: si
el modelo no puede reducir mucho la perdida sobre un conjunto diminuto, no tiene
sentido iniciar entrenamientos largos.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from src.data import BreastDCEDataset, load_samples, split_by_patient_fold
from src.model import MinimalCNN, parameter_breakdown, trainable_parameter_count


@dataclass(frozen=True)
class OverfitConfig:
    samples: int = 24
    epochs: int = 150
    learning_rate: float = 3e-3
    seed: int = 42
    validation_fold: int = 0
    target_accuracy: float = 1.0
    target_loss: float = 0.08
    patience_at_target: int = 5

    def __post_init__(self) -> None:
        if self.samples < 20 or self.samples > 50 or self.samples % 2:
            raise ValueError("samples debe ser un numero par entre 20 y 50")
        if self.epochs < 1:
            raise ValueError("epochs debe ser positivo")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_balanced_patient_subset(
    train_rows: pd.DataFrame,
    sample_count: int,
    seed: int,
) -> pd.DataFrame:
    """Elige una muestra de pacientes distintas, mitad de cada clase."""

    if sample_count % 2:
        raise ValueError("sample_count debe ser par")
    rng = np.random.default_rng(seed)
    selected: list[pd.Series] = []
    per_class = sample_count // 2

    for label in (0, 1):
        class_rows = train_rows.loc[train_rows["pCR"] == label]
        patient_ids = np.array(sorted(class_rows["patient_id"].unique()))
        if len(patient_ids) < per_class:
            raise ValueError(f"No hay {per_class} pacientes para pCR={label}")
        chosen = rng.choice(patient_ids, size=per_class, replace=False)
        for patient_id in chosen:
            cuts = class_rows.loc[class_rows["patient_id"] == patient_id].sort_values(
                "slice_index"
            )
            selected.append(cuts.iloc[len(cuts) // 2])

    result = pd.DataFrame(selected).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if result["patient_id"].nunique() != sample_count:
        raise RuntimeError("El subconjunto no contiene pacientes unicas")
    if result["pCR"].value_counts().to_dict() != {0: per_class, 1: per_class}:
        raise RuntimeError("El subconjunto no esta equilibrado")
    return result


def materialize_subset(dataset: BreastDCEDataset) -> tuple[Tensor, Tensor, list[str], list[str]]:
    """Carga una vez el pequeno subconjunto para acelerar el smoke test."""

    items = [dataset[index] for index in range(len(dataset))]
    images = torch.stack([item["image"] for item in items])
    targets = torch.stack([item["target"] for item in items])
    sample_ids = [item["sample_id"] for item in items]
    patient_ids = [item["patient_id"] for item in items]
    return images, targets, sample_ids, patient_ids


def gradient_summary(model: nn.Module) -> dict[str, float]:
    """Norma L2 del gradiente de cada tensor entrenable."""

    result: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"El parametro {name} no recibio gradiente")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"El gradiente de {name} contiene no finitos")
        result[name] = float(parameter.grad.detach().norm().cpu())
    return result


def run_overfit_experiment(
    root: str | Path = "breastdcedl",
    output: str | Path = "reports/phase3_minimal",
    config: OverfitConfig = OverfitConfig(),
) -> dict[str, Any]:
    """Entrena sobre un subconjunto equilibrado y guarda evidencia del resultado."""

    set_reproducibility(config.seed)
    device = choose_device()
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = load_samples(root)
    splits = split_by_patient_fold(samples, config.validation_fold)
    subset_rows = select_balanced_patient_subset(
        splits.train, config.samples, config.seed
    )
    dataset = BreastDCEDataset(subset_rows, root)
    images, targets, sample_ids, patient_ids = materialize_subset(dataset)
    images = images.to(device)
    targets = targets.to(device)

    # Reiniciamos la semilla justo antes de construir el modelo para que cargar
    # datos no afecte a la inicializacion de pesos.
    set_reproducibility(config.seed)
    model = MinimalCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float | int]] = []
    first_gradients: dict[str, float] | None = None
    consecutive_target_epochs = 0
    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        if first_gradients is None:
            first_gradients = gradient_summary(model)
        optimizer.step()

        with torch.no_grad():
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).to(targets.dtype)
            accuracy = float((predictions == targets).float().mean().cpu())
            current_loss = float(loss.detach().cpu())
        history.append({"epoch": epoch, "loss": current_loss, "accuracy": accuracy})

        reached = accuracy >= config.target_accuracy and current_loss <= config.target_loss
        consecutive_target_epochs = consecutive_target_epochs + 1 if reached else 0
        if epoch == 1 or epoch % 10 == 0 or consecutive_target_epochs == 1:
            print(f"epoca={epoch:03d} loss={current_loss:.5f} accuracy={accuracy:.3f}")
        if consecutive_target_epochs >= config.patience_at_target:
            break

    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        final_logits = model(images)
        final_probabilities = torch.sigmoid(final_logits)
        final_predictions = (final_probabilities >= 0.5).to(targets.dtype)
        final_loss = float(criterion(final_logits, targets).cpu())
        final_accuracy = float((final_predictions == targets).float().mean().cpu())

    success = final_accuracy >= config.target_accuracy and final_loss <= config.target_loss
    report: dict[str, Any] = {
        "experiment": "E01_minimal_cnn_overfit",
        "purpose": "Prueba de cableado; no mide generalizacion",
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "parameter_count": trainable_parameter_count(model),
        "parameter_breakdown": parameter_breakdown(model),
        "shape_trace": [asdict(step) for step in model.trace_shapes(batch_size=2)],
        "subset": {
            "samples": len(sample_ids),
            "patients": len(set(patient_ids)),
            "class_counts": {
                str(key): int(value)
                for key, value in subset_rows["pCR"].value_counts().sort_index().items()
            },
            "sample_ids": sample_ids,
            "patient_ids": patient_ids,
        },
        "first_step_gradient_norms": first_gradients,
        "epochs_completed": len(history),
        "elapsed_seconds": elapsed,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "success": success,
        "history": history,
        "final_predictions": [
            {
                "sample_id": sample_id,
                "target": int(target),
                "probability": float(probability),
                "prediction": int(prediction),
            }
            for sample_id, target, probability, prediction in zip(
                sample_ids,
                targets.detach().cpu().tolist(),
                final_probabilities.detach().cpu().tolist(),
                final_predictions.detach().cpu().tolist(),
            )
        ],
    }

    report_path = output_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_training_curve(history, output_path / "training_curve.png")
    if not success:
        raise RuntimeError(
            f"La CNN minima no memorizo el subconjunto: loss={final_loss:.4f}, "
            f"accuracy={final_accuracy:.3f}. Revisar antes de continuar."
        )
    return report


def _save_training_curve(history: list[dict[str, float | int]], path: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-caso-cancer")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    accuracies = [float(row["accuracy"]) for row in history]
    figure, loss_axis = plt.subplots(figsize=(8, 4.5))
    accuracy_axis = loss_axis.twinx()
    loss_axis.plot(epochs, losses, color="#1d4ed8", label="Loss")
    accuracy_axis.plot(epochs, accuracies, color="#be123c", label="Accuracy")
    loss_axis.set_xlabel("Epoca")
    loss_axis.set_ylabel("BCEWithLogitsLoss", color="#1d4ed8")
    accuracy_axis.set_ylabel("Accuracy", color="#be123c")
    accuracy_axis.set_ylim(0, 1.05)
    loss_axis.set_title("CNN minima: memorizacion deliberada del subconjunto")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("breastdcedl"))
    parser.add_argument("--output", type=Path, default=Path("reports/phase3_minimal"))
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fold", type=int, default=0, choices=range(5))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = OverfitConfig(
        samples=args.samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        validation_fold=args.validation_fold,
    )
    report = run_overfit_experiment(args.root, args.output, config)
    print()
    print("Smoke test de aprendizaje: PASS")
    print(f"Parametros entrenables: {report['parameter_count']:,}")
    print(f"Epocas: {report['epochs_completed']}")
    print(f"Loss final: {report['final_loss']:.6f}")
    print(f"Accuracy final: {report['final_accuracy']:.3f}")
    print(f"Tiempo: {report['elapsed_seconds']:.2f} s")
    print(f"Informe: {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
