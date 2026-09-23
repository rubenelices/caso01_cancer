"""Ensayo general corto de todo el pipeline de entrenamiento."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.experiment_config import load_experiment_config
from src.train import run_experiment


DEFAULT_CONFIG = Path("configs/experiments/SMOKE_base_end_to_end.json")


def validate_smoke_config(config: Any) -> None:
    """Impide convertir accidentalmente el smoke test en un trabajo largo."""

    train_limit = config.data.train_patients_per_class
    validation_limit = config.data.validation_patients_per_class
    if train_limit is None or validation_limit is None:
        raise ValueError("El smoke test exige limites explicitos por paciente y clase")
    if train_limit > 8 or validation_limit > 4:
        raise ValueError("El smoke test admite como maximo 8/4 pacientes por clase")
    if config.training.epochs > 3:
        raise ValueError("El smoke test admite como maximo 3 epocas")


def verify_artifacts(config: Any, summary: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    required = [
        output_dir / "config_resolved.json",
        output_dir / "environment.json",
        output_dir / "history.csv",
        output_dir / "training_curves.png",
        output_dir / "summary.json",
        checkpoint_dir / "best.pt",
        checkpoint_dir / "last.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Faltan artefactos del smoke test: {missing}")

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu")
    checks = {
        "two_classes_in_train": summary["patients"]["train"]
        == 2 * config.data.train_patients_per_class,
        "two_classes_in_validation": summary["patients"]["validation"]
        == 2 * config.data.validation_patients_per_class,
        "test_not_evaluated": summary["test_evaluated"] is False,
        "marked_non_scientific": summary["scientific_result"] is False,
        "best_checkpoint_loads": "model_state_dict" in checkpoint,
        "history_has_all_epochs": summary["epochs_completed"]
        == config.training.epochs,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Comprobaciones del smoke test fallidas: {checks}")
    return {
        "status": "passed",
        "meaning": "El pipeline funciona; no estima la calidad del modelo",
        "checks": checks,
        "patients": summary["patients"],
        "samples": summary["samples"],
        "epochs": summary["epochs_completed"],
        "seconds": summary["total_seconds"],
        "artifacts": [str(path) for path in required],
    }


def run_smoke(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    validate_smoke_config(config)
    summary = run_experiment(config, device_name="cpu", overwrite=True)
    verification = verify_artifacts(config, summary)
    verification_path = Path(config.output_dir) / "smoke_verification.json"
    verification_path.write_text(
        json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verification = run_smoke(args.config)
    print(json.dumps(verification, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
