"""Muestra la arquitectura base sin entrenar ni cargar imagenes."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.architectures import BreastPCRNet, trainable_parameter_count
from src.experiment_config import load_experiment_config


DEFAULT_CONFIG = Path("configs/experiments/E02_base_normal.json")


def describe_architecture(config_path: Path = DEFAULT_CONFIG) -> str:
    config = load_experiment_config(config_path)
    model = BreastPCRNet(config.model)
    lines = [
        "ARQUITECTURA CNN BASE",
        "=" * 72,
        str(model),
        "",
        "RECORRIDO DE TENSORES [batch, canales, alto, ancho]",
        "-" * 72,
    ]
    for step in model.trace_shapes(batch_size=2):
        lines.append(f"{step.operation:<32} {str(step.shape):>28}")
    lines.extend(
        [
            "-" * 72,
            f"Parametros entrenables: {trainable_parameter_count(model):,}".replace(
                ",", "."
            ),
            "Entrada: PRE, EARLY y LATE; no son canales RGB.",
            "Salida: un logit por corte; sigmoid(logit) da la probabilidad.",
            "Esta orden solo inspecciona: no carga datos ni entrena.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(describe_architecture(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
