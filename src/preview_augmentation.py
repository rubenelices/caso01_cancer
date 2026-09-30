"""Lámina local de dos cortes de train: originales y aumentos compartidos."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.augmentation import SharedRandomAffine, apply_shared_affine
from src.data import PHASES, load_dce_triplet, load_samples, row_sources, split_by_patient_fold
from src.experiment_config import load_experiment_config


def create_preview(config_path: Path, output_dir: Path) -> Path:
    config = load_experiment_config(config_path)
    if not config.data.augmentation.enabled:
        raise ValueError("Esta configuracion no activa aumentos")
    splits = split_by_patient_fold(load_samples(config.data.root), config.data.validation_fold)
    generator = torch.Generator().manual_seed(config.training.seed)
    transform = SharedRandomAffine(config.data.augmentation)
    fig, axes = plt.subplots(6, 3, figsize=(11, 17), constrained_layout=True)
    records = []
    for label in (0, 1):
        # Selección fija solo dentro de train; no busca un resultado favorable.
        row = splits.train.loc[splits.train.pCR == label].sort_values("sample_id").iloc[0]
        image = load_dce_triplet(row_sources(row, config.data.root))
        params = [transform.sample_parameters(generator=generator) for _ in range(2)]
        variants = [image, *(apply_shared_affine(image, param) for param in params)]
        for phase_index, phase in enumerate(PHASES):
            for column, variant in enumerate(variants):
                ax = axes[label * 3 + phase_index, column]
                ax.imshow(variant[phase_index].numpy(), cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                if phase_index == 0:
                    title = "Original" if column == 0 else (
                        f"Giro {params[column - 1].angle_degrees:.1f}° · "
                        f"dx {params[column - 1].translation_x_pixels:.1f}, "
                        f"dy {params[column - 1].translation_y_pixels:.1f} px")
                    ax.set_title(f"pCR={label} · {title}", fontsize=10)
                ax.text(0.02, 0.96, phase, transform=ax.transAxes, color="white",
                        va="top", bbox=dict(facecolor="black", alpha=0.6))
        records.append({"sample_id": str(row.sample_id), "patient_id": str(row.patient_id),
                        "pCR": label, "parameters": [asdict(param) for param in params]})
    fig.suptitle("E09 · Misma geometría en PRE / EARLY / LATE\n"
                 "Dos ejemplos de train; no son pacientes nuevas. Uso educativo, no clínico.", fontsize=14)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "original_vs_augmented.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    (output_dir / "preview.json").write_text(json.dumps({"config": str(config_path),
        "seed": config.training.seed, "source": "train_only", "test_pixels_read": False,
        "augmentation": asdict(config.data.augmentation), "examples": records}, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/E09_shared_affine.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/data_augmentation/E09_shared_affine"))
    args = parser.parse_args()
    print(create_preview(args.config, args.output_dir))
