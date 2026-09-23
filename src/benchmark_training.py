"""Benchmark corto para decidir si entrenar localmente o esperar a la GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil
import torch
from torch import nn

from src.architectures import BreastPCRNet, trainable_parameter_count
from src.data import LoaderConfig, create_dataloaders, load_samples, split_by_patient_fold
from src.experiment_config import load_experiment_config
from src.train import choose_device, set_reproducibility


def benchmark(
    config_path: Path,
    output: Path,
    batch_size: int = 4,
    steps: int = 4,
    device_name: str = "auto",
) -> dict[str, object]:
    config = load_experiment_config(config_path)
    device = choose_device(device_name)
    set_reproducibility(config.training.seed)
    samples = load_samples(config.data.root)
    splits = split_by_patient_fold(samples, config.data.validation_fold)
    loaders = create_dataloaders(
        splits,
        config.data.root,
        LoaderConfig(batch_size=batch_size, num_workers=0, pin_memory=False, seed=42),
    )
    model = BreastPCRNet(config.model).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    iterator = iter(loaders.train)
    process = psutil.Process()
    memory_before = process.memory_info().rss

    durations: list[float] = []
    processed = 0
    for step in range(steps + 1):
        start = time.perf_counter()
        batch = next(iterator)
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        if step > 0:  # primer paso: calentamiento
            durations.append(duration)
            processed += images.shape[0]

    elapsed = sum(durations)
    throughput = processed / elapsed
    estimated_epoch = len(splits.train) / throughput
    estimated_total = estimated_epoch * config.training.epochs
    memory_after = process.memory_info().rss
    report: dict[str, object] = {
        "config": str(config_path),
        "device": str(device),
        "batch_size": batch_size,
        "measured_steps": steps,
        "parameter_count": trainable_parameter_count(model),
        "samples_per_second": throughput,
        "seconds_per_step_mean": elapsed / steps,
        "estimated_seconds_per_epoch": estimated_epoch,
        "estimated_hours_for_configured_epochs": estimated_total / 3600,
        "configured_epochs": config.training.epochs,
        "process_memory_delta_bytes": memory_after - memory_before,
        "recommend_gpu": estimated_epoch > 120 or estimated_total > 1800,
        "note": "Estimacion aproximada; validacion y escritura de checkpoints anaden tiempo.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/E02_base_normal.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/benchmarks/base_local.json")
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = benchmark(args.config, args.output, args.batch_size, args.steps, args.device)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
