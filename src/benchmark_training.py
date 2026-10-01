"""Mide por separado datos, transferencia y calculo del entrenamiento CNN."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

import psutil
import torch
from torch import Tensor, nn

from src.architectures import BreastPCRNet, trainable_parameter_count
from src.data import LoaderConfig, create_dataloaders, load_samples, split_by_patient_fold
from src.experiment_config import load_experiment_config
from src.train import build_criterion, choose_device, set_reproducibility


def _synchronize(device: torch.device) -> None:
    """Espera a que CUDA/ROCm termine para que el cronometro sea correcto."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _training_step(
    model: nn.Module,
    images: Tensor,
    targets: Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        loss = criterion(model(images), targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def _time_compute_only(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batch_size: int,
    warmup_steps: int,
    measured_steps: int,
    use_amp: bool,
) -> list[float]:
    """Cronometra la CNN con tensores ya situados en el dispositivo."""

    images = torch.rand(batch_size, 3, 256, 256, device=device)
    targets = torch.randint(0, 2, (batch_size,), device=device).float()
    for _ in range(warmup_steps):
        _training_step(
            model, images, targets, criterion, optimizer, scaler, device, use_amp
        )
    _synchronize(device)

    durations: list[float] = []
    for _ in range(measured_steps):
        start = time.perf_counter()
        _training_step(
            model, images, targets, criterion, optimizer, scaler, device, use_amp
        )
        _synchronize(device)
        durations.append(time.perf_counter() - start)
    return durations


def _next_batch(iterator: Any, loader: Any) -> tuple[Any, Any]:
    """Obtiene el siguiente batch y reinicia el loader solo si se agota."""

    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _time_end_to_end(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    warmup_steps: int,
    measured_steps: int,
    use_amp: bool,
) -> tuple[dict[str, list[float]], int]:
    """Cronometra lectura, copia y calculo usando imagenes reales."""

    iterator = iter(loader)
    for _ in range(warmup_steps):
        batch, iterator = _next_batch(iterator, loader)
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        _training_step(
            model, images, targets, criterion, optimizer, scaler, device, use_amp
        )
        _synchronize(device)

    timings = {"data_loading": [], "host_to_device": [], "compute": [], "total": []}
    samples_processed = 0
    for _ in range(measured_steps):
        total_start = time.perf_counter()

        data_start = time.perf_counter()
        batch, iterator = _next_batch(iterator, loader)
        timings["data_loading"].append(time.perf_counter() - data_start)

        transfer_start = time.perf_counter()
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        _synchronize(device)
        timings["host_to_device"].append(time.perf_counter() - transfer_start)

        compute_start = time.perf_counter()
        _training_step(
            model, images, targets, criterion, optimizer, scaler, device, use_amp
        )
        _synchronize(device)
        timings["compute"].append(time.perf_counter() - compute_start)

        timings["total"].append(time.perf_counter() - total_start)
        samples_processed += int(images.shape[0])
    return timings, samples_processed


def diagnose_bottleneck(timings: dict[str, float]) -> dict[str, Any]:
    """Resume que parte domina el tiempo completo medido."""

    total = timings["total"]
    components = {
        "data_loading": timings["data_loading"],
        "host_to_device": timings["host_to_device"],
        "compute": timings["compute"],
    }
    fractions = {
        name: (duration / total if total > 0 else 0.0)
        for name, duration in components.items()
    }
    bottleneck = max(components, key=components.__getitem__)
    explanations = {
        "data_loading": (
            "La mayor espera esta en leer y preparar los PNG; la GPU queda parcialmente "
            "ociosa. Conviene revisar disco y numero de workers."
        ),
        "host_to_device": (
            "La copia de los batches a la GPU domina el paso; conviene revisar "
            "pin_memory y la conexion CPU-GPU."
        ),
        "compute": (
            "El calculo de la CNN domina el paso, que es lo esperable cuando la GPU "
            "esta bien alimentada."
        ),
    }
    return {
        "bottleneck": bottleneck,
        "component_fraction_of_total": fractions,
        "explanation": explanations[bottleneck],
    }


def _device_environment(device: torch.device) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": str(device),
        "rocm_hip_version": torch.version.hip,
        "cuda_version": torch.version.cuda,
        "hsa_override_gfx_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_architecture_reported": getattr(properties, "gcnArchName", None),
                "gpu_total_memory_bytes": properties.total_memory,
            }
        )
    return environment


def format_summary(report: dict[str, Any]) -> str:
    """Genera un resumen corto para leer y pegar desde la terminal."""

    settings = report["settings"]
    timings = report["timings_seconds_per_step"]
    throughput = report["throughput_samples_per_second"]
    estimates = report["estimates"]
    diagnosis = report["diagnosis"]
    environment = report["environment"]
    memory = report.get("memory", {})
    fractions = diagnosis.get("component_fraction_of_total", {})
    bottleneck_labels = {
        "data_loading": "lectura y preparacion de datos",
        "host_to_device": "copia de CPU a GPU",
        "compute": "calculo de la CNN",
    }
    peak_memory = memory.get("gpu_peak_memory_allocated_bytes")
    total_memory = memory.get("gpu_total_memory_bytes")
    memory_line = "VRAM: no disponible"
    if peak_memory is not None and total_memory is not None:
        gibibyte = 1024**3
        memory_line = (
            f"VRAM maxima: {peak_memory / gibibyte:.2f} de "
            f"{total_memory / gibibyte:.2f} GiB"
        )

    def timing_line(label: str, key: str) -> str:
        percentage = 100 * fractions.get(key, 0.0)
        return f"{label}: {timings[key]:.4f} s/paso ({percentage:.1f} %)"

    return "\n".join(
        [
            "=== BENCHMARK CNN ===",
            f"Dispositivo: {environment.get('gpu_name') or environment['device']}",
            f"Arquitectura informada: {environment.get('gpu_architecture_reported')}",
            f"ROCm/HIP: {environment.get('rocm_hip_version')}",
            f"Override HSA: {environment.get('hsa_override_gfx_version')}",
            f"Batch/workers: {settings['batch_size']}/{settings['num_workers']}",
            f"Precision mixta: {'si' if settings['mixed_precision'] else 'no'}",
            timing_line("Lectura de datos", "data_loading"),
            timing_line("Copia a GPU", "host_to_device"),
            timing_line("Calculo CNN real", "compute"),
            f"Total real: {timings['total']:.4f} s/paso",
            f"Calculo puro: {throughput['compute_only']:.2f} imagenes/s",
            f"Proceso completo: {throughput['end_to_end']:.2f} imagenes/s",
            memory_line,
            (
                "Cuello de botella: "
                f"{bottleneck_labels.get(diagnosis['bottleneck'], diagnosis['bottleneck'])}"
            ),
            diagnosis["explanation"],
            (
                "Estimacion de entrenamiento: "
                f"{estimates['training_minutes_per_epoch']:.2f} min/epoca; "
                f"{estimates['training_hours_for_configured_epochs']:.2f} h para "
                f"{estimates['configured_epochs']} epocas."
            ),
            "La estimacion no incluye validacion, checkpoints ni graficas.",
        ]
    )


def benchmark(
    config_path: Path,
    output: Path,
    batch_size: int | None = None,
    steps: int = 10,
    warmup_steps: int = 3,
    num_workers: int | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps debe ser positivo")
    if warmup_steps < 1:
        raise ValueError("warmup_steps debe ser positivo")

    config = load_experiment_config(config_path)
    resolved_batch_size = (
        config.data.batch_size if batch_size is None else batch_size
    )
    resolved_num_workers = (
        config.data.num_workers if num_workers is None else num_workers
    )
    if resolved_batch_size < 1:
        raise ValueError("batch_size debe ser positivo")
    if resolved_num_workers < 0:
        raise ValueError("num_workers no puede ser negativo")

    device = choose_device(device_name)
    set_reproducibility(config.training.seed)
    samples = load_samples(config.data.root)
    splits = split_by_patient_fold(samples, config.data.validation_fold)
    pin_memory = config.data.pin_memory and device.type == "cuda"
    loaders = create_dataloaders(
        splits,
        config.data.root,
        LoaderConfig(
            batch_size=resolved_batch_size,
            num_workers=resolved_num_workers,
            pin_memory=pin_memory,
            persistent_workers=resolved_num_workers > 0,
            seed=config.training.seed,
        ),
    )

    model = BreastPCRNet(config.model).to(device).train()
    criterion, _ = build_criterion(config, splits.train, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    use_amp = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    process = psutil.Process()
    memory_before = process.memory_info().rss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    compute_only_durations = _time_compute_only(
        model,
        criterion,
        optimizer,
        scaler,
        device,
        resolved_batch_size,
        warmup_steps,
        steps,
        use_amp,
    )
    real_timings, samples_processed = _time_end_to_end(
        model,
        loaders.train,
        criterion,
        optimizer,
        scaler,
        device,
        warmup_steps,
        steps,
        use_amp,
    )

    timing_means = {name: mean(values) for name, values in real_timings.items()}
    compute_only_elapsed = sum(compute_only_durations)
    end_to_end_elapsed = sum(real_timings["total"])
    compute_only_throughput = resolved_batch_size * steps / compute_only_elapsed
    end_to_end_throughput = samples_processed / end_to_end_elapsed
    estimated_epoch = len(splits.train) / end_to_end_throughput
    memory_after = process.memory_info().rss

    environment = _device_environment(device)
    runtime_notes = [
        "El primer arranque de MIOpen puede compilar o seleccionar kernels; se excluye mediante warm-up.",
        "La estimacion usa solo train; validacion, checkpoints y graficas anaden tiempo.",
    ]
    if environment["hsa_override_gfx_version"]:
        runtime_notes.append(
            "HSA_OVERRIDE_GFX_VERSION esta activo: ROCm puede informar una arquitectura distinta de la GPU nativa."
        )

    report: dict[str, Any] = {
        "schema_version": 2,
        "config": str(config_path),
        "parameter_count": trainable_parameter_count(model),
        "environment": environment,
        "settings": {
            "batch_size": resolved_batch_size,
            "num_workers": resolved_num_workers,
            "pin_memory": pin_memory,
            "mixed_precision": use_amp,
            "warmup_steps": warmup_steps,
            "measured_steps": steps,
        },
        "timings_seconds_per_step": {
            **timing_means,
            "compute_only": mean(compute_only_durations),
        },
        "throughput_samples_per_second": {
            "compute_only": compute_only_throughput,
            "end_to_end": end_to_end_throughput,
        },
        "estimates": {
            "training_seconds_per_epoch": estimated_epoch,
            "training_minutes_per_epoch": estimated_epoch / 60,
            "training_hours_for_configured_epochs": (
                estimated_epoch * config.training.epochs / 3600
            ),
            "configured_epochs": config.training.epochs,
            "includes_validation_and_checkpoints": False,
        },
        "memory": {
            "process_memory_delta_bytes": memory_after - memory_before,
            "gpu_peak_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "gpu_total_memory_bytes": environment.get("gpu_total_memory_bytes"),
        },
        "diagnosis": diagnose_bottleneck(timing_means),
        "notes": runtime_notes,
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Por defecto usa el batch_size de la configuracion.",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Por defecto usa num_workers de la configuracion.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = benchmark(
        config_path=args.config,
        output=args.output,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        device_name=args.device,
    )
    print(format_summary(report))
    print(f"Informe JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
