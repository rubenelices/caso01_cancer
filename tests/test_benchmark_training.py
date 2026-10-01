"""Pruebas de los diagnosticos y el resumen del benchmark."""

from __future__ import annotations

from src.benchmark_training import diagnose_bottleneck, format_summary


def test_diagnose_bottleneck_detects_data_loading() -> None:
    diagnosis = diagnose_bottleneck(
        {
            "data_loading": 0.7,
            "host_to_device": 0.1,
            "compute": 0.2,
            "total": 1.0,
        }
    )
    assert diagnosis["bottleneck"] == "data_loading"
    assert diagnosis["component_fraction_of_total"]["data_loading"] == 0.7
    assert "GPU" in diagnosis["explanation"]


def test_format_summary_contains_decision_values() -> None:
    report = {
        "environment": {
            "device": "cuda",
            "gpu_name": "AMD Radeon RX 6700 XT",
            "gpu_architecture_reported": "gfx1030",
            "rocm_hip_version": "7.14",
            "hsa_override_gfx_version": "10.3.0",
        },
        "settings": {
            "batch_size": 16,
            "num_workers": 4,
            "mixed_precision": True,
        },
        "timings_seconds_per_step": {
            "data_loading": 0.1,
            "host_to_device": 0.02,
            "compute": 0.2,
            "total": 0.32,
        },
        "throughput_samples_per_second": {
            "compute_only": 100.0,
            "end_to_end": 50.0,
        },
        "estimates": {
            "training_minutes_per_epoch": 3.0,
            "training_hours_for_configured_epochs": 2.5,
            "configured_epochs": 50,
        },
        "diagnosis": {
            "bottleneck": "compute",
            "component_fraction_of_total": {
                "data_loading": 0.3125,
                "host_to_device": 0.0625,
                "compute": 0.625,
            },
            "explanation": "El calculo domina.",
        },
        "memory": {
            "gpu_peak_memory_allocated_bytes": 1024**3,
            "gpu_total_memory_bytes": 12 * 1024**3,
        },
    }
    summary = format_summary(report)
    assert "AMD Radeon RX 6700 XT" in summary
    assert "Batch/workers: 16/4" in summary
    assert "Precision mixta: si" in summary
    assert "Override HSA: 10.3.0" in summary
    assert "VRAM maxima: 1.00 de 12.00 GiB" in summary
    assert "Cuello de botella: calculo de la CNN" in summary
    assert "2.50 h para 50 epocas" in summary
