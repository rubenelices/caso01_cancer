"""Lectura de métricas agregadas para el cuaderno; sin imágenes ni entrenamiento."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_results(root: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Registro histórico como respaldo; informes locales válidos como fuente actual.

    No busca ejecuciones arbitrarias ni lee predicciones individuales. E05 Mac
    se incorpora al ejecutar de nuevo el cuaderno cuando su summary esté listo.
    """
    registry = json.loads((root / "experimentos/resultados_registrados.json").read_text())
    if registry["schema_version"] != 1:
        raise ValueError("Versión de registro no soportada")
    rows, warnings = [], []
    for saved in registry["runs"]:
        row = dict(saved)
        directory = root / row["report_dir"]
        row["local_history_available"] = False
        path = directory / "summary.json"
        if path.is_file():
            summary = json.loads(path.read_text())
            resolved = json.loads((directory / "config_resolved.json").read_text())
            if (summary.get("experiment_id") != row["experiment_id"]
                    or resolved.get("experiment_id") != row["experiment_id"]
                    or summary.get("status") != "complete"
                    or summary.get("scientific_result") is not True
                    or summary.get("test_evaluated") is not False
                    or resolved["data"].get("train_patients_per_class") is not None
                    or resolved["data"].get("validation_patients_per_class") is not None
                    or resolved["data"]["validation_fold"] != row["validation_fold"]
                    or resolved["training"]["seed"] != row["seed"]
                    or resolved["training"]["threshold"] != row["threshold"]
                    or summary["validation_metrics"].get("aggregation") != "mean"
                    or summary["validation_metrics"]["patient"]["n"] != row["validation_patients"]):
                raise ValueError(f"Informe incompatible o no científico: {row['run_id']}")
            metrics = summary["validation_metrics"]["patient"]
            row.update(status="complete", source="local_artifacts", best_epoch=summary["best_epoch"],
                roc_auc=metrics["roc_auc"], pr_auc=metrics["pr_auc"],
                sensitivity=metrics["sensitivity"], specificity=metrics["specificity"],
                balanced_accuracy=metrics["balanced_accuracy"], parameter_count=summary["parameter_count"],
                total_seconds=summary["total_seconds"], epochs_completed=summary["epochs_completed"],
                device=summary.get("environment", {}).get("device", "no informado"),
                source_note="Informe agregado local validado; no se leen identificadores de pacientes.")
            history_path = directory / "history.csv"
            if history_path.is_file():
                history = pd.read_csv(history_path)
                if (len(history) != summary["epochs_completed"] or history.epoch.duplicated().any()
                        or summary["best_epoch"] not in history.epoch.tolist()):
                    raise ValueError(f"Historial incompleto: {row['run_id']}")
                row["local_history_available"] = True
        elif row["status"] == "complete":
            warnings.append({"run_id": row["run_id"],
                             "note": "Solo resumen comunicado; no reconstruir curvas ni incertidumbre."})
        rows.append(row)
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Identidades de ejecución duplicadas")
    return pd.DataFrame(rows), warnings


def load_histories(root: Path, results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {row.run_id: pd.read_csv(root / row.report_dir / "history.csv")
            for row in results.itertuples() if row.local_history_available}


def load_cohort_reports(root: Path) -> list[dict]:
    """Solo informes conocidos; devuelve tablas agregadas, no CSV de pacientes."""
    reports = []
    snapshot_path = root / "experimentos/cohortes_registradas.json"
    saved = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else {"reports": []}
    snapshots = {report["name"]: report for report in saved["reports"]}
    for name in ("E08_E10_mac", "E05_E08_E10_mac"):
        path = root / "reports/cohorts" / name / "cohort_summary.json"
        if path.is_file():
            report = json.loads(path.read_text())
            report["notebook_source"] = "Informe local"
        elif name in snapshots:
            report = dict(snapshots[name])
            report["notebook_source"] = "Snapshot agregado registrado (sin archivos locales)"
        else:
            continue
        if report.get("scope") != "internal_validation_only" or report.get("test_evaluated") is not False:
            raise ValueError("Informe de cohortes fuera de validación interna")
        reports.append(report)
    return reports
