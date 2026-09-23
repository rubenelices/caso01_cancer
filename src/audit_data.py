#!/usr/bin/env python3
"""Auditoria reproducible del dataset docente BreastDCEDL.

El script no modifica los datos. Comprueba metadatos, relaciones, separaciones
por paciente e integridad de las imagenes, y genera un informe en JSON/HTML con
figuras descriptivas.

Uso recomendado desde la raiz del repositorio::

    python -m src.audit_data

Para una comprobacion rapida durante el desarrollo::

    python -m src.audit_data --image-check sample --sample-size 200
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-caso-cancer")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


REQUIRED_SAMPLE_COLUMNS = {
    "sample_id",
    "patient_id",
    "split",
    "path_pre",
    "path_early",
    "path_late",
    "slice_index",
    "pCR",
    "fold",
}
REQUIRED_PATIENT_COLUMNS = {"pid", "pCR", "dataset", "split"}
PHASE_COLUMNS = {
    "PRE": "path_pre",
    "EARLY": "path_early",
    "LATE": "path_late",
}
VALID_SPLITS = {"train", "test"}
VALID_LABELS = {0, 1}
EXPECTED_IMAGE_SIZE = (256, 256)


@dataclass
class AuditIssue:
    """Hallazgo estructurado de la auditoria."""

    severity: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditState:
    """Acumula hallazgos y resultados durante la auditoria."""

    issues: list[AuditIssue] = field(default_factory=list)

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        **details: Any,
    ) -> None:
        self.issues.append(AuditIssue(severity, code, message, details))

    @property
    def critical_count(self) -> int:
        return sum(issue.severity == "critical" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)


def _json_safe(value: Any) -> Any:
    """Convierte tipos de NumPy/Pandas y no finitos a JSON valido."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return _json_safe(table.reset_index().to_dict(orient="records"))


def load_metadata(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los dos indices principales sin alterar sus columnas."""

    metadata = root / "metadata"
    samples = pd.read_csv(metadata / "samples.csv")
    patients = pd.read_csv(metadata / "patients.csv")
    return samples, patients


def profile_dataframe(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Perfila tipos, nulos, cardinalidad y estadisticos por columna."""

    result: dict[str, dict[str, Any]] = {}
    row_count = len(data)
    for column in data.columns:
        series = data[column]
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        item: dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_pct": 100.0 * null_count / row_count if row_count else 0.0,
            "unique_count": unique_count,
            "cardinality_pct": 100.0 * unique_count / row_count if row_count else 0.0,
            "top_values": {
                str(key): int(value)
                for key, value in series.value_counts(dropna=False).head(5).items()
            },
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            item.update(
                {
                    "min": numeric.min(),
                    "max": numeric.max(),
                    "mean": numeric.mean(),
                    "median": numeric.median(),
                    "std": numeric.std(),
                    "zeros": int((numeric == 0).sum()),
                    "negatives": int((numeric < 0).sum()),
                }
            )
        else:
            text = series.dropna().astype(str)
            item.update(
                {
                    "min_length": int(text.str.len().min()) if len(text) else None,
                    "max_length": int(text.str.len().max()) if len(text) else None,
                }
            )
        result[str(column)] = _json_safe(item)
    return result


def validate_metadata(
    samples: pd.DataFrame,
    patients: pd.DataFrame,
    state: AuditState | None = None,
) -> AuditState:
    """Valida esquema, claves, etiquetas, splits y folds por paciente."""

    state = state or AuditState()
    missing_samples = sorted(REQUIRED_SAMPLE_COLUMNS - set(samples.columns))
    missing_patients = sorted(REQUIRED_PATIENT_COLUMNS - set(patients.columns))
    if missing_samples:
        state.add(
            "critical",
            "missing_sample_columns",
            "Faltan columnas obligatorias en samples.csv.",
            columns=missing_samples,
        )
    if missing_patients:
        state.add(
            "critical",
            "missing_patient_columns",
            "Faltan columnas obligatorias en patients.csv.",
            columns=missing_patients,
        )
    if missing_samples or missing_patients:
        return state

    duplicate_samples = int(samples["sample_id"].duplicated().sum())
    duplicate_patients = int(patients["pid"].duplicated().sum())
    if duplicate_samples:
        state.add(
            "critical",
            "duplicate_sample_id",
            "sample_id no es una clave unica.",
            duplicates=duplicate_samples,
        )
    if duplicate_patients:
        state.add(
            "critical",
            "duplicate_patient_id",
            "pid no es una clave unica en patients.csv.",
            duplicates=duplicate_patients,
        )

    invalid_sample_labels = sorted(set(samples["pCR"].dropna().unique()) - VALID_LABELS)
    invalid_patient_labels = sorted(set(patients["pCR"].dropna().unique()) - VALID_LABELS)
    if invalid_sample_labels or invalid_patient_labels:
        state.add(
            "critical",
            "invalid_labels",
            "La etiqueta pCR contiene valores distintos de 0 y 1.",
            samples=invalid_sample_labels,
            patients=invalid_patient_labels,
        )
    if samples["pCR"].isna().any() or patients["pCR"].isna().any():
        state.add(
            "critical",
            "missing_labels",
            "Hay etiquetas pCR ausentes.",
            sample_nulls=int(samples["pCR"].isna().sum()),
            patient_nulls=int(patients["pCR"].isna().sum()),
        )

    invalid_splits = sorted(
        (set(samples["split"].dropna()) | set(patients["split"].dropna()))
        - VALID_SPLITS
    )
    if invalid_splits:
        state.add(
            "critical",
            "invalid_splits",
            "Existen splits no reconocidos.",
            values=invalid_splits,
        )

    sample_patients = set(samples["patient_id"])
    patient_ids = set(patients["pid"])
    missing_fk = sorted(sample_patients - patient_ids)
    without_samples = sorted(patient_ids - sample_patients)
    if missing_fk:
        state.add(
            "critical",
            "orphan_samples",
            "Hay muestras cuyo patient_id no existe en patients.csv.",
            count=len(missing_fk),
            examples=missing_fk[:10],
        )
    if without_samples:
        state.add(
            "warning",
            "patients_without_samples",
            "Hay pacientes sin cortes en samples.csv.",
            count=len(without_samples),
            examples=without_samples[:10],
        )

    split_counts = samples.groupby("patient_id")["split"].nunique()
    leaked_split = split_counts[split_counts > 1].index.tolist()
    if leaked_split:
        state.add(
            "critical",
            "patient_split_leakage",
            "Una o mas pacientes aparecen en varios splits.",
            count=len(leaked_split),
            examples=leaked_split[:10],
        )

    label_counts = samples.groupby("patient_id")["pCR"].nunique(dropna=False)
    inconsistent_labels = label_counts[label_counts > 1].index.tolist()
    if inconsistent_labels:
        state.add(
            "critical",
            "inconsistent_patient_label",
            "La etiqueta cambia entre cortes de una misma paciente.",
            count=len(inconsistent_labels),
            examples=inconsistent_labels[:10],
        )

    fold_counts = samples.loc[samples["split"] == "train"].groupby("patient_id")[
        "fold"
    ].nunique(dropna=False)
    multi_fold = fold_counts[fold_counts > 1].index.tolist()
    if multi_fold:
        state.add(
            "critical",
            "patient_fold_leakage",
            "Una o mas pacientes de train aparecen en varios folds.",
            count=len(multi_fold),
            examples=multi_fold[:10],
        )

    train_folds = set(samples.loc[samples["split"] == "train", "fold"].dropna())
    test_folds = set(samples.loc[samples["split"] == "test", "fold"].dropna())
    if not train_folds.issubset(set(range(5))):
        state.add(
            "critical",
            "invalid_train_fold",
            "Train contiene folds fuera del rango 0-4.",
            values=sorted(train_folds),
        )
    if test_folds != {-1}:
        state.add(
            "warning",
            "unexpected_test_fold",
            "Se esperaba fold=-1 para todas las muestras de test.",
            values=sorted(test_folds),
        )

    patient_view = patients.set_index("pid")[["pCR", "split"]]
    sample_view = samples.groupby("patient_id")[["pCR", "split"]].first()
    common = sample_view.index.intersection(patient_view.index)
    label_mismatch = int(
        (sample_view.loc[common, "pCR"] != patient_view.loc[common, "pCR"]).sum()
    )
    split_mismatch = int(
        (sample_view.loc[common, "split"] != patient_view.loc[common, "split"]).sum()
    )
    if label_mismatch:
        state.add(
            "critical",
            "metadata_label_mismatch",
            "pCR no coincide entre samples.csv y patients.csv.",
            count=label_mismatch,
        )
    if split_mismatch:
        state.add(
            "critical",
            "metadata_split_mismatch",
            "split no coincide entre samples.csv y patients.csv.",
            count=split_mismatch,
        )

    if not state.critical_count:
        state.add(
            "info",
            "metadata_integrity_ok",
            "Claves, etiquetas, splits y folds son coherentes por paciente.",
        )
    return state


def analyze_distributions(
    samples: pd.DataFrame,
    patients: pd.DataFrame,
    state: AuditState,
) -> None:
    """Registra riesgos estadisticos que no invalidan la integridad."""

    train_patients = patients.loc[patients["split"] == "train"]
    prevalence = float(train_patients["pCR"].mean())
    state.add(
        "warning",
        "class_imbalance",
        "La clase pCR es minoritaria; accuracy aislada seria enganosa.",
        train_patient_prevalence_pct=100.0 * prevalence,
        majority_baseline_accuracy_pct=100.0 * max(prevalence, 1.0 - prevalence),
    )

    patient_fold = (
        samples.loc[samples["split"] == "train"]
        .groupby("patient_id")["fold"]
        .first()
        .rename("fold")
    )
    fold_rates = train_patients.set_index("pid").join(patient_fold).groupby("fold")["pCR"].mean()
    fold_spread = float(fold_rates.max() - fold_rates.min())
    if fold_spread > 0.05:
        state.add(
            "warning",
            "fold_prevalence_variation",
            "Los folds son seguros por paciente, pero su prevalencia de pCR no es uniforme.",
            spread_percentage_points=100.0 * fold_spread,
            rates_pct={str(key): 100.0 * value for key, value in fold_rates.items()},
        )

    cohort_rates = patients.groupby("dataset")["pCR"].mean()
    cohort_spread = float(cohort_rates.max() - cohort_rates.min())
    if cohort_spread > 0.05:
        state.add(
            "warning",
            "cohort_prevalence_variation",
            "La prevalencia de pCR cambia entre cohortes; existe riesgo de confundir cohorte con respuesta.",
            spread_percentage_points=100.0 * cohort_spread,
            rates_pct={str(key): 100.0 * value for key, value in cohort_rates.items()},
        )

    columns_with_missing = patients.columns[patients.isna().any()].tolist()
    if columns_with_missing:
        state.add(
            "info",
            "patient_metadata_missingness",
            "patients.csv contiene valores ausentes que deben tratarse como no disponibles, no como cero.",
            columns=columns_with_missing,
        )


def _resolved_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_paths(
    root: Path,
    samples: pd.DataFrame,
    state: AuditState,
) -> dict[str, Any]:
    """Comprueba seguridad, unicidad, existencia y ficheros extra."""

    expected: list[Path] = []
    bad_suffix: list[str] = []
    escaped: list[str] = []
    for row in samples.itertuples(index=False):
        for phase, column in PHASE_COLUMNS.items():
            relative = Path(getattr(row, column))
            path = root / relative
            expected.append(path)
            if relative.suffix.lower() != ".png" or not relative.stem.endswith(f"_{phase}"):
                bad_suffix.append(str(relative))
            if not _resolved_inside(path, root):
                escaped.append(str(relative))

    string_paths = [str(path) for path in expected]
    duplicate_count = len(string_paths) - len(set(string_paths))
    missing = [str(path.relative_to(root)) for path in expected if not path.is_file()]
    found = {path.resolve() for path in (root / "dataset").rglob("*.png")}
    expected_set = {path.resolve() for path in expected}
    extras = sorted(str(path.relative_to(root.resolve())) for path in found - expected_set)

    if escaped:
        state.add(
            "critical",
            "paths_outside_root",
            "Hay rutas que salen de la raiz del dataset.",
            count=len(escaped),
            examples=escaped[:10],
        )
    if bad_suffix:
        state.add(
            "critical",
            "invalid_phase_path",
            "Hay rutas cuyo sufijo no coincide con la fase declarada.",
            count=len(bad_suffix),
            examples=bad_suffix[:10],
        )
    if duplicate_count:
        state.add(
            "critical",
            "duplicate_image_paths",
            "Una misma ruta de imagen esta referenciada mas de una vez.",
            count=duplicate_count,
        )
    if missing:
        state.add(
            "critical",
            "missing_images",
            "Faltan imagenes referenciadas por samples.csv.",
            count=len(missing),
            examples=missing[:10],
        )
    if extras:
        state.add(
            "warning",
            "unreferenced_images",
            "Hay PNG en dataset/ que no aparecen en samples.csv.",
            count=len(extras),
            examples=extras[:10],
        )
    if not (escaped or bad_suffix or duplicate_count or missing or extras):
        state.add(
            "info",
            "path_integrity_ok",
            "Todas las rutas existen, son unicas y corresponden con su fase.",
            expected=len(expected),
        )

    return {
        "expected_images": len(expected),
        "found_png": len(found),
        "missing": len(missing),
        "extra": len(extras),
        "duplicate_references": duplicate_count,
    }


def _select_rows(
    samples: pd.DataFrame,
    mode: str,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if mode == "none":
        return samples.iloc[0:0]
    if mode == "all" or sample_size >= len(samples):
        return samples

    chosen: list[pd.DataFrame] = []
    groups = samples.groupby(["split", "pCR"], dropna=False)
    per_group = max(1, sample_size // max(1, groups.ngroups))
    for _, group in groups:
        chosen.append(group.sample(min(per_group, len(group)), random_state=seed))
    selected = pd.concat(chosen).drop_duplicates("sample_id")
    if len(selected) < sample_size:
        remaining = samples.loc[~samples.index.isin(selected.index)]
        extra = remaining.sample(
            min(sample_size - len(selected), len(remaining)), random_state=seed
        )
        selected = pd.concat([selected, extra])
    return selected.iloc[:sample_size]


def audit_images(
    root: Path,
    samples: pd.DataFrame,
    state: AuditState,
    mode: str,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Abre imagenes y comprueba formato, modo, dimensiones y estadisticas."""

    selected = _select_rows(samples, mode, sample_size, seed)
    if selected.empty:
        state.add(
            "warning",
            "image_content_skipped",
            "Se omitio la apertura del contenido de las imagenes.",
        )
        return {"mode": mode, "samples_checked": 0, "images_checked": 0}

    errors: list[dict[str, str]] = []
    phase_means: list[dict[str, Any]] = []
    global_min = 255
    global_max = 0
    images_checked = 0

    total_images = len(selected) * len(PHASE_COLUMNS)
    print(f"Comprobando contenido de {total_images:,} imagenes ({mode})...")
    for position, row in enumerate(selected.itertuples(index=False), start=1):
        means: dict[str, float] = {}
        row_valid = True
        for phase, column in PHASE_COLUMNS.items():
            relative = Path(getattr(row, column))
            path = root / relative
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.format != "PNG":
                        raise ValueError(f"formato={image.format}, esperado=PNG")
                    if image.size != EXPECTED_IMAGE_SIZE:
                        raise ValueError(
                            f"tamano={image.size}, esperado={EXPECTED_IMAGE_SIZE}"
                        )
                    if image.mode != "L":
                        raise ValueError(f"modo={image.mode}, esperado=L")
                    array = np.asarray(image, dtype=np.uint8)
                    means[phase] = float(array.mean() / 255.0)
                    global_min = min(global_min, int(array.min()))
                    global_max = max(global_max, int(array.max()))
                    images_checked += 1
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                row_valid = False
                errors.append({"path": str(relative), "error": str(exc)})
        if row_valid:
            phase_means.append(
                {
                    "sample_id": row.sample_id,
                    "patient_id": row.patient_id,
                    "split": row.split,
                    "pCR": int(row.pCR),
                    **means,
                }
            )
        if position % 2000 == 0 or position == len(selected):
            print(f"  cortes comprobados: {position:,}/{len(selected):,}")

    if errors:
        state.add(
            "critical",
            "invalid_image_content",
            "Hay imagenes corruptas o con formato, modo o dimensiones incorrectos.",
            count=len(errors),
            examples=errors[:20],
        )
    else:
        state.add(
            "info",
            "image_content_ok",
            "Todas las imagenes abiertas cumplen formato PNG, modo L y 256 x 256.",
            images_checked=images_checked,
        )

    phase_frame = pd.DataFrame(phase_means)
    phase_summary: dict[str, Any] = {}
    if not phase_frame.empty:
        early_above_pre = phase_frame["EARLY"] > phase_frame["PRE"]
        late_above_early = phase_frame["LATE"] > phase_frame["EARLY"]
        patient_means = phase_frame.groupby("patient_id")[["PRE", "EARLY", "LATE"]].mean()
        phase_summary = {
            "whole_image_mean": {
                phase: float(phase_frame[phase].mean()) for phase in PHASE_COLUMNS
            },
            "cuts_early_above_pre_pct": float(100.0 * early_above_pre.mean()),
            "cuts_late_above_early_pct": float(100.0 * late_above_early.mean()),
            "patients_early_above_pre_pct": float(
                100.0 * (patient_means["EARLY"] > patient_means["PRE"]).mean()
            ),
            "patients_late_above_early_pct": float(
                100.0 * (patient_means["LATE"] > patient_means["EARLY"]).mean()
            ),
        }
        if phase_summary["patients_early_above_pre_pct"] < 95.0:
            state.add(
                "warning",
                "unexpected_phase_order",
                "Menos del 95 % de pacientes tienen media global EARLY > PRE. Revisar orden de fases.",
                percentage=phase_summary["patients_early_above_pre_pct"],
            )

    return _json_safe(
        {
            "mode": mode,
            "samples_checked": len(selected),
            "images_checked": images_checked,
            "errors": len(errors),
            "pixel_min": global_min if images_checked else None,
            "pixel_max": global_max if images_checked else None,
            "phase_summary": phase_summary,
        }
    )


def build_summary(samples: pd.DataFrame, patients: pd.DataFrame) -> dict[str, Any]:
    """Construye tablas agregadas a nivel de corte y paciente."""

    slices_per_patient = samples.groupby("patient_id").size()
    patient_fold = (
        samples.loc[samples["split"] == "train"]
        .groupby("patient_id")["fold"]
        .first()
        .rename("fold")
    )
    train_patients = patients.loc[patients["split"] == "train"].set_index("pid")
    fold_table = train_patients[["pCR"]].join(patient_fold).groupby("fold")["pCR"].agg(
        patients="count", positives="sum", prevalence="mean"
    )

    patient_split_class = patients.groupby(["split", "pCR"]).size().rename("patients")
    sample_split_class = samples.groupby(["split", "pCR"]).size().rename("cuts")
    cohort_table = patients.groupby(["dataset", "split", "pCR"]).size().rename("patients")
    cohort_prevalence = patients.groupby("dataset")["pCR"].agg(
        patients="count", positives="sum", prevalence="mean"
    )
    missingness = (
        patients.isna().mean().mul(100).sort_values(ascending=False).rename("missing_pct")
    )

    return _json_safe(
        {
            "patients": len(patients),
            "samples": len(samples),
            "expected_images": len(samples) * 3,
            "patient_split_class": _records(patient_split_class.to_frame()),
            "sample_split_class": _records(sample_split_class.to_frame()),
            "cohort_split_class": _records(cohort_table.to_frame()),
            "cohort_prevalence": _records(cohort_prevalence),
            "folds": _records(fold_table),
            "slices_per_patient": slices_per_patient.describe().to_dict(),
            "patient_missingness": _records(missingness.to_frame()),
        }
    )


def build_source_catalog(root: Path, output: Path) -> dict[str, Any]:
    """Genera un catalogo ligero de las fuentes estructuradas."""

    sources: list[dict[str, Any]] = []
    for path in sorted((root / "metadata").glob("*")):
        if not path.is_file():
            continue
        entry: dict[str, Any] = {
            "name": path.stem,
            "type": path.suffix.lower().lstrip("."),
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            entry.update({"rows": len(frame), "columns": list(frame.columns)})
        elif path.suffix.lower() == ".json":
            entry["keys"] = list(json.loads(path.read_text(encoding="utf-8")).keys())
        sources.append(entry)
    catalog = {"root": str(root), "sources": sources}
    (output / "source_catalog.json").write_text(
        json.dumps(_json_safe(catalog), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return catalog


def _save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def create_figures(
    root: Path,
    samples: pd.DataFrame,
    patients: pd.DataFrame,
    output: Path,
    seed: int,
) -> list[str]:
    """Genera figuras descriptivas y una galeria determinista."""

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    counts = patients.groupby(["split", "pCR"]).size().unstack(fill_value=0)
    counts = counts.rename(columns={0: "no pCR", 1: "pCR"})
    counts.plot(kind="bar", color=["#64748b", "#c0265e"], figsize=(7.5, 4.5))
    plt.title("Pacientes por split y clase")
    plt.xlabel("Split")
    plt.ylabel("Pacientes")
    plt.xticks(rotation=0)
    plt.legend(title="Clase")
    path = figures / "pacientes_split_clase.png"
    _save_current_figure(path)
    created.append(str(path.relative_to(output)))

    prevalence = patients.groupby("dataset")["pCR"].mean().sort_values()
    ax = prevalence.mul(100).plot(kind="bar", color="#0f766e", figsize=(7.5, 4.5))
    ax.axhline(patients["pCR"].mean() * 100, color="#b91c1c", linestyle="--", label="Global")
    plt.title("Prevalencia de pCR por cohorte")
    plt.xlabel("Cohorte")
    plt.ylabel("Pacientes con pCR (%)")
    plt.xticks(rotation=0)
    plt.legend()
    path = figures / "prevalencia_cohorte.png"
    _save_current_figure(path)
    created.append(str(path.relative_to(output)))

    patient_fold = (
        samples.loc[samples["split"] == "train"]
        .groupby("patient_id")["fold"]
        .first()
    )
    fold_data = patients.set_index("pid").join(patient_fold.rename("fold")).dropna(subset=["fold"])
    fold_prevalence = fold_data.groupby("fold")["pCR"].mean().mul(100)
    fold_prevalence.plot(kind="bar", color="#2563eb", figsize=(7.5, 4.5))
    plt.title("Prevalencia de pCR por fold de train")
    plt.xlabel("Fold")
    plt.ylabel("Pacientes con pCR (%)")
    plt.xticks(rotation=0)
    path = figures / "prevalencia_folds.png"
    _save_current_figure(path)
    created.append(str(path.relative_to(output)))

    slices = samples.groupby("patient_id").size()
    slices.value_counts().sort_index().plot(kind="bar", color="#7c3aed", figsize=(7.5, 4.5))
    plt.title("Numero de cortes por paciente")
    plt.xlabel("Cortes")
    plt.ylabel("Pacientes")
    plt.xticks(rotation=0)
    path = figures / "cortes_por_paciente.png"
    _save_current_figure(path)
    created.append(str(path.relative_to(output)))

    missingness = patients.isna().mean().mul(100)
    missingness = missingness[missingness > 0].sort_values(ascending=False)
    if not missingness.empty:
        missingness.plot(kind="bar", color="#d97706", figsize=(10, 4.8))
        plt.title("Valores ausentes en patients.csv")
        plt.xlabel("Columna")
        plt.ylabel("Ausentes (%)")
        plt.xticks(rotation=45, ha="right")
        path = figures / "valores_ausentes.png"
        _save_current_figure(path)
        created.append(str(path.relative_to(output)))

    # Una paciente de train por cohorte y clase. El corte central se usa solo
    # para visualizacion; no constituye una seleccion para entrenamiento.
    rng = np.random.default_rng(seed)
    gallery_rows: list[pd.Series] = []
    patient_train = patients.loc[patients["split"] == "train"]
    for (_, _), group in patient_train.groupby(["dataset", "pCR"]):
        patient_id = str(rng.choice(group["pid"].to_numpy()))
        cuts = samples.loc[samples["patient_id"] == patient_id].sort_values("slice_index")
        gallery_rows.append(cuts.iloc[len(cuts) // 2])

    if gallery_rows:
        rows = len(gallery_rows)
        figure, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows), squeeze=False)
        patient_lookup = patients.set_index("pid")
        for row_index, sample in enumerate(gallery_rows):
            arrays = []
            for phase, column in PHASE_COLUMNS.items():
                with Image.open(root / sample[column]) as image:
                    arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
            enhancement = arrays[1] - arrays[0]
            for column_index, (phase, array) in enumerate(zip(PHASE_COLUMNS, arrays)):
                axes[row_index, column_index].imshow(array, cmap="gray", vmin=0, vmax=1)
                axes[row_index, column_index].set_title(phase)
                axes[row_index, column_index].axis("off")
            vmax = max(float(np.quantile(np.abs(enhancement), 0.995)), 1e-6)
            axes[row_index, 3].imshow(enhancement, cmap="magma", vmin=0, vmax=vmax)
            axes[row_index, 3].set_title("EARLY - PRE")
            axes[row_index, 3].axis("off")
            patient = patient_lookup.loc[sample["patient_id"]]
            axes[row_index, 0].text(
                0.02,
                0.03,
                f"{str(patient['dataset']).upper()} | pCR={int(patient['pCR'])}\n{sample['patient_id']}",
                transform=axes[row_index, 0].transAxes,
                color="white",
                fontsize=8,
                fontweight="bold",
                va="bottom",
                ha="left",
                bbox={"facecolor": "black", "alpha": 0.7, "pad": 4, "edgecolor": "none"},
            )
        figure.suptitle("Galeria estratificada: una paciente por cohorte y clase", fontsize=14)
        path = figures / "galeria_cohortes_clases.png"
        figure.tight_layout(rect=[0, 0, 1, 0.98])
        figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        created.append(str(path.relative_to(output)))

    return created


def _html_table(records: Iterable[dict[str, Any]], columns: list[str] | None = None) -> str:
    frame = pd.DataFrame(list(records))
    if columns:
        existing = [column for column in columns if column in frame.columns]
        frame = frame[existing]
    return frame.to_html(index=False, border=0, classes="dataframe", escape=True)


def render_html(report: dict[str, Any], output: Path) -> Path:
    """Renderiza un informe autocontenido salvo por las figuras PNG."""

    summary = report["summary"]
    issues = report["issues"]
    status = "APROBADA" if report["status"] == "pass" else "FALLIDA"
    status_class = "ok" if report["status"] == "pass" else "critical"
    issue_rows = [
        {
            "severidad": item["severity"],
            "codigo": item["code"],
            "mensaje": item["message"],
            "detalles": json.dumps(item.get("details", {}), ensure_ascii=False),
        }
        for item in issues
    ]
    image_stats = report["image_audit"]
    phase = image_stats.get("phase_summary", {})
    figures_html = "".join(
        f'<figure><img src="{html.escape(path)}" alt="Figura de auditoria"></figure>'
        for path in report["figures"]
    )
    profile_rows = []
    for source, profile in report["profiles"].items():
        for column, values in profile.items():
            profile_rows.append(
                {
                    "fuente": source,
                    "columna": column,
                    "tipo": values.get("dtype"),
                    "nulos": values.get("null_count"),
                    "nulos_pct": round(values.get("null_pct", 0.0), 2),
                    "unicos": values.get("unique_count"),
                    "cardinalidad_pct": round(values.get("cardinality_pct", 0.0), 2),
                }
            )

    document = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Auditoria BreastDCEDL</title>
  <style>
    :root {{ --navy:#172033; --ink:#172033; --muted:#526076; --line:#d9e0ea; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#eef2f7; color:var(--ink); font-family:system-ui,sans-serif; }}
    header {{ background:var(--navy); color:white; padding:42px max(5vw,24px); }}
    header h1 {{ margin:0 0 8px; font-size:clamp(28px,4vw,48px); }}
    header p {{ margin:0; color:#cbd5e1; }}
    main {{ max-width:1200px; margin:0 auto; padding:32px 24px 60px; }}
    section {{ background:white; border:1px solid var(--line); border-radius:14px; padding:24px; margin:0 0 24px; }}
    h2 {{ margin-top:0; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; }}
    .card {{ background:#f8fafc; border:1px solid var(--line); border-radius:10px; padding:16px; }}
    .card strong {{ display:block; font-size:28px; }}
    .badge {{ display:inline-block; padding:7px 12px; border-radius:999px; font-weight:750; }}
    .ok {{ background:#dcfce7; color:#166534; }}
    .critical {{ background:#fee2e2; color:#991b1b; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th {{ background:#1e293b; color:#f8fafc; text-align:left; }}
    th,td {{ padding:9px 10px; border:1px solid #dbe3ed; vertical-align:top; }}
    tr:nth-child(even) td {{ background:#f1f5f9; }}
    figure {{ margin:22px 0; }}
    img {{ display:block; max-width:100%; margin:auto; border-radius:8px; }}
    .note {{ color:var(--muted); font-size:14px; }}
    code {{ background:#e8edf4; padding:2px 5px; border-radius:4px; }}
  </style>
</head>
<body>
<header>
  <h1>Auditoria de BreastDCEDL</h1>
  <p>Integridad, estructura, separacion por paciente y exploracion descriptiva.</p>
</header>
<main>
  <section>
    <h2>Resumen ejecutivo</h2>
    <p><span class="badge {status_class}">AUDITORIA {status}</span></p>
    <div class="cards">
      <div class="card"><span>Pacientes</span><strong>{summary['patients']:,}</strong></div>
      <div class="card"><span>Cortes</span><strong>{summary['samples']:,}</strong></div>
      <div class="card"><span>Imagenes esperadas</span><strong>{summary['expected_images']:,}</strong></div>
      <div class="card"><span>Criticos</span><strong>{report['critical_count']}</strong></div>
      <div class="card"><span>Advertencias</span><strong>{report['warning_count']}</strong></div>
    </div>
    <p class="note">Generado: {html.escape(report['generated_at'])}. La auditoria no modifica los datos.</p>
  </section>
  <section>
    <h2>Hallazgos</h2>
    {_html_table(issue_rows)}
  </section>
  <section>
    <h2>Integridad de imagenes</h2>
    <div class="cards">
      <div class="card"><span>Modo</span><strong>{html.escape(str(image_stats.get('mode')))}</strong></div>
      <div class="card"><span>Imagenes abiertas</span><strong>{image_stats.get('images_checked', 0):,}</strong></div>
      <div class="card"><span>Errores</span><strong>{image_stats.get('errors', 0):,}</strong></div>
      <div class="card"><span>Rango observado</span><strong>{image_stats.get('pixel_min')}–{image_stats.get('pixel_max')}</strong></div>
    </div>
    <p>Estadisticas de fase sobre la media de la imagen completa:</p>
    <pre>{html.escape(json.dumps(phase, indent=2, ensure_ascii=False))}</pre>
    <p class="note">Estas medias incluyen fondo y no equivalen a mediciones dentro de una mascara tumoral.</p>
  </section>
  <section>
    <h2>Distribuciones y galeria</h2>
    {figures_html}
  </section>
  <section>
    <h2>Perfilado de columnas</h2>
    {_html_table(profile_rows)}
  </section>
  <section>
    <h2>Relaciones verificadas</h2>
    <pre>patients.pid (1) ──────── (N) samples.patient_id
                              │
                              ├── tres rutas por corte: PRE, EARLY, LATE
                              └── pCR constante dentro de cada paciente</pre>
  </section>
  <section>
    <h2>Limites</h2>
    <p>Este analisis es educativo. No valida utilidad clinica, causalidad, equidad ni generalizacion externa.</p>
  </section>
</main>
</body>
</html>
"""
    path = output / "informe_auditoria.html"
    path.write_text(document, encoding="utf-8")
    return path


def run_audit(
    root: Path,
    output: Path,
    image_check: str = "all",
    sample_size: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    """Ejecuta la auditoria completa y escribe todos los artefactos."""

    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = AuditState()

    samples, patients = load_metadata(root)
    validate_metadata(samples, patients, state)
    analyze_distributions(samples, patients, state)
    path_audit = validate_paths(root, samples, state)
    image_audit = audit_images(
        root, samples, state, image_check, sample_size=sample_size, seed=seed
    )
    summary = build_summary(samples, patients)
    catalog = build_source_catalog(root, output)
    figures = create_figures(root, samples, patients, output, seed)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "status": "pass" if state.critical_count == 0 else "fail",
        "critical_count": state.critical_count,
        "warning_count": state.warning_count,
        "issues": [asdict(issue) for issue in state.issues],
        "summary": summary,
        "path_audit": path_audit,
        "image_audit": image_audit,
        "profiles": {
            "samples.csv": profile_dataframe(samples),
            "patients.csv": profile_dataframe(patients),
        },
        "source_catalog": catalog,
        "figures": figures,
    }
    report = _json_safe(report)
    (output / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    render_html(report, output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("breastdcedl"),
        help="Raiz que contiene metadata/ y dataset/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/audit"),
        help="Directorio de salida del informe.",
    )
    parser.add_argument(
        "--image-check",
        choices=("none", "sample", "all"),
        default="all",
        help="Cantidad de contenido de imagen que se abre y valida.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Cortes a comprobar cuando --image-check=sample.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_audit(
        root=args.root,
        output=args.output,
        image_check=args.image_check,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print()
    print(f"Estado: {report['status'].upper()}")
    print(f"Criticos: {report['critical_count']}")
    print(f"Advertencias: {report['warning_count']}")
    print(f"JSON: {args.output / 'audit_report.json'}")
    print(f"HTML: {args.output / 'informe_auditoria.html'}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
