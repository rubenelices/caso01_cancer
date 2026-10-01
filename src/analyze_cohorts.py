"""Diagnóstico por cohorte de predicciones ya guardadas; no lee imágenes ni test.

Las cohortes solo anotan el informe, no son una nueva entrada de la CNN.
No ajusta modelos, umbrales, calibración ni agregación.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, roc_curve

from src.data import load_samples, split_by_patient_fold
from src.experiment_config import experiment_config_from_dict
from src.metrics import aggregate_by_patient, binary_metrics

NOTICE = "Diagnóstico exploratorio por paciente. Uso educativo, no clínico. Test no evaluado."


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validation(directory: Path, data_root: Path | None = None):
    """Comprueba trazabilidad contra el fold completo antes de analizar grupos."""
    raw = json.loads((directory / "config_resolved.json").read_text())
    config = experiment_config_from_dict(raw)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config.experiment_id):
        raise ValueError("Identificador de experimento inválido para guardar archivos")
    summary = json.loads((directory / "summary.json").read_text())
    evaluation = json.loads((directory / "evaluation_summary.json").read_text())
    if (summary.get("status") != "complete" or summary.get("scientific_result") is not True
            or summary.get("test_evaluated") is not False
            or summary.get("experiment_id") != config.experiment_id
            or config.data.train_patients_per_class is not None
            or config.data.validation_patients_per_class is not None):
        raise ValueError("Se requiere una ejecución científica completa, sin test ni debug")
    if (evaluation.get("scope") != "internal_validation_only"
            or evaluation.get("aggregation") != "mean"
            or evaluation.get("threshold") != config.training.threshold):
        raise ValueError("Se exige validación interna, agregación mean y umbral de la configuración")
    root = data_root or Path(config.data.root)
    samples = load_samples(root)
    expected = split_by_patient_fold(samples, config.data.validation_fold).validation
    cuts = pd.read_csv(directory / "predictions_cut.csv", dtype={"patient_id": str, "sample_id": str})
    patients = pd.read_csv(directory / "predictions_patient.csv", dtype={"patient_id": str})
    required = {"patient_id", "sample_id", "split", "fold", "target", "probability"}
    if not required.issubset(cuts.columns) or cuts.empty:
        raise ValueError("Faltan columnas o predicciones por corte")
    if (set(cuts.split) != {"train"} or set(cuts.fold) != {config.data.validation_fold}
            or cuts.sample_id.duplicated().any()
            or set(cuts.sample_id) != set(expected.sample_id)):
        raise ValueError("Las predicciones no corresponden exactamente al fold interno completo")
    aligned = cuts.merge(expected[["sample_id", "patient_id", "pCR"]], on="sample_id",
                         validate="one_to_one", suffixes=("", "_metadata"))
    if (not aligned.patient_id.equals(aligned.patient_id_metadata)
            or not np.array_equal(aligned.target, aligned.pCR)):
        raise ValueError("Identidades o etiquetas por corte inconsistentes")
    probabilities = cuts.probability.to_numpy(dtype=float)
    if (not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Probabilidades inválidas")
    required_patients = {"patient_id", "split", "fold", "target", "probability", "cuts"}
    if (not required_patients.issubset(patients.columns) or patients.patient_id.duplicated().any()
            or set(patients.split) != {"validation"}
            or set(patients.fold) != {config.data.validation_fold}):
        raise ValueError("Predicciones por paciente inválidas o ajenas a validación")
    reconstructed = aggregate_by_patient(cuts.patient_id, cuts.target, cuts.probability)
    patients = patients.sort_values("patient_id").reset_index(drop=True)
    reconstructed = reconstructed.sort_values("patient_id").reset_index(drop=True)
    if (patients.patient_id.tolist() != reconstructed.patient_id.tolist()
            or not np.array_equal(patients.target, reconstructed.target)
            or not np.array_equal(patients.cuts, reconstructed.cuts)
            or not np.allclose(patients.probability, reconstructed.probability, atol=1e-7, rtol=0)):
        raise ValueError("La agregación por paciente no coincide con los cortes")
    metadata = pd.read_csv(root / "metadata/patients.csv", dtype={"pid": str},
                           usecols=["pid", "dataset", "pCR", "split"])
    metadata = metadata[metadata.pid.isin(patients.patient_id)].copy()
    if (metadata.pid.duplicated().any() or set(metadata.pid) != set(patients.patient_id)
            or set(metadata.split) != {"train"}
            or metadata.dataset.isna().any() or metadata.dataset.astype(str).str.strip().eq("").any()):
        raise ValueError("Metadatos de cohorte ausentes, duplicados o ajenos a train")
    annotated = patients.merge(metadata[["pid", "dataset", "pCR"]], left_on="patient_id",
                               right_on="pid", validate="one_to_one")
    if not np.array_equal(annotated.target, annotated.pCR):
        raise ValueError("Etiquetas de patients.csv inconsistentes")
    annotated = annotated.drop(columns=["pid", "pCR"]).rename(columns={"dataset": "cohort"})
    return annotated, {
        "experiment_id": config.experiment_id, "best_epoch": summary["best_epoch"],
        "validation_fold": config.data.validation_fold, "seed": config.training.seed,
        "threshold": config.training.threshold, "aggregation": "mean", "config": raw,
        "environment": summary.get("environment", {}),
        "sources": {name: {"path": str(directory / name), "sha256": sha256(directory / name)}
                    for name in ("config_resolved.json", "summary.json", "evaluation_summary.json",
                                 "predictions_cut.csv", "predictions_patient.csv")},
        "metadata_sources": {name: {"path": str(root / "metadata" / name),
                                     "sha256": sha256(root / "metadata" / name)}
                             for name in ("samples.csv", "patients.csv")}}


def group_statistics(frame, threshold, repetitions, seed):
    targets = frame.target.to_numpy(dtype=int)
    probabilities = frame.probability.to_numpy(dtype=float)
    metrics = binary_metrics(targets, probabilities, threshold)
    metrics["brier_score"] = float(brier_score_loss(targets, probabilities))
    negative, positive = np.flatnonzero(targets == 0), np.flatnonzero(targets == 1)
    intervals = {name: {"lower": None, "upper": None} for name in ("roc_auc", "pr_auc")}
    if len(negative) and len(positive):
        rng = np.random.default_rng(seed)
        scores = []
        for _ in range(repetitions):
            indices = np.r_[rng.choice(negative, len(negative)), rng.choice(positive, len(positive))]
            scores.append((roc_auc_score(targets[indices], probabilities[indices]),
                           average_precision_score(targets[indices], probabilities[indices])))
        for index, name in enumerate(intervals):
            lower, upper = np.quantile(np.asarray(scores)[:, index], [0.025, 0.975])
            intervals[name] = {"lower": float(lower), "upper": float(upper)}
    distributions = {}
    for label in (0, 1):
        values = probabilities[targets == label]
        distributions[str(label)] = {"n": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "std": float(values.std()) if len(values) else None}
    return clean_json({"n": len(frame), "negative": len(negative), "positive": len(positive),
        "prevalence": float(targets.mean()), "mean_probability": float(probabilities.mean()),
        "metrics": metrics, "intervals": intervals, "probabilities_by_class": distributions,
        "small_group_warning": min(len(negative), len(positive)) < 10,
        "auc_defined": bool(len(negative) and len(positive))})


def paired_auc(left, right, repetitions, seed):
    """Delta right-left; mismas pacientes remuestreadas dentro de cohorte/clase."""
    columns = ["patient_id", "target", "fold", "cohort"]
    if not left[columns].equals(right[columns]):
        raise ValueError("Comparación exige las mismas pacientes, etiquetas, folds y cohortes")
    targets = left.target.to_numpy(dtype=int)
    if np.unique(targets).size < 2:
        return {"difference": None, "lower": None, "upper": None}
    first, second = left.probability.to_numpy(), right.probability.to_numpy()
    strata = [np.asarray(indices) for indices in left.groupby(["cohort", "target"]).indices.values()]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(group, len(group)) for group in strata])
        values.append(roc_auc_score(targets[indices], second[indices])
                      - roc_auc_score(targets[indices], first[indices]))
    lower, upper = np.quantile(values, [.025, .975])
    return {"difference": float(roc_auc_score(targets, second) - roc_auc_score(targets, first)),
            "lower": float(lower), "upper": float(upper)}


def save_figure(frames, results, output):
    figure, axes = plt.subplots(len(frames), 3, figsize=(17, 4.6 * len(frames)), squeeze=False)
    for row, (frame, result) in enumerate(zip(frames, results)):
        cohorts = sorted(frame.cohort.unique())
        for cohort in cohorts:
            subset = frame[frame.cohort == cohort]
            stats = result["groups"][cohort]
            if stats["auc_defined"]:
                fpr, tpr, _ = roc_curve(subset.target, subset.probability)
                axes[row, 0].plot(fpr, tpr, label=f"{cohort}: {stats['metrics']['roc_auc']:.3f} (n={len(subset)})")
        axes[row, 0].plot([0, 1], [0, 1], "--", color="grey")
        axes[row, 0].set(xlabel="1 − especificidad", ylabel="Sensibilidad", title="ROC dentro de cada cohorte")
        if axes[row, 0].get_legend_handles_labels()[0]:
            axes[row, 0].legend(fontsize=9)
        for index, cohort in enumerate(cohorts):
            for label, offset, color in ((0, -.16, "#457b9d"), (1, .16, "#e09f3e")):
                values = frame.loc[(frame.cohort == cohort) & (frame.target == label), "probability"]
                if len(values):
                    box = axes[row, 1].boxplot([values], positions=[index + offset], widths=.25,
                                               patch_artist=True, showfliers=True)
                    box["boxes"][0].set_facecolor(color)
        axes[row, 1].axhline(result["threshold"], color="grey", linestyle="--")
        axes[row, 1].set(xticks=range(len(cohorts)), xticklabels=cohorts, ylim=(0, 1),
                         ylabel="Probabilidad de pCR", title="Azul: pCR=0 · naranja: pCR=1")
        indices = np.arange(len(cohorts))
        axes[row, 2].bar(indices - .18, [result["groups"][name]["prevalence"] for name in cohorts],
                         width=.36, label="Prevalencia observada")
        axes[row, 2].bar(indices + .18, [result["groups"][name]["mean_probability"] for name in cohorts],
                         width=.36, label="Probabilidad media")
        labels = [f"{name}\nn={result['groups'][name]['n']} (+{result['groups'][name]['positive']})" for name in cohorts]
        axes[row, 2].set(xticks=indices, xticklabels=labels, ylim=(0, 1), title="Medias por cohorte (no causalidad)")
        axes[row, 2].legend(fontsize=9)
        axes[row, 0].text(0, 1.18, result["experiment_id"], transform=axes[row, 0].transAxes, fontsize=13)
        for axis in axes[row]:
            axis.grid(alpha=.15)
    figure.suptitle("Validación interna por paciente · Exploratorio · Uso educativo, no clínico", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, .97))
    figure.savefig(output / "cohort_dashboard.png", dpi=140, bbox_inches="tight")
    plt.close(figure)


def analyze_cohorts(directories, output: Path, *, data_root=None, repetitions=500, seed=42):
    if repetitions < 1:
        raise ValueError("repetitions debe ser positivo")
    if output.exists():
        raise FileExistsError(f"El informe ya existe; elige otra carpeta: {output}")
    if not directories:
        raise ValueError("Se requiere al menos un experimento")
    loaded = [load_validation(Path(directory), data_root) for directory in directories]
    frames, sources = zip(*loaded)
    if len({source["experiment_id"] for source in sources}) != len(sources):
        raise ValueError("Experimentos repetidos")
    for frame, source in loaded[1:]:
        if (source["validation_fold"], source["threshold"], source["aggregation"], source["seed"]) != (
                sources[0]["validation_fold"], sources[0]["threshold"], sources[0]["aggregation"], sources[0]["seed"]):
            raise ValueError("Comparación exige el mismo fold, umbral, agregación y semilla")
        paired_auc(frames[0], frame, 1, seed)  # Verifica alineación antes de guardar.
    results = []
    for frame, source in loaded:
        groups = {"ALL": group_statistics(frame, source["threshold"], repetitions, seed)}
        for cohort, subset in frame.groupby("cohort"):
            groups[cohort] = group_statistics(subset, source["threshold"], repetitions, seed)
        results.append({**source, "groups": groups})
    comparisons = []
    for index in range(1, len(frames)):
        groups = {"ALL": paired_auc(frames[0], frames[index], repetitions, seed)}
        for cohort in sorted(frames[0].cohort.unique()):
            left = frames[0][frames[0].cohort == cohort].reset_index(drop=True)
            right = frames[index][frames[index].cohort == cohort].reset_index(drop=True)
            groups[cohort] = paired_auc(left, right, repetitions, seed)
        comparisons.append({"reference": sources[0]["experiment_id"],
                            "candidate": sources[index]["experiment_id"], "auc_delta": groups})
    report = {"scope": "internal_validation_only", "test_evaluated": False, "notice": NOTICE,
        "bootstrap": {"method": "patient_stratified_percentile", "repetitions": repetitions,
                      "seed": seed, "confidence_level": .95,
                      "paired_strata": "cohort_and_target"},
        "warning": "Exploratorio: intervalos condicionados a composición, no corrigen selección de checkpoint ni múltiples ensayos. No prueba sesgo causal.",
        "experiments": results, "paired_comparisons": comparisons}
    output.mkdir(parents=True)
    rows = []
    for frame, result in zip(frames, results):
        frame.to_csv(output / f"{result['experiment_id']}_patients.csv", index=False)
        for cohort, stats in result["groups"].items():
            rows.append({"experiment_id": result["experiment_id"], "cohort": cohort,
                "n": stats["n"], "positive": stats["positive"], "negative": stats["negative"],
                "prevalence": stats["prevalence"], "small_group_warning": stats["small_group_warning"],
                **{key: value for key, value in stats["metrics"].items() if key != "confusion_matrix"}})
    pd.DataFrame(rows).to_csv(output / "cohort_metrics.csv", index=False)
    (output / "cohort_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    save_figure(frames, results, output)
    def show(value):
        return "No definido" if value is None else f"{value:.3f}"
    lines = ["# Diagnóstico por cohorte", "", NOTICE, "", "![Gráficos](cohort_dashboard.png)", "",
             "| Experimento | Cohorte | n (pCR=1) | AUC [IC 95 %] | AP | Sensibilidad | Especificidad |",
             "|---|---|---:|---|---:|---:|---:|"]
    for result in results:
        for cohort, stats in result["groups"].items():
            interval = stats["intervals"]["roc_auc"]
            metrics = stats["metrics"]
            lines.append(f"| {result['experiment_id']} | {cohort} | {stats['n']} ({stats['positive']}) | "
                f"{show(metrics['roc_auc'])} [{show(interval['lower'])}, {show(interval['upper'])}] | "
                f"{show(metrics['pr_auc'])} | {show(metrics['sensitivity'])} | {show(metrics['specificity'])} |")
    lines += ["", "## Diferencias pareadas de AUC: candidato − referencia", ""]
    for comparison in comparisons:
        for cohort, delta in comparison["auc_delta"].items():
            lines.append(f"- {comparison['candidate']} − {comparison['reference']} / {cohort}: "
                         f"{show(delta['difference'])} [{show(delta['lower'])}, {show(delta['upper'])}]")
    lines += ["", report["warning"], "",
        "Los grupos con menos de diez pacientes en alguna clase llevan aviso en JSON/CSV. Una sola clase deja AUC/AP no definidos.",
        "La PR-AUC/AP depende de prevalencia: no comparar cohortes como si su dificultad fuera idéntica.",
        "La ROC global compara también pacientes de cohortes distintas: diferencias respecto a ROC internas son pistas, no demostraciones de sesgo.",
        "Las probabilidades se agregan por media y mantienen el umbral guardado. No se optimiza por cohorte ni se cambia la CNN.",
        "Los CSV contienen identificadores y etiquetas: permanecen locales e ignorados por Git.",
        "Consultar config/entorno de cada ejecución; dispositivos o precisiones diferentes limitan comparaciones. Confirmar con más semillas/folds."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = analyze_cohorts(args.experiment_dir, args.output_dir, data_root=args.data_root,
                            repetitions=args.repetitions, seed=args.seed)
    print(json.dumps({"report_directory": str(args.output_dir), "test_evaluated": False,
        "experiments": [{"experiment_id": item["experiment_id"], "groups": item["groups"]}
                        for item in report["experiments"]],
        "paired_comparisons": report["paired_comparisons"]}, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
