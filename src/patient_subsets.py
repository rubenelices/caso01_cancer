"""Subconjuntos ANIDADOS de train por paciente para diagnosticar cantidad de datos.

Sin PNG, test, recortes, reetiquetado ni balance artificial 50/50. Mantiene
aproximadamente la proporción de cada estrato cohorte×pCR. Hash estable ordena
las pacientes sin consumir el azar del modelo/loader. Las fracciones menores
son prefijos del mismo orden; validación permanece idéntica.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd

from src.data import DatasetSplits, DataValidationError


def selection_hash(rows: pd.DataFrame) -> str:
    records = rows[['sample_id', 'patient_id', 'pCR']].sort_values('sample_id')
    return hashlib.sha256(records.to_csv(index=False).encode()).hexdigest()


def select_training_patients(splits: DatasetSplits, root: str | Path,
                             fraction: float = 1., seed: int = 42):
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Fracción fuera de (0,1]")
    if type(seed) is not int or seed < 0:
        raise ValueError("Semilla de subconjunto inválida")
    rows = splits.train
    manifest = {"fraction": fraction, "subset_seed": seed,
                "available_patients": int(rows.patient_id.nunique()),
                "available_cuts": len(rows), "strata": [],
                "validation_changed": False, "test_evaluated": False}
    if fraction == 1:
        # Compatibilidad: no carga metadata extra ni consume RNG con configs antiguas.
        manifest.update(method='full_train', selected_patients=manifest['available_patients'],
                        selected_cuts=len(rows), selection_sha256=selection_hash(rows))
        return splits, manifest
    if (rows.empty or rows.patient_id.isna().any() or rows.sample_id.duplicated().any()
            or not rows.pCR.isin([0, 1]).all()
            or set(rows.split) != {'train'}
            or rows.groupby('patient_id').pCR.nunique().gt(1).any()
            or rows.groupby('patient_id')['fold'].nunique().gt(1).any()
            or splits.validation_fold in set(rows['fold'])):
        raise DataValidationError("Train interno inválido para seleccionar pacientes")
    patients = rows[['patient_id', 'pCR']].drop_duplicates().copy()
    patients['patient_id'] = patients.patient_id.astype(str)
    metadata = pd.read_csv(Path(root) / 'metadata/patients.csv', dtype={'pid': str},
                           usecols=['pid', 'dataset', 'pCR', 'split'])
    metadata = metadata.loc[metadata.pid.isin(patients.patient_id)]
    if (metadata.pid.duplicated().any() or set(metadata.pid) != set(patients.patient_id)
            or metadata.dataset.isna().any() or metadata.dataset.str.strip().eq('').any()
            or set(metadata.split) != {'train'}):
        raise DataValidationError("Cohorte ausente, duplicada o ajena a train")
    patients = patients.merge(metadata, left_on='patient_id', right_on='pid',
                              validate='one_to_one', suffixes=('', '_metadata'))
    if not patients.pCR.equals(patients.pCR_metadata):
        raise DataValidationError("Etiquetas de paciente y cortes no coinciden")
    selected = []
    for (cohort, label), group in patients.groupby(['dataset', 'pCR'], sort=True):
        count = int(math.floor(len(group) * fraction + .5))
        if count < 1:
            raise DataValidationError("Fracción demasiado pequeña: algún estrato queda vacío")
        def order(pid):
            key = f'{seed}\0{cohort}\0{label}\0{pid}'.encode()
            return hashlib.sha256(key).hexdigest(), pid
        ids = sorted(group.patient_id.tolist(), key=order)[:count]
        selected.extend(ids)
        manifest['strata'].append({'cohort': str(cohort), 'pCR': int(label),
                                  'available': len(group), 'selected': count})
    subset = rows.loc[rows.patient_id.astype(str).isin(selected)].reset_index(drop=True)
    for frame in (splits.validation, splits.test):
        if set(subset.patient_id.astype(str)) & set(frame.patient_id.astype(str)):
            raise DataValidationError("Fuga de pacientes en la selección")
    manifest.update(method='nested_sha256_cohort_class_v1', selected_patients=len(selected),
                    selected_cuts=len(subset), selection_sha256=selection_hash(subset))
    return DatasetSplits(subset, splits.validation, splits.test, splits.validation_fold), manifest
