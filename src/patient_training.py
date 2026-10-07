"""Lotes completos y pérdida por paciente, sin cambiar la CNN por corte.

La probabilidad agregada es exactamente la media usada por el evaluador.
Se obtiene su logit de forma estable y se conserva BCEWithLogitsLoss.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Sampler


def _patient_groups(rows: pd.DataFrame) -> list[list[int]]:
    if rows.empty or not {"patient_id", "pCR"}.issubset(rows.columns):
        raise ValueError("Se necesitan filas y etiquetas por paciente")
    if rows.patient_id.isna().any() or rows.pCR.isna().any():
        raise ValueError("Paciente o etiqueta ausente")
    if not rows.patient_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Identificador de paciente inválido")
    if not rows.pCR.isin([0, 1]).all():
        raise ValueError("Las etiquetas deben ser binarias")
    if rows.groupby("patient_id").pCR.nunique().ne(1).any():
        raise ValueError("Una paciente tiene etiquetas contradictorias")
    # Posiciones, no etiquetas del índice: coincide con el Dataset reseteado.
    return [list(indices) for indices in rows.reset_index(drop=True)
            .groupby("patient_id", sort=True).indices.values()]


class PatientBatchSampler(Sampler[list[int]]):
    """Cada paciente aparece una vez por época, con todos sus cortes juntos.

    Se mezclan pacientes, no se mezclan clases deliberadamente ni se descartan
    pacientes del último lote. El número de cortes de un lote puede variar.
    """
    def __init__(self, rows: pd.DataFrame, patients_per_batch: int,
                 generator: torch.Generator) -> None:
        if type(patients_per_batch) is not int or patients_per_batch < 1:
            raise ValueError("patients_per_batch debe ser un entero positivo")
        self.groups = _patient_groups(rows)
        self.patients_per_batch = patients_per_batch
        self.generator = generator

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(len(self.groups), generator=self.generator).tolist()
        for start in range(0, len(order), self.patients_per_batch):
            yield [index for patient in order[start:start + self.patients_per_batch]
                   for index in self.groups[patient]]

    def __len__(self) -> int:
        return math.ceil(len(self.groups) / self.patients_per_batch)


def patient_level_pos_weight(rows: pd.DataFrame) -> float:
    _patient_groups(rows)
    targets = rows.drop_duplicates("patient_id").pCR
    n0, n1 = int(targets.eq(0).sum()), int(targets.eq(1).sum())
    if not n0 or not n1:
        raise ValueError("Se necesitan ambas clases para calcular pos_weight")
    return n0 / n1


def pool_patient_logits(logits: Tensor, targets: Tensor,
                        patient_ids: Sequence[str]) -> tuple[Tensor, Tensor]:
    """logit(mean(sigmoid(z))) sin redondear probabilidades ni hacer clamp.

    log(q/(1-q)) = logsumexp(logsigmoid(z))
                    - logsumexp(logsigmoid(-z)).
    Los factores 1/n se cancelan. No es mean(z). Gradiente a todos los cortes.
    No exige diez cortes ni presupone que el orden de pacientes sea contiguo.
    El llamador debe proporcionar todos los cortes: el sampler lo garantiza.
    """
    if (logits.ndim != 1 or targets.shape != logits.shape or not logits.numel()
            or len(patient_ids) != logits.numel()):
        raise ValueError("Logits, etiquetas e identificadores incompatibles")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("Logits no finitos o no flotantes")
    if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("Etiquetas no binarias o no finitas")
    if targets.device != logits.device:
        raise ValueError("Logits y etiquetas deben estar en el mismo dispositivo")
    indices: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_ids):
        if not isinstance(patient, str) or not patient.strip():
            raise ValueError("Identificador de paciente inválido")
        indices.setdefault(patient, []).append(index)
    working = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    pooled, labels = [], []
    for positions in indices.values():
        index = torch.tensor(positions, dtype=torch.long, device=logits.device)
        group_targets = targets.index_select(0, index)
        if not torch.all(group_targets == group_targets[0]):
            raise ValueError("Una paciente tiene etiquetas contradictorias")
        values = working.index_select(0, index)
        pooled.append(torch.logsumexp(F.logsigmoid(values), dim=0)
                      - torch.logsumexp(F.logsigmoid(-values), dim=0))
        labels.append(group_targets[0])
    return torch.stack(pooled), torch.stack(labels)


def objective_loss(logits: Tensor, targets: Tensor, patient_ids: Sequence[str],
                   criterion: nn.Module, objective: str) -> tuple[Tensor, int]:
    """Devuelve pérdida media y número de unidades para registrar la época."""
    if objective == "cut":
        return criterion(logits, targets), logits.numel()
    if objective != "patient_mean_probability":
        raise ValueError("Objetivo de entrenamiento no reconocido")
    pooled, labels = pool_patient_logits(logits, targets, patient_ids)
    return criterion(pooled, labels), pooled.numel()
