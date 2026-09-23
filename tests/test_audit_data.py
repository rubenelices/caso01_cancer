"""Pruebas de las reglas estructurales de la auditoria."""

from __future__ import annotations

import pandas as pd

from src.audit_data import AuditState, validate_metadata


def minimal_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.DataFrame(
        {
            "sample_id": ["P1_z001", "P2_z001"],
            "patient_id": ["P1", "P2"],
            "split": ["train", "test"],
            "path_pre": ["dataset/train/P1/P1_z001_PRE.png", "dataset/test/P2/P2_z001_PRE.png"],
            "path_early": ["dataset/train/P1/P1_z001_EARLY.png", "dataset/test/P2/P2_z001_EARLY.png"],
            "path_late": ["dataset/train/P1/P1_z001_LATE.png", "dataset/test/P2/P2_z001_LATE.png"],
            "slice_index": [1, 1],
            "pCR": [0, 1],
            "fold": [0, -1],
        }
    )
    patients = pd.DataFrame(
        {
            "pid": ["P1", "P2"],
            "pCR": [0, 1],
            "dataset": ["spy1", "duke"],
            "split": ["train", "test"],
        }
    )
    return samples, patients


def issue_codes(state: AuditState) -> set[str]:
    return {issue.code for issue in state.issues}


def test_valid_metadata_has_no_critical_findings() -> None:
    samples, patients = minimal_frames()
    state = validate_metadata(samples, patients)
    assert state.critical_count == 0
    assert "metadata_integrity_ok" in issue_codes(state)


def test_duplicate_sample_id_is_critical() -> None:
    samples, patients = minimal_frames()
    samples.loc[1, "sample_id"] = samples.loc[0, "sample_id"]
    state = validate_metadata(samples, patients)
    assert "duplicate_sample_id" in issue_codes(state)
    assert state.critical_count >= 1


def test_patient_split_leakage_is_critical() -> None:
    samples, patients = minimal_frames()
    leaked = samples.iloc[[0]].copy()
    leaked["sample_id"] = "P1_z002"
    leaked["split"] = "test"
    leaked["fold"] = -1
    samples = pd.concat([samples, leaked], ignore_index=True)
    state = validate_metadata(samples, patients)
    assert "patient_split_leakage" in issue_codes(state)


def test_inconsistent_label_within_patient_is_critical() -> None:
    samples, patients = minimal_frames()
    inconsistent = samples.iloc[[0]].copy()
    inconsistent["sample_id"] = "P1_z002"
    inconsistent["pCR"] = 1
    samples = pd.concat([samples, inconsistent], ignore_index=True)
    state = validate_metadata(samples, patients)
    assert "inconsistent_patient_label" in issue_codes(state)
