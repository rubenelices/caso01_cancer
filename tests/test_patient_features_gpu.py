from pathlib import Path

import pytest

from src.train_patient_features import load_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('prefix', ['E56_nonlinear_cut_control', 'E57_patient_feature_mean'])
def test_gpu_preserves_scientific_protocol_and_separates_outputs(prefix):
    mac, a = load_spec(ROOT / 'configs/patient_features' / (prefix + '_mac.json'))
    gpu, b = load_spec(ROOT / 'configs/patient_features/gpu' / (prefix + '_cuda.json'))
    assert a.data == b.data and a.model == b.model and a.training == b.training
    assert mac['pool_stage'] == gpu['pool_stage']
    assert a.output_dir != b.output_dir and a.checkpoint_dir != b.checkpoint_dir
    assert b.output_dir.startswith('reports/gpu/')
    assert not b.training.mixed_precision
