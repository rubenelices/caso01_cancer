from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from src.architectures import BaseCNNConfig, trainable_parameter_count
from src.patient_feature_model import PatientFeatureNet, group_features, load_feature_checkpoint
from src.train_patient_features import load_spec, evaluate_features, train_epoch, central_cut_predictions, run

ROOT = Path(__file__).resolve().parents[1]


def config():
    return BaseCNNConfig(input_representation='pre_differences', pool_position='between_convolutions')


def test_matched_pair_only_changes_pool_stage_and_identity():
    a, ca = load_spec(ROOT / 'configs/patient_features/E56_nonlinear_cut_control_mac.json')
    b, cb = load_spec(ROOT / 'configs/patient_features/E57_patient_feature_mean_mac.json')
    assert ca.model == cb.model and ca.data == cb.data and ca.training == cb.training
    assert a['pool_stage'] == 'probabilities' and b['pool_stage'] == 'features'
    assert ca.output_dir != cb.output_dir and ca.checkpoint_dir != cb.checkpoint_dir


def test_shapes_parameters_gradients_and_one_cut_equivalence():
    model = PatientFeatureNet(config()).eval()
    assert trainable_parameter_count(model) == 298161
    x = torch.rand(3, 3, 32, 32)
    y = torch.tensor([0., 0., 1.])
    z, labels, ids = model.forward_patients(x, y, ['a', 'a', 'b'])
    assert z.shape == labels.shape == (2,) and ids == ['a', 'b']
    assert model.extract_features(x).shape == (3, 128)
    single, _, _ = model.forward_patients(x[:1], y[:1], ['a'])
    torch.testing.assert_close(single, model(x[:1]))
    z.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_permutation_variable_sizes_and_gradient_to_all_cuts():
    f = torch.randn(5, 128, requires_grad=True)
    y = torch.tensor([0., 1., 0., 1., 0.])
    ids = ['a', 'b', 'a', 'b', 'a']
    m, labels, names = group_features(f, y, ids)
    torch.testing.assert_close(m[0], f[[0, 2, 4]].mean(0))
    torch.testing.assert_close(m[1], f[[1, 3]].mean(0))
    m.sum().backward()
    assert (f.grad.abs().sum(1) > 0).all()
    perm = [4, 3, 2, 1, 0]
    m2, _, _ = group_features(f.detach()[perm], y[perm], [ids[i] for i in perm])
    torch.testing.assert_close(m, m2)


@pytest.mark.parametrize('bad', ['labels', 'ids', 'nan', 'shape'])
def test_invalid_groups(bad):
    f, y, ids = torch.ones(2, 128), torch.zeros(2), ['a', 'a']
    if bad == 'labels': y[1] = 1
    if bad == 'ids': ids[0] = ''
    if bad == 'nan': f[0, 0] = float('nan')
    if bad == 'shape': f = f.flatten()
    with pytest.raises(ValueError): group_features(f, y, ids)


def test_linear_mean_identity_but_nonlinear_mean_not_equivalent():
    f = torch.tensor([[0., 2.], [2., 0.]])
    linear = nn.Linear(2, 1)
    torch.testing.assert_close(linear(f.mean(0)), linear(f).mean(0))
    nonlinear = nn.Sequential(nn.Linear(2, 1), nn.ReLU())
    with torch.no_grad():
        nonlinear[0].weight.copy_(torch.tensor([[1., -1.]]))
        nonlinear[0].bias.zero_()
    assert not torch.allclose(nonlinear(f.mean(0)), nonlinear(f).mean(0))


def batches(size):
    torch.manual_seed(14)
    x = torch.rand(4, 3, 32, 32)
    rows = []
    for start in range(0, 4, size):
        idx = list(range(start, min(start + size, 4)))
        rows.append(dict(image=x[idx], target=torch.tensor([0., 0., 1., 1.])[idx],
            patient_id=[['a', 'a', 'b', 'b'][i] for i in idx], sample_id=[str(i) for i in idx],
            slice_index=torch.tensor(idx)))
    return rows


def test_evaluation_independent_of_batch_patient_boundaries():
    model = PatientFeatureNet(config()).eval()
    a = evaluate_features(model, batches(1), torch.device('cpu'), .5)
    b = evaluate_features(model, batches(4), torch.device('cpu'), .5)
    pd.testing.assert_frame_equal(a[5], b[5], check_exact=False, atol=1e-6, rtol=1e-5)
    assert a[2]['patient']['n'] == a[3]['n'] == 2


@pytest.mark.parametrize('stage', ['features', 'probabilities'])
def test_train_step_both_stages(stage):
    model = PatientFeatureNet(config())
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    before = model.head[0].weight.detach().clone()
    loss = train_epoch(model, batches(4), opt, torch.device('cpu'), stage)
    assert 0 < loss < 100 and not torch.equal(before, model.head[0].weight)


def test_checkpoint_restore_and_old_checkpoint_rejected(tmp_path):
    spec, cfg = load_spec(ROOT / 'configs/patient_features/E57_patient_feature_mean_mac.json')
    model = PatientFeatureNet(cfg.model).eval()
    path = tmp_path / 'feature.pt'
    torch.save(dict(model_class='PatientFeatureNet_v1', config=spec, model_state_dict=model.state_dict()), path)
    restored = load_feature_checkpoint(path)
    x = torch.rand(1, 3, 32, 32)
    torch.testing.assert_close(restored(x), model(x))
    torch.save({'model_state_dict': model.state_dict()}, path)
    with pytest.raises(ValueError): load_feature_checkpoint(path)


def test_central_selection_does_not_use_labels_or_probabilities():
    frame = pd.DataFrame(dict(patient_id=['b','a','a','b','a'], sample_id=['b2','a0','a1','b0','a2'],
        slice_index=[2,0,1,0,2], target=[0,1,1,0,1], probability=[.9,.1,.2,.3,.4]))
    selected = central_cut_predictions(frame)
    assert selected.sample_id.tolist() == ['a1','b2']
    frame['probability'] = 1 - frame.probability
    assert central_cut_predictions(frame).sample_id.tolist() == ['a1','b2']


def test_same_seed_pair_has_identical_initial_weights():
    torch.manual_seed(42)
    a = PatientFeatureNet(config())
    torch.manual_seed(42)
    b = PatientFeatureNet(config())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


def test_runner_preserves_existing_even_partial_folder(tmp_path, monkeypatch):
    import json
    spec, cfg = load_spec(ROOT / 'configs/patient_features/E57_patient_feature_mean_mac.json')
    # Relative path inside reports; reject before trying data load.
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / 'reports/smoke/existing_partial'
    folder.mkdir(parents=True)
    spec['experiment']['output_dir'] = 'reports/smoke/existing_partial'
    path = tmp_path / 'blocked.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(FileExistsError): run(path, 'cpu')
