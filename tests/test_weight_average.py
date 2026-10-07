"""Componente EMA aislado, no demuestra calidad ni integración científica."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from src.weight_average import EMAConfig, ExponentialModelAverage


def test_two_updates_exact_arithmetic_and_integer_counter_copy():
    current = nn.BatchNorm1d(2).double()
    average = ExponentialModelAverage(current,decay=0.5)
    with torch.no_grad():
        current.weight.fill_(3); current.running_mean.fill_(4)
        current.running_var.fill_(5); current.num_batches_tracked.fill_(7)
    average.update(current)
    assert torch.equal(average.model.weight,torch.tensor([2.,2.],dtype=torch.float64))
    assert torch.equal(average.model.running_mean,torch.tensor([2.,2.],dtype=torch.float64))
    assert torch.equal(average.model.running_var,torch.tensor([3.,3.],dtype=torch.float64))
    assert average.model.num_batches_tracked.item()==7
    with torch.no_grad():
        current.weight.fill_(5); current.num_batches_tracked.fill_(8)
    average.update(current)
    assert torch.equal(average.model.weight,torch.tensor([3.5,3.5],dtype=torch.float64))
    assert average.updates==2 and average.model.num_batches_tracked.item()==8


def test_copy_uses_no_rng_or_gradients_and_does_not_change_source_mode_or_weights():
    current = nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4),nn.Dropout(0.3))
    current.train()
    before = deepcopy(current.state_dict())
    rng = torch.get_rng_state()
    average = ExponentialModelAverage(current)
    average.update(current)
    assert torch.equal(rng,torch.get_rng_state()) and current.training
    assert not average.model.training and all(not p.requires_grad for p in average.model.parameters())
    assert all(torch.equal(value,current.state_dict()[name]) for name,value in before.items())
    assert all(p.grad is None for p in current.parameters())
    assert average.state_dict()['updates']==1 and average.state_dict()['decay']==0.99


def test_zero_decay_is_exact_current_copy():
    current = nn.Linear(3,2)
    average = ExponentialModelAverage(current,0)
    with torch.no_grad():
        current.weight.fill_(5); current.bias.fill_(-2)
    average.update(current)
    assert all(torch.equal(value,average.model.state_dict()[name]) for name,value in current.state_dict().items())


@pytest.mark.parametrize('decay',[-0.01,1,True,float('nan'),float('inf'),None,'0.99'])
def test_invalid_decay_rejected(decay):
    with pytest.raises(ValueError):
        EMAConfig(decay=decay)


def test_enabled_requires_real_boolean():
    with pytest.raises(ValueError):
        EMAConfig(enabled=1)


def test_nonfinite_update_is_rejected_before_any_copy_changes():
    current = nn.Linear(3,2)
    average = ExponentialModelAverage(current)
    before = deepcopy(average.model.state_dict())
    with torch.no_grad():
        current.weight.fill_(5); current.bias.fill_(float('nan'))
    with pytest.raises(ValueError,match='no finito'):
        average.update(current)
    assert average.updates==0
    assert all(torch.equal(value,average.model.state_dict()[name]) for name,value in before.items())


@pytest.mark.parametrize('bad',[nn.Linear(4,2),nn.Linear(3,2).double(),nn.Sequential(nn.Linear(3,2))])
def test_mismatched_structure_or_dtype_rejected_before_change(bad):
    average = ExponentialModelAverage(nn.Linear(3,2))
    with pytest.raises(ValueError):
        average.update(bad)
    assert average.updates==0


def test_half_complex_and_empty_models_rejected():
    for current in (nn.Linear(3,2).half(),nn.Identity()):
        with pytest.raises(ValueError):
            ExponentialModelAverage(current)
    with pytest.warns(UserWarning,match='Complex modules'):
        complex_model=nn.Linear(3,2).to(torch.complex64)
    with pytest.raises(ValueError):
        ExponentialModelAverage(complex_model)


def test_real_e19_cnn_update_tracks_every_weight_and_bn_buffer_without_rng():
    # Modelo real, entrada sintética pequeña para control algebraico, no calidad.
    from pathlib import Path
    from src.architectures import BreastPCRNet,trainable_parameter_count
    from src.experiment_config import load_experiment_config
    root=Path(__file__).resolve().parents[1]
    model=BreastPCRNet(load_experiment_config(root/'configs/experiments/E19_weight_decay_001_mac.json').model)
    average=ExponentialModelAverage(model,0.99)
    before=deepcopy(average.model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    loss=nn.BCEWithLogitsLoss()(model(torch.rand(2,3,32,32)),torch.tensor([0.,1.]))
    loss.backward(); optimizer.step()
    current=deepcopy(model.state_dict())
    rng=torch.get_rng_state()
    average.update(model)
    assert torch.equal(rng,torch.get_rng_state()) and model.training
    assert trainable_parameter_count(model)==294129
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in current.items())
    for name,value in average.model.state_dict().items():
        expected=torch.lerp(before[name],current[name],0.01) if value.is_floating_point() else current[name]
        assert torch.equal(value,expected)
    assert average.updates==1
    assert all(not p.requires_grad and p.grad is None for p in average.model.parameters())
