"""Solo componente aislado: álgebra BCE/gradientes/default, no rendimiento."""
import pytest
import torch
from torch import nn
from src.label_smoothing import LabelSmoothingConfig,smooth_binary_targets


def test_formula_no_mutation_dtype_device_rng():
    y=torch.tensor([0.,1.,1.,0.],dtype=torch.float64);original=y.clone()
    state=torch.get_rng_state();actual=smooth_binary_targets(y,LabelSmoothingConfig(True,.1))
    assert torch.allclose(actual,torch.tensor([.05,.95,.95,.05],dtype=y.dtype))
    assert actual.dtype==y.dtype and actual.device==y.device
    assert torch.equal(state,torch.get_rng_state()) and torch.equal(y,original)

@pytest.mark.parametrize('cfg',[LabelSmoothingConfig(),LabelSmoothingConfig(True,0)])
def test_disabled_identity(cfg):
    y=torch.tensor([0.,1.]);state=torch.get_rng_state()
    assert smooth_binary_targets(y,cfg) is y
    assert torch.equal(state,torch.get_rng_state())

def test_bce_gradient_and_uniform_loss_mixture():
    logits=torch.tensor([-4.,-.2,.4,5.],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0.,1.,0.,1.],dtype=logits.dtype);epsilon=.1
    smooth=smooth_binary_targets(y,LabelSmoothingConfig(True,epsilon))
    criterion=nn.BCEWithLogitsLoss(reduction='none')
    actual=criterion(logits,smooth)
    expected=(1-epsilon)*criterion(logits,y)+epsilon*criterion(logits,torch.full_like(y,.5))
    assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
    actual.sum().backward()
    assert torch.allclose(logits.grad,torch.sigmoid(logits.detach())-smooth,atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('epsilon',[True,-.1,1.,float('nan'),float('inf'),'0.1'])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):LabelSmoothingConfig(True,epsilon)

def test_invalid_enabled():
    with pytest.raises(ValueError):LabelSmoothingConfig(1)

@pytest.mark.parametrize('target',[torch.tensor([0,1]),torch.tensor([[0.,1.]]),
    torch.tensor([]),torch.tensor([float('nan')]),torch.tensor([.3]),torch.tensor([2.])])
def test_invalid_active_targets(target):
    with pytest.raises(ValueError):smooth_binary_targets(target,LabelSmoothingConfig(True))
