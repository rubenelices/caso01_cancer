"""Realce firmado, compatibilidad de pesos y comparación de una sola variable."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.architectures import BaseCNNConfig, BreastPCRNet, trainable_parameter_count
from src.document_architectures import trace
from src.experiment_config import experiment_config_from_dict, load_experiment_config
from src.phase_representation import PhaseRepresentation


ROOT = Path(__file__).resolve().parents[1]


def test_signed_differences_are_invertible_without_mutating_input():
    image = torch.tensor([.8, .9, .2]).view(1, 3, 1, 1)
    original = image.clone()
    output = PhaseRepresentation("pre_differences")(image)
    assert torch.allclose(output.flatten(), torch.tensor([.8, .1, -.7]))
    assert torch.equal(image, original)
    pre, rise, late_change = output.unbind(1)
    recovered = torch.stack((pre, pre + rise, pre + rise + late_change), 1)
    assert torch.allclose(recovered, image)
    assert output.dtype == image.dtype and output.device == image.device


def test_differences_preserve_negative_early_changes_and_range():
    image = torch.tensor([1., 0., 1.]).view(1, 3, 1, 1)
    assert torch.equal(PhaseRepresentation("pre_differences")(image).flatten(),
                       torch.tensor([1., -1., 1.]))


def test_gradient_reaches_all_original_phases():
    image = torch.rand(2, 3, 4, 4, requires_grad=True)
    output = PhaseRepresentation("pre_differences")(image)
    weights = torch.tensor([1., 2., 4.]).view(1, 3, 1, 1)
    (output * weights).sum().backward()
    expected = torch.tensor([-1., -2., 4.]).view(1, 3, 1, 1).expand_as(image)
    assert torch.equal(image.grad, expected)


def test_raw_is_identity_without_parameters_or_rng_changes():
    image = torch.rand(2, 3, 4, 4)
    state = torch.random.get_rng_state()
    transform = PhaseRepresentation()
    assert transform(image) is image
    assert list(transform.parameters()) == [] and transform.state_dict() == {}
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("image", [torch.zeros(3, 4, 4), torch.zeros(1, 5, 4, 4),
                                   torch.zeros(1, 3, 4, 4, dtype=torch.uint8)])
def test_reject_invalid_input_contract(image):
    with pytest.raises(ValueError):
        PhaseRepresentation("pre_differences")(image)


def test_reject_unknown_mode():
    with pytest.raises(ValueError, match="input_representation"):
        BaseCNNConfig(input_representation="unknown")
    with pytest.raises(ValueError, match="input_representation"):
        PhaseRepresentation("unknown")


def test_e13_changes_only_input_representation():
    base = load_experiment_config(ROOT / "configs/experiments/E05_mac_reference.json")
    candidate = load_experiment_config(ROOT / "configs/experiments/E13_phase_differences.json")
    assert candidate.data == base.data
    assert candidate.training == base.training
    assert candidate.model == replace(base.model, input_representation="pre_differences")
    assert candidate.output_dir != base.output_dir
    assert candidate.checkpoint_dir != base.checkpoint_dir
    assert candidate.experiment_id != base.experiment_id
    assert experiment_config_from_dict(candidate.to_dict()) == candidate
    assert base.model.input_representation == "raw"


def test_smoke_preserves_representation_and_is_small():
    smoke = load_experiment_config(ROOT / "configs/experiments/SMOKE_phase_differences.json")
    candidate = load_experiment_config(ROOT / "configs/experiments/E13_phase_differences.json")
    assert smoke.model == candidate.model
    assert smoke.training.learning_rate == candidate.training.learning_rate
    assert smoke.training.epochs == 2
    assert smoke.data.train_patients_per_class == 4
    assert smoke.data.validation_patients_per_class == 2
    assert smoke.output_dir != candidate.output_dir
    assert smoke.checkpoint_dir != candidate.checkpoint_dir


def test_same_initial_weights_keys_parameters_and_legacy_checkpoint():
    torch.manual_seed(42)
    original = BreastPCRNet()
    torch.manual_seed(42)
    candidate = BreastPCRNet(BaseCNNConfig(input_representation="pre_differences"))
    assert trainable_parameter_count(original) == trainable_parameter_count(candidate) == 294129
    assert original.state_dict().keys() == candidate.state_dict().keys()
    assert all(torch.equal(value, candidate.state_dict()[key])
               for key, value in original.state_dict().items())
    old_config = original.config.to_dict()
    old_config.pop("input_representation")
    legacy_model = BreastPCRNet(BaseCNNConfig(**old_config)).eval()
    legacy_model.load_state_dict(original.state_dict(), strict=True)
    original.eval()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(legacy_model(image), original(image))


def test_forward_uses_transform_once_before_first_conv_and_backpropagates():
    model = BreastPCRNet(BaseCNNConfig(channels=(4, 8), input_representation="pre_differences"))
    image = torch.rand(2, 3, 32, 32, requires_grad=True)
    received = []
    handle = model.blocks[0].features[0].register_forward_pre_hook(
        lambda layer, inputs: received.append(inputs[0].detach().clone()))
    logits = model(image)
    handle.remove()
    assert logits.shape == (2,)
    assert len(received) == 1
    assert torch.equal(received[0], PhaseRepresentation("pre_differences")(image))
    logits.sum().backward()
    assert torch.isfinite(image.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_diagram_traces_real_transform_without_changing_receptive_field():
    config = load_experiment_config(ROOT / "configs/experiments/E13_phase_differences.json")
    model = BreastPCRNet(config.model)
    steps = trace(model)
    assert steps[1]["label"] == "PRE\nEARLY−PRE\nLATE−EARLY"
    assert steps[1]["shape"] == (3, 256, 256)
    assert steps[1]["parameters"] == 0 and steps[1]["rf"] == 1
    assert len([s for s in steps if s["kind"] == "conv"]) == 8
    assert len([s for s in steps if s["kind"] == "pool"]) == 4
    assert [s for s in steps if s["kind"] == "conv"][-1]["rf"] == 106
    assert model.trace_shapes(1)[1].shape == (1, 3, 256, 256)
