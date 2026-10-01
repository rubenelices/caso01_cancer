"""Ablaciones controladas y compatibilidad del dropout de mapas aprendidos."""
from dataclasses import replace
from io import BytesIO

import pytest
import torch
from torch import nn

from src.architectures import BaseCNNConfig, BreastPCRNet, trainable_parameter_count
from src.document_architectures import draw, trace
from src.experiment_config import experiment_config_from_dict, load_experiment_config
from src.smoke_training import validate_smoke_config


@pytest.mark.parametrize("name,p", [("E14_spatial_dropout_010", 0.1),
                                     ("E15_spatial_dropout_020", 0.2)])
def test_only_spatial_dropout_changes_against_long_reference(name, p):
    reference = load_experiment_config("configs/experiments/E13_50epochs.json")
    candidate = load_experiment_config(f"configs/experiments/{name}.json")
    assert candidate.model == replace(reference.model, spatial_dropout=p)
    assert candidate.training == reference.training
    assert candidate.data == reference.data
    assert candidate.experiment_id != reference.experiment_id
    assert candidate.output_dir != reference.output_dir
    assert candidate.checkpoint_dir != reference.checkpoint_dir
    assert candidate.training.early_stopping_patience > candidate.training.epochs
    smoke = load_experiment_config(
        f"configs/experiments/{name.replace(name[:3], 'SMOKE', 1)}.json"
    )
    validate_smoke_config(smoke)
    assert smoke.model == candidate.model
    assert smoke.output_dir != candidate.output_dir
    assert smoke.checkpoint_dir != candidate.checkpoint_dir


@pytest.mark.parametrize("p", [-0.01, 1.0, 1.1, float("nan"), float("inf")])
def test_invalid_spatial_dropout_is_rejected(p):
    with pytest.raises(ValueError, match="spatial_dropout"):
        BaseCNNConfig(spatial_dropout=p)


def test_spatial_dropout_masks_whole_maps_not_pixels_and_eval_is_identity():
    torch.manual_seed(42)
    layer = nn.Dropout2d(0.2)
    image = torch.ones(16, 128, 4, 4)
    output = layer(image)
    maps = output.flatten(2)
    assert torch.equal(maps, maps[:, :, :1].expand_as(maps))
    assert (maps == 0).any() and (maps != 0).any()
    assert set(output.unique().tolist()) == {0.0, 1.25}
    layer.eval()
    assert torch.equal(layer(image), image)


@pytest.mark.parametrize("pool_position", ["between_convolutions", "after_convolutions"])
def test_dropout_runs_at_block_output_and_preserves_gradients(pool_position):
    model = BreastPCRNet(BaseCNNConfig(pool_position=pool_position, spatial_dropout=0.1))
    events = []
    block = model.blocks[0]
    handles = [
        block.features[-1].register_forward_hook(lambda *_: events.append("last_relu")),
        block.pool.register_forward_hook(lambda *_: events.append("pool")),
        block.spatial_dropout.register_forward_hook(lambda *_: events.append("dropout")),
    ]
    try:
        logits = model(torch.rand(2, 3, 256, 256))
        nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0])).backward()
    finally:
        for handle in handles:
            handle.remove()
    assert events == (["pool", "last_relu", "dropout"] if pool_position == "between_convolutions"
                      else ["last_relu", "pool", "dropout"])
    assert logits.shape == (2,)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("p", [0.1, 0.2])
def test_actual_diagram_trace_shapes_parameters_and_receptive_field(p):
    model = BreastPCRNet(BaseCNNConfig(pool_position="between_convolutions",
                                      input_representation="pre_differences", spatial_dropout=p))
    steps = trace(model)
    masks = [step for step in steps if step["kind"] == "regularization"]
    assert [step["shape"] for step in masks] == [(16, 128, 128), (32, 64, 64),
                                                (64, 32, 32), (128, 16, 16)]
    assert all(step["parameters"] == 0 for step in masks)
    assert masks[-1]["rf"] == 106
    assert len([s for s in steps if s["kind"] == "conv"]) == 8
    assert len([s for s in steps if s["kind"] == "pool"]) == 4
    assert trainable_parameter_count(model) == 294129


def test_legacy_config_state_keys_initialization_and_eval_are_preserved():
    legacy = load_experiment_config("configs/experiments/E13_phase_differences.json")
    assert legacy.model.spatial_dropout == 0
    torch.manual_seed(42)
    reference = BreastPCRNet(legacy.model).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = BreastPCRNet(replace(legacy.model, spatial_dropout=0.1)).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    assert all(torch.equal(value, candidate.state_dict()[key])
               for key, value in reference.state_dict().items())
    image = torch.rand(2, 3, 256, 256)
    with torch.inference_mode():
        assert torch.equal(reference(image), candidate(image))
    assert all(isinstance(block.spatial_dropout, nn.Identity) for block in reference.blocks)


def test_serialized_checkpoint_restores_regularization_and_eval_predictions():
    config = load_experiment_config("configs/experiments/E14_spatial_dropout_010.json")
    model = BreastPCRNet(config.model).eval()
    buffer = BytesIO()
    torch.save({"config": config.to_dict(), "model_state_dict": model.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    restored_config = experiment_config_from_dict(checkpoint["config"])
    restored = BreastPCRNet(restored_config.model).eval()
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert restored_config == config
    assert all(block.spatial_dropout.p == 0.1 for block in restored.blocks)
    image = torch.rand(2, 3, 256, 256)
    with torch.inference_mode():
        assert torch.equal(restored(image), model(image))


def test_diagram_regeneration_does_not_change_content(tmp_path):
    model = BreastPCRNet(BaseCNNConfig(spatial_dropout=0.1))
    steps = trace(model)
    path = tmp_path / "architecture"
    draw(steps, "Prueba reproducible", "Sin datos de pacientes", 294129, path)
    original = {suffix: path.with_suffix(suffix).read_bytes() for suffix in (".svg", ".png")}
    draw(steps, "Prueba reproducible", "Sin datos de pacientes", 294129, path)
    assert all(path.with_suffix(suffix).read_bytes() == content
               for suffix, content in original.items())
