"""La candidata de cinco bloques cambia profundidad, no el protocolo de E05."""

from dataclasses import replace

import torch
from torch import nn

from src.architectures import BreastPCRNet, trainable_parameter_count
from src.document_architectures import trace
from src.experiment_config import load_experiment_config


CONFIG = "configs/experiments/E06_five_blocks_normal.json"


def test_deep_config_preserves_e05_training_and_data() -> None:
    reference = load_experiment_config("configs/experiments/E05_pool_between_convs.json")
    candidate = load_experiment_config(CONFIG)
    assert candidate.data == reference.data
    assert candidate.training == reference.training
    assert candidate.model == replace(reference.model, channels=(16, 32, 64, 128, 128))
    assert candidate.output_dir != reference.output_dir
    assert candidate.checkpoint_dir != reference.checkpoint_dir


def test_deep_shapes_real_order_and_receptive_field() -> None:
    model = BreastPCRNet(load_experiment_config(CONFIG).model)
    steps = trace(model)
    convs = [step for step in steps if step["kind"] == "conv"]
    pools = [step for step in steps if step["kind"] == "pool"]
    assert len(convs) == 10
    assert len(pools) == 5
    assert trainable_parameter_count(model) == 589_553
    assert convs[-1]["shape"] == (128, 8, 8)
    assert convs[-1]["rf"] == 218
    assert steps[-1]["shape"] == (1,)
    assert [step["kind"] for step in steps[1:16]] == ["conv", "pool", "conv"] * 5


def test_deep_backward_and_weight_roundtrip(tmp_path) -> None:
    torch.manual_seed(42)
    config = load_experiment_config(CONFIG).model
    model = BreastPCRNet(config)
    images = torch.rand(2, 3, 256, 256)
    loss = nn.BCEWithLogitsLoss()(model(images), torch.tensor([0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path / "deep_state.pt"
    torch.save(model.state_dict(), path)
    restored = BreastPCRNet(config)
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    restored.eval()
    with torch.inference_mode():
        assert torch.equal(model(images), restored(images))


def test_deep_smoke_uses_real_architecture_and_small_patient_subset() -> None:
    real = load_experiment_config(CONFIG)
    smoke = load_experiment_config("configs/experiments/SMOKE_five_blocks_end_to_end.json")
    assert smoke.model == real.model
    assert smoke.data.train_patients_per_class == 4
    assert smoke.data.validation_patients_per_class == 2
    assert smoke.training.epochs == 2
    assert smoke.output_dir != real.output_dir
    assert smoke.checkpoint_dir != real.checkpoint_dir
