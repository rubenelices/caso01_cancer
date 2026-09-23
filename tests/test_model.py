"""Pruebas de formas, parametros, forward y backward de la CNN minima."""

from __future__ import annotations

import torch
from torch import nn

from src.model import MinimalCNN, trainable_parameter_count


def test_minimal_cnn_output_and_shape_trace() -> None:
    model = MinimalCNN()
    batch = torch.zeros(2, 3, 256, 256)
    logits = model(batch)
    assert logits.shape == (2,)
    assert [step.shape for step in model.trace_shapes(batch_size=2)] == [
        (2, 3, 256, 256),
        (2, 8, 256, 256),
        (2, 8, 256, 256),
        (2, 8, 128, 128),
        (2, 16, 128, 128),
        (2, 16, 128, 128),
        (2, 16, 64, 64),
        (2, 32, 64, 64),
        (2, 32, 64, 64),
        (2, 32, 32, 32),
        (2, 32, 4, 4),
        (2, 512),
        (2,),
    ]


def test_minimal_cnn_parameter_count_is_explainable() -> None:
    model = MinimalCNN()
    assert model.conv1.weight.numel() + model.conv1.bias.numel() == 224
    assert model.conv2.weight.numel() + model.conv2.bias.numel() == 1_168
    assert model.conv3.weight.numel() + model.conv3.bias.numel() == 4_640
    assert model.classifier.weight.numel() + model.classifier.bias.numel() == 513
    assert trainable_parameter_count(model) == 6_545


def test_backward_produces_finite_gradients_and_optimizer_changes_weights() -> None:
    torch.manual_seed(42)
    model = MinimalCNN()
    images = torch.rand(4, 3, 256, 256)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.conv1.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(images), targets)
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    optimizer.step()
    assert not torch.equal(before, model.conv1.weight.detach())
