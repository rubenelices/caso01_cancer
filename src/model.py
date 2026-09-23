"""CNN minima construida desde cero para validar el pipeline completo."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ShapeStep:
    """Forma de un tensor tras una operacion de la red."""

    operation: str
    shape: tuple[int, ...]


class MinimalCNN(nn.Module):
    """CNN educativa pequena, sin transfer learning ni bloques importados.

    Recibe ``[B, 3, 256, 256]`` y devuelve un logit por muestra con forma
    ``[B]``. No aplica sigmoide: durante entrenamiento se utiliza
    ``BCEWithLogitsLoss`` y durante inferencia ``torch.sigmoid``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.spatial_summary = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(32 * 4 * 4, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Inicializa las capas entrenables de forma explicita y reproducible."""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _forward_features(
        self,
        image: Tensor,
        trace: list[ShapeStep] | None = None,
    ) -> Tensor:
        def record(operation: str, value: Tensor) -> Tensor:
            if trace is not None:
                trace.append(ShapeStep(operation, tuple(value.shape)))
            return value

        image = record("entrada", image)
        image = record("conv1 3→8", self.conv1(image))
        image = record("relu1", self.relu1(image))
        image = record("pool1", self.pool1(image))
        image = record("conv2 8→16", self.conv2(image))
        image = record("relu2", self.relu2(image))
        image = record("pool2", self.pool2(image))
        image = record("conv3 16→32", self.conv3(image))
        image = record("relu3", self.relu3(image))
        image = record("pool3", self.pool3(image))
        image = record("adaptive_avg_pool 4×4", self.spatial_summary(image))
        return image

    def forward(self, image: Tensor) -> Tensor:
        features = self._forward_features(image)
        flattened = self.flatten(features)
        logits = self.classifier(flattened)
        return logits.squeeze(1)

    @torch.no_grad()
    def trace_shapes(self, batch_size: int = 2) -> list[ShapeStep]:
        """Traza dimensiones con una entrada ficticia sin modificar pesos."""

        device = next(self.parameters()).device
        trace: list[ShapeStep] = []
        dummy = torch.zeros(batch_size, 3, 256, 256, device=device)
        features = self._forward_features(dummy, trace)
        flattened = self.flatten(features)
        trace.append(ShapeStep("flatten", tuple(flattened.shape)))
        logits = self.classifier(flattened).squeeze(1)
        trace.append(ShapeStep("linear 512→1", tuple(logits.shape)))
        return trace


def trainable_parameter_count(model: nn.Module) -> int:
    """Numero de escalares que backpropagation puede modificar."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_breakdown(model: nn.Module) -> list[dict[str, int | str]]:
    """Desglose de parametros por tensor entrenable."""

    return [
        {
            "name": name,
            "shape": str(tuple(parameter.shape)),
            "parameters": parameter.numel(),
        }
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
