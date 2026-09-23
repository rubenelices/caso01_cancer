"""Arquitecturas CNN propias del proyecto, sin modelos preentrenados."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from src.model import ShapeStep


@dataclass(frozen=True)
class BaseCNNConfig:
    """Decisiones estructurales de la primera arquitectura candidata."""

    channels: tuple[int, ...] = (16, 32, 64, 128)
    convolutions_per_block: int = 2
    kernel_size: int = 3
    use_batch_norm: bool = True
    dropout: float = 0.30

    def __post_init__(self) -> None:
        if not self.channels or any(channel < 1 for channel in self.channels):
            raise ValueError("channels debe contener enteros positivos")
        if self.convolutions_per_block < 1:
            raise ValueError("convolutions_per_block debe ser positivo")
        if self.kernel_size % 2 == 0 or self.kernel_size < 1:
            raise ValueError("kernel_size debe ser impar y positivo")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout debe estar en [0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ConvBlock(nn.Module):
    """Convoluciones que conservan HxW seguidas de pooling 2x2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convolutions: int,
        kernel_size: int,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(convolutions):
            layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    bias=not use_batch_norm,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=False))
            current_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, image: Tensor) -> Tensor:
        return self.pool(self.features(image))


class BreastPCRNet(nn.Module):
    """Primera CNN candidata para predecir pCR desde PRE/EARLY/LATE.

    Recibe ``[B, 3, 256, 256]`` y devuelve logits ``[B]``. La arquitectura se
    construye exclusivamente con capas basicas de PyTorch y pesos inicializados
    desde cero.
    """

    def __init__(self, config: BaseCNNConfig = BaseCNNConfig()) -> None:
        super().__init__()
        self.config = config
        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in config.channels:
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    convolutions=config.convolutions_per_block,
                    kernel_size=config.kernel_size,
                    use_batch_norm=config.use_batch_norm,
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.channels[-1], 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image: Tensor) -> Tensor:
        for block in self.blocks:
            image = block(image)
        image = self.global_pool(image)
        image = self.flatten(image)
        image = self.dropout(image)
        return self.classifier(image).squeeze(1)

    @torch.no_grad()
    def trace_shapes(self, batch_size: int = 2) -> list[ShapeStep]:
        device = next(self.parameters()).device
        image = torch.zeros(batch_size, 3, 256, 256, device=device)
        trace = [ShapeStep("entrada", tuple(image.shape))]
        for index, block in enumerate(self.blocks, start=1):
            image = block(image)
            trace.append(ShapeStep(f"bloque {index}", tuple(image.shape)))
        image = self.global_pool(image)
        trace.append(ShapeStep("global average pooling", tuple(image.shape)))
        image = self.flatten(image)
        trace.append(ShapeStep("flatten", tuple(image.shape)))
        logits = self.classifier(image).squeeze(1)
        trace.append(ShapeStep("clasificador", tuple(logits.shape)))
        return trace


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_breakdown(model: nn.Module) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "parameters": parameter.numel(),
        }
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
