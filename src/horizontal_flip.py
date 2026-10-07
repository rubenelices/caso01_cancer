"""Espejo espacial sincronizado para aumentos exclusivamente de train."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class HorizontalFlipConfig:
    enabled: bool = False
    probability: float = 0.5

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("horizontal_flip.enabled debe ser booleano")
        if (type(self.probability) not in {int, float}
                or not math.isfinite(self.probability)
                or not 0 <= self.probability <= 1):
            raise ValueError("horizontal_flip.probability debe ser finito en [0,1]")


def apply_shared_horizontal_flip(image: Tensor) -> Tensor:
    """Invierte solo W; no fases, intensidades, etiquetas ni archivos originales.

    Sin interpolación/cropping/relleno. Una única operación para PRE/EARLY/LATE,
    con soporte de valores negativos para comprobar diferencias firmadas.
    """
    if image.shape != (3,256,256) or image.dtype != torch.float32:
        raise ValueError("El espejo exige tensor float32 [3,256,256]")
    if not torch.isfinite(image).all():
        raise ValueError("El espejo exige valores finitos")
    return torch.flip(image,dims=(-1,)).contiguous()


class SharedRandomHorizontalFlip:
    """Una moneda por tripleta; default desactivado no consume azar."""
    def __init__(self, config: HorizontalFlipConfig) -> None:
        self.config = config

    def sample_flip(self, *, generator: torch.Generator | None = None) -> bool:
        if not self.config.enabled or self.config.probability == 0:
            return False
        if self.config.probability == 1:
            return True
        return torch.rand((),generator=generator).item() < self.config.probability

    def __call__(self, image: Tensor) -> Tensor:
        return apply_shared_horizontal_flip(image) if self.sample_flip() else image
