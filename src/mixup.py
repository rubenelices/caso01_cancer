"""Mixup por lote, solo para entrenamiento por corte con BCE.

Las tres fases y la etiqueta comparten pareja y coeficiente. No genera
pacientes independientes ni debe aplicarse en validación o inferencia.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class MixupConfig:
    enabled: bool = False
    alpha: float = 0.2

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("mixup.enabled debe ser booleano")
        if (type(self.alpha) not in (int, float)
                or not math.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError("mixup.alpha debe ser finito y positivo")


def mix_batch(images: Tensor, targets: Tensor, coefficient: float,
              permutation: Tensor) -> tuple[Tensor, Tensor]:
    """Combinación convexa explícita; no modifica los tensores originales."""
    if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] < 1:
        raise ValueError("Mixup requiere imágenes [B,3,H,W], B>=1")
    if targets.shape != (images.shape[0],):
        raise ValueError("Mixup requiere una etiqueta por corte [B]")
    if (not images.is_floating_point() or not targets.is_floating_point()
            or images.device != targets.device):
        raise ValueError("Imágenes y etiquetas deben ser flotantes en el mismo dispositivo")
    if (type(coefficient) not in (int, float) or not math.isfinite(coefficient)
            or not 0 <= coefficient <= 1):
        raise ValueError("El coeficiente debe estar en [0,1]")
    if permutation.device.type != "cpu" or permutation.dtype != torch.long:
        raise ValueError("La permutación debe ser int64 en CPU")
    if (permutation.shape != targets.shape
            or not torch.equal(permutation.sort().values, torch.arange(images.shape[0]))):
        raise ValueError("Índices deben formar una permutación del lote")
    # Las etiquetas originales proceden del cargador binario validado.
    indices = permutation.to(images.device)
    return (coefficient * images + (1 - coefficient) * images[indices],
            coefficient * targets + (1 - coefficient) * targets[indices])


def apply_mixup(images: Tensor, targets: Tensor,
                config: MixupConfig | None = None) -> tuple[Tensor, Tensor]:
    """Beta(alpha,alpha) y randperm, reproducibles con las semillas del runner.

    Desactivado (y B=1) es identidad y no consume azar: conserva las ejecuciones
    antiguas. La permutación puede emparejar un corte consigo mismo o con otro
    de su paciente; no presupone independencia ni parejas de distinta clase.
    """
    if config is None or not config.enabled or images.shape[0] == 1:
        return images, targets
    coefficient = float(np.random.beta(config.alpha, config.alpha))
    permutation = torch.randperm(images.shape[0], device="cpu")
    return mix_batch(images, targets, coefficient, permutation)
