"""Objetivos binarios suavizados solo para regularizar la pérdida de train.

Integrado únicamente en train; desactivado por defecto para modelos anteriores.
No modifica etiquetas originales, imágenes, predicciones ni validación/test.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class LabelSmoothingConfig:
    enabled: bool = False
    epsilon: float = .1

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError('label_smoothing.enabled debe ser booleano')
        if (type(self.epsilon) not in (int,float) or not math.isfinite(self.epsilon)
                or not 0 <= self.epsilon < 1):
            raise ValueError('label_smoothing.epsilon debe estar en [0,1)')


def smooth_binary_targets(target: Tensor, config: LabelSmoothingConfig) -> Tensor:
    """y_s=(1-epsilon)*y+epsilon/2; epsilon.1 da0.05/0.95.

    Default/epsilon0 son identidad sin consumir RNG ni copiar tensores. La
    política activa exige etiquetas binarias flotantes, conserva dtype/device
    y no cambia target in-place. No muestreo, pseudoetiquetado ni Mixup.
    """
    if not config.enabled or config.epsilon == 0:
        return target
    if not target.is_floating_point() or target.ndim != 1 or not target.numel():
        raise ValueError('Label smoothing exige vector flotante no vacío')
    if not torch.isfinite(target).all() or not ((target==0)|(target==1)).all():
        raise ValueError('Label smoothing exige etiquetas binarias finitas')
    return target*(1-config.epsilon)+config.epsilon/2
