"""EMA propia de entrenamiento y evaluación; sin pesos externos.

Promedia parámetros y buffers flotantes; copia contadores enteros. No consume
azar, no produce gradientes y nunca modifica el modelo del optimizador.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class EMAConfig:
    enabled: bool = False
    decay: float = 0.99

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("EMA enabled debe ser booleano")
        if (type(self.decay) not in {int, float} or not math.isfinite(self.decay)
                or not 0 <= self.decay < 1):
            raise ValueError("EMA decay debe ser finito en [0,1)")


class ExponentialModelAverage:
    """Copia evaluable de UNA trayectoria; no un modelo pretrained/teacher.

    Construir después de mover el modelo al dispositivo. Solo float32/64,
    un dispositivo y arquitectura fija. No implementa resume ni reinyección.
    """

    def __init__(self, model: nn.Module, decay: float = 0.99) -> None:
        self.config = EMAConfig(True, decay)
        tensors = model.state_dict()
        if not tensors or len({value.device for value in tensors.values()}) != 1:
            raise ValueError("EMA requiere estado no vacío en un único dispositivo")
        if any(value.is_complex() or (value.is_floating_point() and value.dtype
                not in {torch.float32, torch.float64}) for value in tensors.values()):
            raise ValueError("EMA requiere pesos/buffers reales float32 o float64")
        self._check_finite(tensors)
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.updates = 0

    @staticmethod
    def _check_finite(tensors: dict) -> None:
        checks = [torch.isfinite(value).all() for value in tensors.values()
                  if value.is_floating_point()]
        # Una comprobación conjunta evita sincronizar por cada peso en MPS.
        if checks and not torch.stack(checks).all().item():
            raise ValueError("EMA rechaza estado no finito antes de modificar la copia")

    @torch.no_grad()
    def update(self, current: nn.Module) -> None:
        source = current.state_dict()
        target = self.model.state_dict()
        if source.keys() != target.keys():
            raise ValueError("EMA requiere las mismas claves de arquitectura")
        for name, value in source.items():
            destination = target[name]
            if (value.shape != destination.shape or value.dtype != destination.dtype
                    or value.device != destination.device):
                raise ValueError("EMA requiere formas, tipos y dispositivo idénticos")
        self._check_finite(source)
        for name, value in source.items():
            destination = target[name]
            if value.is_floating_point() and self.config.decay != 0:
                destination.lerp_(value, 1 - self.config.decay)
            else:
                destination.copy_(value)
        self.updates += 1

    def state_dict(self) -> dict:
        """Estado para guardado inmediato, no una copia histórica inmutable."""
        return {"decay": self.config.decay, "updates": self.updates,
                "model_state_dict": self.model.state_dict()}
