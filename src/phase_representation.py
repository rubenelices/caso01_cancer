"""Representación fija de las fases DCE; no aprende pesos ni usa etiquetas."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PhaseRepresentation(nn.Module):
    """Recibe fases originales [B, 3, H, W], siempre PRE/EARLY/LATE.

    ``raw`` conserva la entrada. ``pre_differences`` devuelve PRE,
    EARLY-PRE y LATE-EARLY. No recorta negativos ni normaliza cada fase.
    ``temporal_differences`` devuelve solo EARLY-PRE y LATE-EARLY:
    dos canales internos, aunque la entrada externa sigue teniendo tres fases.
    Vive dentro del modelo para compartir exactamente train/eval/inferencia.
    No tiene parámetros ni buffers: checkpoints antiguos mantienen sus claves.
    """

    def __init__(self, mode: str = "raw") -> None:
        super().__init__()
        if mode not in {"raw", "pre_differences", "temporal_differences"}:
            raise ValueError("input_representation no reconocida")
        self.mode = mode

    @property
    def output_channels(self) -> int:
        """Canales que recibe la primera convolución, sin estado ni pesos."""
        return 2 if self.mode == "temporal_differences" else 3

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Las fases originales deben tener forma [B, 3, H, W]")
        if not image.is_floating_point():
            raise ValueError("Las restas requieren fases en coma flotante, no uint8")
        if self.mode == "raw":
            return image
        pre, early, late = image.unbind(dim=1)
        if self.mode == "temporal_differences":
            return torch.stack((early - pre, late - early), dim=1)
        return torch.stack((pre, early - pre, late - early), dim=1)
