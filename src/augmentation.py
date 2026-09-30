"""Aumentos geométricos de train sobre PRE/EARLY/LATE como un solo tensor."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = False
    max_rotation_degrees: float = 5.0
    max_translation_fraction: float = 0.03

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("augmentation.enabled debe ser booleano")
        if not math.isfinite(self.max_rotation_degrees) or not 0 <= self.max_rotation_degrees <= 180:
            raise ValueError("max_rotation_degrees debe estar entre 0 y 180")
        if not math.isfinite(self.max_translation_fraction) or not 0 <= self.max_translation_fraction <= 0.25:
            raise ValueError("max_translation_fraction debe estar entre 0 y 0.25")


@dataclass(frozen=True)
class AffineParameters:
    angle_degrees: float
    translation_x_pixels: float
    translation_y_pixels: float


def apply_shared_affine(image: Tensor, params: AffineParameters) -> Tensor:
    """Una sola rejilla bilineal para los tres canales, sin normalizar intensidades.

    Desplazamientos positivos mueven contenido hacia derecha/abajo. La rejilla
    mapea coordenadas de salida a entrada (transformación inversa). Fuera de
    la imagen se rellena con cero, igual en las tres fases. No modifica la entrada.
    """
    if image.shape != (3, 256, 256) or image.dtype != torch.float32:
        raise ValueError("El aumento exige tensor float32 [3,256,256]")
    if not torch.isfinite(image).all():
        raise ValueError("El aumento exige valores finitos")
    values = (params.angle_degrees, params.translation_x_pixels, params.translation_y_pixels)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Parametros afines no finitos")
    if all(value == 0 for value in values):
        return image
    angle = math.radians(params.angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    tx = 2 * params.translation_x_pixels / image.shape[2]
    ty = 2 * params.translation_y_pixels / image.shape[1]
    theta = image.new_tensor([
        [cosine, sine, -cosine * tx - sine * ty],
        [-sine, cosine, sine * tx - cosine * ty],
    ]).unsqueeze(0)
    batch = image.unsqueeze(0)
    grid = F.affine_grid(theta, batch.shape, align_corners=False)
    return F.grid_sample(batch, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False).squeeze(0).contiguous()


class SharedRandomAffine:
    """Parámetros nuevos por muestra; respeta torch.manual_seed y semillas workers."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def sample_parameters(self, *, generator: torch.Generator | None = None) -> AffineParameters:
        random_values = (torch.rand(3, generator=generator) * 2 - 1).tolist()
        pixels = 256 * self.config.max_translation_fraction
        return AffineParameters(random_values[0] * self.config.max_rotation_degrees,
                                random_values[1] * pixels, random_values[2] * pixels)

    def __call__(self, image: Tensor) -> Tensor:
        if not self.config.enabled:
            return image
        return apply_shared_affine(image, self.sample_parameters())


def build_train_transform(config: AugmentationConfig) -> SharedRandomAffine | None:
    """Desactivado por defecto para preservar todos los experimentos anteriores."""
    return SharedRandomAffine(config) if config.enabled else None
