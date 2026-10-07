"""Aumentos de train compartidos por PRE/EARLY/LATE, nunca aumentos RGB."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from src.horizontal_flip import HorizontalFlipConfig, SharedRandomHorizontalFlip


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


@dataclass(frozen=True)
class IntensityAugmentationConfig:
    """Ganancia uniforme compartida: g ~ U(1-delta, 1+delta), sin clipping.

    Separada de la geometría para activar una única variable en E25.
    Los valores por defecto no cambian ningún experimento anterior.
    """

    enabled: bool = False
    max_gain_delta: float = 0.1

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("intensity_augmentation.enabled debe ser booleano")
        if (type(self.max_gain_delta) not in (int, float)
                or not math.isfinite(self.max_gain_delta)
                or not 0 <= self.max_gain_delta <= 0.25):
            raise ValueError("max_gain_delta debe ser finito entre 0 y 0.25")


def apply_shared_gain(image: Tensor, gain: float) -> Tensor:
    """Multiplica las tres fases por UN escalar positivo sin alterar la entrada.

    Entrada del cargador en [0,1]. Salida de entrenamiento hasta 1.25 (E25: 1.1).
    No recorta a 1: (g*EARLY - g*PRE) = g*(EARLY-PRE), también con negativos.
    No es normalización, simulación física de contraste ni cambio de etiquetas.
    """
    if image.shape != (3, 256, 256) or image.dtype != torch.float32:
        raise ValueError("La ganancia exige tensor float32 [3,256,256]")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("La ganancia exige fases finitas en [0,1] antes del aumento")
    if type(gain) not in (int, float) or not math.isfinite(gain) or not .75 <= gain <= 1.25:
        raise ValueError("gain debe ser un escalar finito entre 0.75 y 1.25")
    return image if gain == 1 else image * gain


class SharedRandomGain:
    """Un factor nuevo por tripleta y lectura; no uno por canal ni por píxel."""

    def __init__(self, config: IntensityAugmentationConfig) -> None:
        self.config = config

    def sample_gain(self, *, generator: torch.Generator | None = None) -> float:
        if not self.config.enabled or self.config.max_gain_delta == 0:
            return 1.0
        return 1.0 + (2 * torch.rand((), generator=generator).item() - 1) * self.config.max_gain_delta

    def __call__(self, image: Tensor) -> Tensor:
        if not self.config.enabled or self.config.max_gain_delta == 0:
            return image
        return apply_shared_gain(image, self.sample_gain())


class SharedTrainAugmentations:
    """Composición serializable: geometría primero, ganancia después.

    E25 solo usa ganancia. Se conserva esta composición para evitar dos pipelines
    distintos si un experimento futuro autoriza combinar ambos aumentos.
    """

    def __init__(self, affine: SharedRandomAffine, gain: SharedRandomGain) -> None:
        self.affine, self.gain = affine, gain

    def __call__(self, image: Tensor) -> Tensor:
        return self.gain(self.affine(image))


class SharedTrainHorizontalFlip:
    """Espejo primero, pipeline anterior después; solo se construye si activo."""
    def __init__(self,previous,flip:SharedRandomHorizontalFlip) -> None:
        self.previous,self.flip=previous,flip

    def __call__(self,image:Tensor) -> Tensor:
        return self.previous(self.flip(image))


def build_train_transform(
    config: AugmentationConfig,
    intensity_config: IntensityAugmentationConfig | None = None,
    horizontal_config: HorizontalFlipConfig | None = None,
) -> SharedRandomAffine | SharedRandomGain | SharedTrainAugmentations | SharedTrainHorizontalFlip | SharedRandomHorizontalFlip | None:
    """Desactivado por defecto; el camino antiguo conserva incluso el consumo RNG."""
    affine = SharedRandomAffine(config) if config.enabled else None
    if intensity_config is None or not intensity_config.enabled or intensity_config.max_gain_delta == 0:
        previous = affine
    else:
        gain = SharedRandomGain(intensity_config)
        previous = SharedTrainAugmentations(affine, gain) if affine is not None else gain
    if horizontal_config is None or not horizontal_config.enabled or horizontal_config.probability == 0:
        return previous
    flip=SharedRandomHorizontalFlip(horizontal_config)
    return SharedTrainHorizontalFlip(previous,flip) if previous is not None else flip
