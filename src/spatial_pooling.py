"""Reducción espacial de features, sin pesos ni manipulación de las fases DCE."""
from torch import nn


def make_downsampling_pool(mode: str = "max") -> nn.Module:
    """Reduce H y W a la mitad, conservando canales; kernel2/stride2/padding0.

    max conserva la respuesta mayor de cada ventana; average su media.
    No modifica el cargador ni añade parámetros/RNG. El default reproduce la
    construcción anterior de ConvBlock exactamente. No importa una arquitectura.
    """
    if mode == "max":
        return nn.MaxPool2d(kernel_size=2, stride=2)
    if mode == "average":
        return nn.AvgPool2d(kernel_size=2, stride=2, padding=0, count_include_pad=False)
    raise ValueError("downsampling_pool debe ser max o average")
