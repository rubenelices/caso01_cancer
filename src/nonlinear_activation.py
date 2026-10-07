"""Activaciones del extractor; ReLU anterior por defecto, SiLU para E52."""
from torch import nn


def make_activation(name: str = 'relu') -> nn.Module:
    """Default ReLU exacto; SiLU=x*sigmoid(x), sin parámetros aprendibles.

    No importa una CNN ni cambia kernels, dimensiones o intensidades de entrada.
    Ambas fuera de lugar para no modificar tensores compartidos por ramas.
    """
    if type(name) is not str or name not in {'relu','silu'}:
        raise ValueError('activation debe ser relu o silu')
    return nn.ReLU(inplace=False) if name=='relu' else nn.SiLU(inplace=False)
