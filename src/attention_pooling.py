"""Pooling espacial con compuerta, construido con capas básicas desde cero.

Inspiración matemática: Ilse et al. (ICML 2018), no su arquitectura ni pesos.
Las instancias aquí son posiciones del mapa de UN corte, no otras pacientes.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class GatedSpatialPool(nn.Module):
    """[B,C,H,W] -> [B,C,1,1] mediante una media espacial aprendida.

    Dos ramas paralelas 1x1 C->16: tanh(Vh) y sigmoid(Uh). Su producto
    alimenta una tercera 1x1 16->1; softmax sobre H*W, nunca sobre batch/C.
    Score sin bias (softmax elimina cualquier desplazamiento común).
    C=128: 2*(128*16+16)+16=4.144 parámetros. Kernel1, padding0, stride1:
    no reducción ni expansión del campo receptivo local de cada posición.
    La suma ponderada final sí usa el mapa completo.
    """

    def __init__(self, channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels debe ser un entero positivo")
        if type(hidden_channels) is not int or hidden_channels < 1:
            raise ValueError("hidden_channels debe ser un entero positivo")
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.value = nn.Conv2d(channels, hidden_channels, 1, bias=True)
        self.gate = nn.Conv2d(channels, hidden_channels, 1, bias=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.score = nn.Conv2d(hidden_channels, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.value.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.value.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        # Arranque uniforme: equivale a GAP. Primero aprende score; luego las ramas.
        nn.init.zeros_(self.score.weight)

    def attention_weights(self, features: Tensor) -> Tensor:
        if (features.ndim != 4 or features.shape[1] != self.channels
                or features.shape[-2] < 1 or features.shape[-1] < 1):
            raise ValueError("La atención requiere un mapa [B,C,H,W] no vacío")
        hidden = self.tanh(self.value(features)) * self.sigmoid(self.gate(features))
        scores = self.score(hidden)
        # Softmax/reducción estables en float32 con AMP; float64 para gradcheck.
        stable_scores = scores if scores.dtype == torch.float64 else scores.float()
        return self.softmax(stable_scores.flatten(2)).reshape_as(stable_scores)

    def forward(self, features: Tensor) -> Tensor:
        weights = self.attention_weights(features)
        stable_features = features if features.dtype == torch.float64 else features.float()
        pooled = (stable_features * weights).sum(dim=(-2, -1), keepdim=True)
        return pooled.to(features.dtype)
