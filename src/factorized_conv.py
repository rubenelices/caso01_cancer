"""Operador propio depthwise + pointwise, opcional en nuestra CNN.

No es una arquitectura importada. No incluye BN/ReLU: ConvBlock las añade
después de la mezcla de canales, no entre ambas operaciones.
"""
from torch import Tensor, nn


class FactorizedSpatialConv(nn.Module):
    """Filtro espacial por canal seguido de mezcla aprendida 1×1.

    Mismo H×W con stride1 y kernel impar. Sin bias espacial; el bias opcional de
    salida pertenece solo a pointwise. Capacidad menor que una conv densa general.
    El constructor utiliza el reset estándar de Conv2d; BreastPCRNet aplica
    la política de inicialización prefijada al construir la arquitectura.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        settings = (in_channels, out_channels, kernel_size, dilation)
        if any(type(value) is not int or value < 1 for value in settings):
            raise ValueError("Canales, kernel y dilatación deben ser enteros positivos")
        if kernel_size % 2 == 0:
            raise ValueError("El kernel debe ser impar")
        if type(bias) is not bool:
            raise ValueError("bias debe ser booleano")
        # No mezcla canales: cada fase/mapa tiene su filtro espacial propio.
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        # Mezcla esos mapas, sin añadir otra expansión del campo receptivo.
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=bias
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(image))
