"""Media generalizada propia con exponente fijo; cabeza opcional E54, no default."""
import math
import torch
from torch import Tensor,nn


class FixedPowerMeanPool(nn.Module):
    """[B,C,H,W]→[B,C,1,1], (mean(x**p))**(1/p), p fijo.

    Features no negativos tras ReLU, no fases/etiquetas/probabilidades. p3 da
    más peso a activaciones intensas sin usar solo un máximo. Epsilon evita
    raíz problemática en mapas cero. Escalado por máximo evita overflow;
    cálculo float32 bajo AMP, float64 conservado para comprobaciones.
    """
    def __init__(self,power:float=3.,epsilon:float=1e-6) -> None:
        super().__init__()
        if type(power) not in (int,float) or not math.isfinite(power) or not 1<=power<=8:
            raise ValueError('power debe estar en [1,8]')
        if type(epsilon) not in (int,float) or not math.isfinite(epsilon) or epsilon<=0:
            raise ValueError('epsilon debe ser finito positivo')
        self.power,self.epsilon=power,epsilon

    def forward(self,image:Tensor) -> Tensor:
        if image.ndim!=4 or not image.numel() or not image.is_floating_point():
            raise ValueError('GeM exige mapa flotante no vacío [B,C,H,W]')
        if not torch.isfinite(image).all() or (image<0).any():
            raise ValueError('GeM exige features finitos no negativos')
        work=image if image.dtype==torch.float64 else image.float()
        if self.power==1:
            return work.mean((-2,-1),keepdim=True).to(image.dtype)
        values=work.clamp_min(self.epsilon)
        scale=values.amax((-2,-1),keepdim=True)
        result=scale*(values/scale).pow(self.power).mean((-2,-1),keepdim=True).pow(1/self.power)
        return result.to(image.dtype)
