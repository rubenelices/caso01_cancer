"""Normalización conjunta por tripleta, independiente y desactivada por defecto.

No aprende estadísticas del dataset ni normaliza cada fase por separado.
Integrada en la CNN con mode none por defecto; E53 activa shared_mean_std.
"""
import math
import torch
from torch import Tensor,nn


class TripletStandardization(nn.Module):
    def __init__(self,mode:str='none',epsilon:float=1e-6) -> None:
        super().__init__()
        if type(mode) is not str or mode not in {'none','shared_mean_std'}:
            raise ValueError('input_normalization debe ser none o shared_mean_std')
        if type(epsilon) not in (int,float) or not math.isfinite(epsilon) or epsilon<=0:
            raise ValueError('epsilon debe ser finito positivo')
        self.mode,self.epsilon=mode,epsilon

    def forward(self,image:Tensor) -> Tensor:
        if self.mode=='none':return image
        if (image.ndim!=4 or not image.numel() or image.shape[1]!=3
                or image.dtype not in {torch.float32,torch.float64}):
            raise ValueError('Normalización conjunta exige [B,3,H,W] no vacío float32/float64')
        if not torch.isfinite(image).all():
            raise ValueError('Normalización conjunta exige valores finitos')
        # El ancla común conserva la normalización matemática y evita que
        # el redondeo de mean(.8) en MPS cree señal en tripletas constantes.
        centered=image-image[:, :1, :1, :1]
        variance,mean=torch.var_mean(centered,dim=(1,2,3),correction=0,keepdim=True)
        # Clamp antes de sqrt evita gradientes NaN ante tripleta constante.
        scale=variance.clamp_min(self.epsilon**2).sqrt()
        return (centered-mean)/scale
