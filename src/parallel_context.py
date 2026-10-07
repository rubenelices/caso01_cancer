"""Dos convoluciones propias paralelas: contexto local y dilatado.

Componente aislado hasta una futura integración verificada. No importa redes,
pesos ni código externo; CNN principal controla inicialización de sus ramas.
"""
from torch import Tensor, nn
import torch
import math


class ChannelConcatenation(nn.Module):
    """Une canales, no pacientes ni dimensiones espaciales."""
    def forward(self, local: Tensor, context: Tensor) -> Tensor:
        return torch.cat((local,context),dim=1)


class ParallelContextConv(nn.Module):
    """[B,Cin,H,W] → dos [B,Cout/2,H,W] → [B,Cout,H,W].

    Kernel3×3 stride1; dilatación/padding1/1 y2/2. Ambos consumen la MISMA
    entrada. No normalización/activación entre ramas y concatenación. La CNN
    aplica una BN y ReLU al resultado completo, igual que a una Conv densa.
    Parámetros9*Cin*Cout (+Cout si bias), igual que una Conv3×3 convencional.
    No equivale a todos los kernels5×5; solo 9 pesos por canal/salida/ramita.
    """
    def __init__(self,in_channels:int,out_channels:int,bias:bool=False) -> None:
        super().__init__()
        if (type(in_channels) is not int or in_channels<1
                or type(out_channels) is not int or out_channels<2 or out_channels%2):
            raise ValueError('Contexto paralelo exige Cin positivo y Cout positivo par')
        if type(bias) is not bool:
            raise ValueError('bias debe ser booleano')
        self.in_channels,self.out_channels=in_channels,out_channels
        self.local=nn.Conv2d(in_channels,out_channels//2,3,padding=1,dilation=1,bias=bias)
        self.context=nn.Conv2d(in_channels,out_channels//2,3,padding=2,dilation=2,bias=bias)
        self.concatenate=ChannelConcatenation()

    def forward(self,image:Tensor) -> Tensor:
        return self.concatenate(self.local(image),self.context(image))

    def reset_branch_parameters(self) -> None:
        """Kaiming/ReLU con fan_out TOTAL, no medio Cout de cada rama.

        Lo llama explícitamente la CNN en su reset; el constructor aislado
        conserva reset de Conv2d. No hay pesos externos ni una capa nueva.
        Ambas ramas representan mitades de una única salida de Cout canales.
        """
        std=math.sqrt(2.0/(self.out_channels*9))
        for branch in (self.local,self.context):
            nn.init.normal_(branch.weight,mean=0.0,std=std)
            if branch.bias is not None:
                nn.init.zeros_(branch.bias)
