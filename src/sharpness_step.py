"""Paso SAM propio y aislado; todavía no conectado al entrenamiento científico.

Dos gradientes del mismo lote: en w y en w+rho*g/||g||. El optimizador base
se aplica una sola vez, en los pesos originales. No importa modelos ni pesos.
"""
from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SAMConfig:
    """Preparación tipada; desactivada conserva entrenamiento habitual."""
    enabled: bool = False
    rho: float = 0.05

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("sam.enabled debe ser booleano")
        if type(self.rho) not in (int, float) or not math.isfinite(self.rho) or not 0 < self.rho <= 1:
            raise ValueError("sam.rho debe estar en (0,1]")


def _random_state(device: torch.device) -> tuple[Tensor, Tensor | None]:
    extra = None
    if device.type == "mps":
        extra = torch.mps.get_rng_state()
    elif device.type == "cuda":
        extra = torch.cuda.get_rng_state(device)
    return torch.get_rng_state(), extra


def _restore_random(state: tuple[Tensor, Tensor | None], device: torch.device) -> None:
    torch.set_rng_state(state[0])
    if device.type == "mps" and state[1] is not None:
        torch.mps.set_rng_state(state[1])
    elif device.type == "cuda" and state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


@torch.no_grad()
def _restore_values(original: list[tuple[Tensor, Tensor]]) -> None:
    for value, saved in original:
        value.copy_(saved)


def sam_step(model: nn.Module, optimizer: torch.optim.Optimizer,
             closure: Callable[[], Tensor], rho: float = 0.05) -> Tensor:
    """SAM L2 no adaptativo en float32/64; devuelve loss original, separada.

La closure solo calcula una pérdida escalar diferenciable del mismo lote; no
muestrea datos ni hace backward/step. Reutilizamos máscaras Dropout y dejamos
avanzar RNG y buffers (incluido BN) solo como en la primera pasada. La segunda
usa estadísticas de lote normales, no cambia model.train() a eval().
Errores antes de optimizer.step restauran pesos/buffers/RNG y limpian grads.
No garantiza deshacer efectos internos de un optimizador que falle en step().
"""
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho < 0:
        raise ValueError("rho debe ser finito y no negativo")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("SAM requiere parámetros entrenables")
    device = parameters[0].device
    if torch.is_autocast_enabled(device.type):
        raise ValueError("SAM no admite autocast/AMP")
    if any(p.device != device or p.dtype not in {torch.float32, torch.float64} for p in parameters):
        raise ValueError("SAM requiere un solo dispositivo y float32/64 sin AMP")
    optimized = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    if len(optimized) != len(parameters) or {id(p) for p in optimized} != {id(p) for p in parameters}:
        raise ValueError("El optimizador debe incluir exactamente los parámetros entrenables")
    weights = [(p, p.detach().clone()) for p in parameters]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    before_random = _random_state(device)
    optimizer.zero_grad(set_to_none=True)
    try:
        loss = closure()
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise ValueError("Pérdida SAM no escalar o no finita")
        loss.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item():
            raise ValueError("Gradientes SAM ausentes o no finitos")
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
        if not torch.isfinite(norm).item():
            raise ValueError("Norma de gradientes SAM no finita")
        first_buffers = [(b, b.detach().clone()) for b in model.buffers()]
        after_random = _random_state(device)
        with torch.no_grad():
            scale = rho / norm.clamp_min(torch.finfo(norm.dtype).tiny)
            for p in parameters:
                if p.grad is not None:
                    p.add_(p.grad * scale)
        optimizer.zero_grad(set_to_none=True)
        _restore_random(before_random, device)
        second_loss = closure()
        if second_loss.ndim != 0 or not torch.isfinite(second_loss).item():
            raise ValueError("Segunda pérdida SAM no escalar o no finita")
        second_loss.backward()
        second_gradients = [p.grad for p in parameters if p.grad is not None]
        if not second_gradients or not torch.stack([torch.isfinite(g).all() for g in second_gradients]).all().item():
            raise ValueError("Segundo gradiente SAM no finito")
        _restore_values(weights)
        _restore_values(first_buffers)
        _restore_random(after_random, device)
    except BaseException:
        _restore_values(weights)
        _restore_values(buffers)
        _restore_random(before_random, device)
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.step()
    return loss.detach()
