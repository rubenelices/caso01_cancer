"""Arquitecturas CNN propias del proyecto, sin modelos preentrenados."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from src.model import ShapeStep
from src.phase_representation import PhaseRepresentation
from src.attention_pooling import GatedSpatialPool
from src.spatial_pooling import make_downsampling_pool
from src.factorized_conv import FactorizedSpatialConv
from src.parallel_context import ParallelContextConv
from src.nonlinear_activation import make_activation
from src.power_mean_pool import FixedPowerMeanPool
from src.triplet_standardization import TripletStandardization


@dataclass(frozen=True)
class BaseCNNConfig:
    """Decisiones estructurales de la primera arquitectura candidata."""

    channels: tuple[int, ...] = (16, 32, 64, 128)
    convolutions_per_block: int = 2
    kernel_size: int = 3
    use_batch_norm: bool = True
    dropout: float = 0.30
    pool_position: str = "after_convolutions"
    input_representation: str = "raw"
    spatial_dropout: float = 0.0
    # None conserva exactamente BatchNorm (o ninguna normalización) anterior.
    group_norm_groups: int | None = None
    global_pooling: str = "average"
    # 1 conserva GAP y todas las claves de los checkpoints anteriores.
    global_pool_grid_size: int = 1
    downsampling_pool: str = "max"
    # Separación entre muestras del kernel; default conserva pesos/claves/RNG.
    dilation: int = 1
    # La primera convolución queda densa para fusionar las fases espacialmente.
    convolution_type: str = "standard"
    activation: str = "relu"
    input_normalization: str = "none"

    def __post_init__(self) -> None:
        if type(self.input_normalization) is not str or self.input_normalization not in {"none", "shared_mean_std"}:
            raise ValueError("input_normalization debe ser none o shared_mean_std")
        if type(self.activation) is not str or self.activation not in {"relu", "silu"}:
            raise ValueError("activation debe ser relu o silu")
        if type(self.convolution_type) is not str or self.convolution_type not in {
            "standard", "factorized_after_stem", "parallel_context"
        }:
            raise ValueError("convolution_type debe ser standard, factorized_after_stem o parallel_context")
        if self.convolution_type == "parallel_context" and (
                self.kernel_size != 3 or self.dilation != 1
                or any(type(c) is not int or c % 2 for c in self.channels)):
            raise ValueError("parallel_context exige kernel3,dilation1 y canales pares")
        if type(self.dilation) is not int or self.dilation not in {1, 2}:
            raise ValueError("dilation debe ser 1 o 2")
        if self.downsampling_pool not in {"max", "average"}:
            raise ValueError("downsampling_pool debe ser max o average")
        if (type(self.global_pool_grid_size) is not int
                or self.global_pool_grid_size not in {1, 2, 4}
                or (self.global_pool_grid_size != 1 and self.global_pooling != "average")):
            raise ValueError("global_pool_grid_size debe ser 1, 2 o 4; rejillas solo con average")
        if self.global_pooling not in {"average", "average_max", "gated_attention", "power_mean"}:
            raise ValueError("global_pooling debe ser average, average_max, gated_attention o power_mean")
        if self.global_pooling == "power_mean" and self.activation != "relu":
            raise ValueError("power_mean exige ReLU para features no negativos")
        if self.group_norm_groups is not None:
            if (type(self.group_norm_groups) is not int or self.group_norm_groups < 1
                    or not self.use_batch_norm
                    or any(c % self.group_norm_groups for c in self.channels)):
                raise ValueError("group_norm_groups debe dividir todos los canales y requiere use_batch_norm=True")
        if self.input_representation not in {"raw", "pre_differences", "temporal_differences"}:
            raise ValueError("input_representation no reconocida")
        if not self.channels or any(channel < 1 for channel in self.channels):
            raise ValueError("channels debe contener enteros positivos")
        if self.convolutions_per_block < 1:
            raise ValueError("convolutions_per_block debe ser positivo")
        if self.kernel_size % 2 == 0 or self.kernel_size < 1:
            raise ValueError("kernel_size debe ser impar y positivo")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout debe estar en [0, 1)")
        if not 0 <= self.spatial_dropout < 1:
            raise ValueError("spatial_dropout debe estar en [0, 1)")
        if self.pool_position not in {"after_convolutions", "between_convolutions"}:
            raise ValueError("pool_position no reconocido")
        if self.pool_position == "between_convolutions" and self.convolutions_per_block < 2:
            raise ValueError(
                "between_convolutions requiere al menos dos convoluciones por bloque"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ConvBlock(nn.Module):
    """Convoluciones que conservan HxW seguidas de pooling 2x2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convolutions: int,
        kernel_size: int,
        use_batch_norm: bool,
        pool_position: str,
        spatial_dropout: float = 0.0,
        group_norm_groups: int | None = None,
        downsampling_pool: str = "max",
        dilation: int = 1,
        convolution_type: str = "standard",
        keep_first_dense: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        layers: list[nn.Module] = []
        current_channels = in_channels
        for convolution_index in range(convolutions):
            factorized = (convolution_type == "factorized_after_stem"
                          and not (keep_first_dense and convolution_index == 0))
            if convolution_type == "parallel_context" and convolution_index == 0:
                layers.append(ParallelContextConv(
                    current_channels, out_channels, bias=not use_batch_norm))
            elif factorized:
                # Una unidad lógica contiene DW→PW; BN/ReLU van después de PW.
                layers.append(FactorizedSpatialConv(
                    current_channels, out_channels, kernel_size, dilation,
                    bias=not use_batch_norm,
                ))
            else:
                layers.append(nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=not use_batch_norm,
                ))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels) if group_norm_groups is None
                              else nn.GroupNorm(group_norm_groups, out_channels))
            layers.append(make_activation(activation))
            current_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = make_downsampling_pool(downsampling_pool)
        self.pool_position = pool_position
        self.first_convolution_end = 3 if use_batch_norm else 2
        # No mueve ni renombra capas aprendidas: los checkpoints anteriores
        # conservan sus claves. Identity no consume aleatoriedad cuando p=0.
        self.spatial_dropout = (
            nn.Dropout2d(spatial_dropout) if spatial_dropout > 0 else nn.Identity()
        )

    def forward(self, image: Tensor) -> Tensor:
        for layer_index, layer in enumerate(self.features, start=1):
            image = layer(image)
            if (
                self.pool_position == "between_convolutions"
                and layer_index == self.first_convolution_end
            ):
                image = self.pool(image)
        if self.pool_position == "after_convolutions":
            image = self.pool(image)
        # En train enmascara mapas aprendidos completos, no las fases de entrada.
        # En eval es identidad; resolución y número de canales nunca cambian.
        return self.spatial_dropout(image)


class GlobalAverageMaxPool(nn.Module):
    """Concatena media y máximo ESPACIALES: [B,C,H,W] -> [B,2C,1,1].

    No agrega cortes ni pacientes. No introduce pesos; conserva una descripción
    global y otra de respuestas locales intensas, que también podrían ser ruido.
    """

    def forward(self, image: Tensor) -> Tensor:
        average = image.mean(dim=(-2, -1), keepdim=True)
        maximum = image.amax(dim=(-2, -1), keepdim=True)
        return torch.cat((average, maximum), dim=1)


class BreastPCRNet(nn.Module):
    """Primera CNN candidata para predecir pCR desde PRE/EARLY/LATE.

    Recibe ``[B, 3, 256, 256]`` y devuelve logits ``[B]``. La arquitectura se
    construye exclusivamente con capas basicas de PyTorch y pesos inicializados
    desde cero.
    """

    def __init__(self, config: BaseCNNConfig = BaseCNNConfig()) -> None:
        super().__init__()
        self.config = config
        self.input_normalizer = TripletStandardization(config.input_normalization)
        self.input_transform = PhaseRepresentation(config.input_representation)
        blocks: list[nn.Module] = []
        # El contrato externo sigue siendo PRE/EARLY/LATE. E42 elimina PRE
        # como canal directo dentro del modelo, no del cargador ni de la web.
        in_channels = self.input_transform.output_channels
        for block_index, out_channels in enumerate(config.channels):
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    convolutions=config.convolutions_per_block,
                    kernel_size=config.kernel_size,
                    use_batch_norm=config.use_batch_norm,
                    pool_position=config.pool_position,
                    spatial_dropout=config.spatial_dropout,
                    group_norm_groups=config.group_norm_groups,
                    downsampling_pool=config.downsampling_pool,
                    dilation=config.dilation,
                    convolution_type=config.convolution_type,
                    keep_first_dense=block_index == 0,
                    activation=config.activation,
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.global_pool = (nn.AdaptiveAvgPool2d((config.global_pool_grid_size,) * 2)
                            if config.global_pooling != "average_max" else GlobalAverageMaxPool())
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(config.dropout)
        features = config.channels[-1] * (2 if config.global_pooling == "average_max" else 1)
        if config.global_pooling == "average":
            features *= config.global_pool_grid_size ** 2
        self.classifier = nn.Linear(features, 1)
        self.reset_parameters()
        if config.global_pooling == "power_mean":
            self.global_pool = FixedPowerMeanPool(power=3., epsilon=1e-6)
        if config.global_pooling == "gated_attention":
            # Crear después del reset clásico conserva inicialización del extractor
            # y clasificador compartidos para una misma semilla; no cargar E19.
            self.global_pool = GatedSpatialPool(config.channels[-1])

    def reset_parameters(self) -> None:
        depthwise_layers = {layer.depthwise for layer in self.modules()
                            if isinstance(layer, FactorizedSpatialConv)}
        parallel_children = {branch for unit in self.modules()
                             if isinstance(unit, ParallelContextConv)
                             for branch in (unit.local, unit.context)}
        attention_children = (set(self.global_pool.modules())
                              if isinstance(self.global_pool, GatedSpatialPool) else set())
        for module in self.modules():
            if module in attention_children:
                continue
            if isinstance(module, ParallelContextConv):
                module.reset_branch_parameters()
                continue
            if module in parallel_children:
                continue
            if isinstance(module, nn.Conv2d):
                if module in depthwise_layers:
                    # No ReLU entre DW y PW: ganancia lineal, fan_in por grupo.
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        if isinstance(self.global_pool, GatedSpatialPool):
            self.global_pool.reset_parameters()

    def forward(self, image: Tensor) -> Tensor:
        image = self.input_transform(self.input_normalizer(image))
        for block in self.blocks:
            image = block(image)
        image = self.global_pool(image)
        image = self.flatten(image)
        image = self.dropout(image)
        return self.classifier(image).squeeze(1)

    @torch.no_grad()
    def trace_shapes(self, batch_size: int = 2) -> list[ShapeStep]:
        """Inspección ficticia sin alterar modos, BN ni RNG de dropout."""
        modes = {layer: layer.training for layer in self.modules()}
        try:
            self.eval()
            return self._trace_shapes(batch_size)
        finally:
            for layer, training in modes.items():
                layer.training = training

    def _trace_shapes(self, batch_size: int) -> list[ShapeStep]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        image = torch.zeros(batch_size, 3, 256, 256, device=device, dtype=dtype)
        trace = [ShapeStep("entrada", tuple(image.shape))]
        image = self.input_normalizer(image)
        if self.config.input_normalization != "none":
            trace.append(ShapeStep("normalización conjunta por tripleta",tuple(image.shape)))
        image = self.input_transform(image)
        if self.config.input_representation != "raw":
            label = ("EARLY-PRE, LATE-EARLY" if self.config.input_representation == "temporal_differences"
                     else "PRE, EARLY-PRE, LATE-EARLY")
            trace.append(ShapeStep(label, tuple(image.shape)))
        for index, block in enumerate(self.blocks, start=1):
            image = block(image)
            trace.append(ShapeStep(f"bloque {index}", tuple(image.shape)))
        image = self.global_pool(image)
        trace.append(ShapeStep(("global average pooling" if self.config.global_pool_grid_size == 1
                               else f"spatial average pooling {self.config.global_pool_grid_size}x{self.config.global_pool_grid_size}") if self.config.global_pooling == "average"
                               else "gated spatial attention pooling" if self.config.global_pooling == "gated_attention"
                               else "power mean pooling p3 fixed" if self.config.global_pooling == "power_mean"
                               else "global average + max pooling", tuple(image.shape)))
        image = self.flatten(image)
        trace.append(ShapeStep("flatten", tuple(image.shape)))
        logits = self.classifier(image).squeeze(1)
        trace.append(ShapeStep("clasificador", tuple(logits.shape)))
        return trace


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_breakdown(model: nn.Module) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "parameters": parameter.numel(),
        }
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
