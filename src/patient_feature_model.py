"""CNN propia compartida y cabeza no lineal para bolsas de cortes.

No modifica BreastPCRNet ni carga pesos anteriores. La media es invariante al
orden y NO modela explícitamente relaciones axiales o interacciones por pares.
"""
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from src.architectures import BaseCNNConfig, BreastPCRNet


def group_features(features: Tensor, targets: Tensor, patient_ids: Sequence[str]):
    if (features.ndim != 2 or features.shape[0] == 0
            or targets.shape != (features.shape[0],)
            or len(patient_ids) != features.shape[0]):
        raise ValueError('Features, etiquetas e IDs incompatibles')
    if (not features.is_floating_point() or not torch.isfinite(features).all()
            or targets.device != features.device
            or not torch.isfinite(targets).all()
            or not ((targets == 0) | (targets == 1)).all()):
        raise ValueError('Features/etiquetas no válidas')
    groups = {}
    for index, pid in enumerate(patient_ids):
        if not isinstance(pid, str) or not pid.strip():
            raise ValueError('ID de paciente inválido')
        groups.setdefault(pid, []).append(index)
    means, labels = [], []
    for indices in groups.values():
        index = torch.tensor(indices, device=features.device)
        ys = targets.index_select(0, index)
        if not (ys == ys[0]).all():
            raise ValueError('Etiquetas contradictorias para una paciente')
        means.append(features.index_select(0, index).mean(0))
        labels.append(ys[0])
    return torch.stack(means), torch.stack(labels), list(groups)


class PatientFeatureNet(nn.Module):
    """[N,3,256,256]→[N,128]→media por paciente→MLP128→32→1.

    forward habitual interpreta cada corte como bolsa de tamaño uno. El dropout
    va en la cabeza después de ReLU; no altera features antes de la media.
    """
    def __init__(self, config: BaseCNNConfig, hidden: int = 32):
        super().__init__()
        if (config.channels[-1] != 128 or config.global_pooling != 'average'
                or config.global_pool_grid_size != 1 or type(hidden) is not int
                or hidden != 32):
            raise ValueError('Esta prueba exige GAP128 y cabeza32 prefijada')
        self.encoder = BreastPCRNet(config)
        # Eliminar la cabeza lineal anterior: no quedan parámetros huérfanos.
        self.encoder.classifier = nn.Identity()
        self.encoder.dropout = nn.Identity()
        self.head = nn.Sequential(nn.Linear(128, hidden), nn.ReLU(),
                                  nn.Dropout(config.dropout), nn.Linear(hidden, 1))
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def extract_features(self, image: Tensor) -> Tensor:
        image = self.encoder.input_transform(self.encoder.input_normalizer(image))
        for block in self.encoder.blocks:
            image = block(image)
        return self.encoder.flatten(self.encoder.global_pool(image))

    def classify_features(self, features: Tensor) -> Tensor:
        return self.head(features).squeeze(-1)

    def forward(self, image: Tensor) -> Tensor:
        return self.classify_features(self.extract_features(image))

    def forward_patients(self, image: Tensor, targets: Tensor, patient_ids: Sequence[str]):
        means, labels, ids = group_features(self.extract_features(image), targets, patient_ids)
        return self.classify_features(means), labels, ids


def load_feature_checkpoint(path, device='cpu'):
    """Restauración explícita propia; no confundir con checkpoints BreastPCRNet."""
    from src.experiment_config import experiment_config_from_dict
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('model_class') != 'PatientFeatureNet_v1':
        raise ValueError('No es un checkpoint de características por paciente')
    config = experiment_config_from_dict(checkpoint['config']['experiment'])
    model = PatientFeatureNet(config.model).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.eval()
