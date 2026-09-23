"""Pipeline unico de datos para BreastDCEDL.

Este modulo es la unica puerta de entrada de PRE, EARLY y LATE. El Dataset de
entrenamiento y la futura aplicacion web deben reutilizar ``load_dce_triplet``
para garantizar que una misma muestra produce exactamente el mismo tensor.

No se aplica normalizacion de ImageNet. Los PNG ya conservan una ventana comun
entre fases; aqui solo se convierten de uint8 [0, 255] a float32 [0, 1].
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


PHASES = ("PRE", "EARLY", "LATE")
PHASE_COLUMNS = {
    "PRE": "path_pre",
    "EARLY": "path_early",
    "LATE": "path_late",
}
EXPECTED_IMAGE_SIZE = (256, 256)

ImageSource: TypeAlias = str | Path | bytes | bytearray | BinaryIO
TensorTransform: TypeAlias = Callable[[Tensor], Tensor]


class DataValidationError(ValueError):
    """La entrada no cumple el contrato de datos del proyecto."""


@dataclass(frozen=True)
class DatasetSplits:
    """Particiones publicas con validacion interna por paciente."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_fold: int

    def patient_counts(self) -> dict[str, int]:
        return {
            "train": int(self.train["patient_id"].nunique()),
            "validation": int(self.validation["patient_id"].nunique()),
            "test": int(self.test["patient_id"].nunique()),
        }

    def sample_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


@dataclass(frozen=True)
class LoaderConfig:
    """Configuracion reproducible de los DataLoaders."""

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size debe ser positivo")
        if self.num_workers < 0:
            raise ValueError("num_workers no puede ser negativo")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requiere num_workers > 0")


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def load_samples(root: str | Path = "breastdcedl") -> pd.DataFrame:
    """Carga samples.csv sin mezclar ni transformar las particiones."""

    root = Path(root)
    samples = pd.read_csv(root / "metadata" / "samples.csv")
    required = {
        "sample_id",
        "patient_id",
        "split",
        "path_pre",
        "path_early",
        "path_late",
        "slice_index",
        "pCR",
        "fold",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise DataValidationError(f"Faltan columnas en samples.csv: {missing}")
    return samples


def split_by_patient_fold(samples: pd.DataFrame, validation_fold: int = 0) -> DatasetSplits:
    """Separa train/validacion/test respetando pacientes.

    La columna ``fold`` ya fue construida por paciente. Test se devuelve para su
    uso final, pero no debe iterarse durante el desarrollo del modelo.
    """

    if validation_fold not in range(5):
        raise ValueError("validation_fold debe estar entre 0 y 4")

    public_train = samples.loc[samples["split"] == "train"].copy()
    test = samples.loc[samples["split"] == "test"].copy()
    train = public_train.loc[public_train["fold"] != validation_fold].copy()
    validation = public_train.loc[public_train["fold"] == validation_fold].copy()

    split_frames = {"train": train, "validation": validation, "test": test}
    patient_sets = {
        name: set(frame["patient_id"].astype(str)) for name, frame in split_frames.items()
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for first, second in pairs:
        overlap = patient_sets[first] & patient_sets[second]
        if overlap:
            examples = sorted(overlap)[:5]
            raise DataValidationError(
                f"Fuga de pacientes entre {first} y {second}: "
                f"{len(overlap)}; ejemplos={examples}"
            )

    if len(train) + len(validation) != len(public_train):
        raise DataValidationError("Train interno y validacion no reconstruyen train publico")
    if train.empty or validation.empty or test.empty:
        raise DataValidationError("Alguna particion ha quedado vacia")

    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        validation_fold=validation_fold,
    )


def balanced_patient_subset(
    rows: pd.DataFrame,
    patients_per_class: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Selecciona pacientes completos de ambas clases para pruebas tecnicas.

    Nunca selecciona cortes de forma independiente. Todos los cortes de cada
    paciente elegida permanecen juntos y la seleccion es reproducible.
    """

    if patients_per_class < 1:
        raise ValueError("patients_per_class debe ser positivo")
    label_counts = rows.groupby("patient_id")["pCR"].nunique()
    if (label_counts > 1).any():
        raise DataValidationError("La etiqueta cambia dentro de una paciente")

    patients = rows.groupby("patient_id", as_index=False)["pCR"].first()
    selected: list[str] = []
    for label in (0, 1):
        candidates = patients.loc[patients["pCR"] == label, "patient_id"].astype(str)
        if len(candidates) < patients_per_class:
            raise DataValidationError(
                f"Clase {label}: se pidieron {patients_per_class} pacientes y hay "
                f"{len(candidates)}"
            )
        chosen = candidates.sample(n=patients_per_class, random_state=seed + label)
        selected.extend(chosen.tolist())

    subset = rows.loc[rows["patient_id"].astype(str).isin(selected)].copy()
    if subset["patient_id"].nunique() != 2 * patients_per_class:
        raise DataValidationError("El subconjunto no conserva las pacientes esperadas")
    return subset.reset_index(drop=True)


def limit_splits_for_debug(
    splits: DatasetSplits,
    train_patients_per_class: int | None,
    validation_patients_per_class: int | None,
    seed: int = 42,
) -> DatasetSplits:
    """Limita train/validacion por paciente; test se mantiene intacto y cerrado."""

    train = splits.train
    validation = splits.validation
    if train_patients_per_class is not None:
        train = balanced_patient_subset(train, train_patients_per_class, seed)
    if validation_patients_per_class is not None:
        validation = balanced_patient_subset(
            validation, validation_patients_per_class, seed + 10_000
        )
    train_ids = set(train["patient_id"].astype(str))
    validation_ids = set(validation["patient_id"].astype(str))
    if train_ids & validation_ids:
        raise DataValidationError("Fuga de pacientes en el subconjunto de depuracion")
    return DatasetSplits(
        train=train,
        validation=validation,
        test=splits.test,
        validation_fold=splits.validation_fold,
    )


def resolve_inside_root(root: str | Path, relative_path: str | Path) -> Path:
    """Resuelve una ruta relativa y rechaza escapes fuera de la raiz."""

    root_path = Path(root).resolve()
    candidate = (root_path / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise DataValidationError(
            f"La ruta sale de la raiz permitida: {relative_path}"
        ) from exc
    return candidate


def _open_source(source: ImageSource) -> tuple[Any, bool]:
    """Devuelve una fuente aceptada por Pillow y si debe cerrarse despues."""

    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source), True
    if isinstance(source, (str, Path)):
        return Path(source), False
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        return source, False
    raise TypeError(f"Fuente de imagen no soportada: {type(source)!r}")


def decode_phase_png(
    source: ImageSource,
    *,
    expected_size: tuple[int, int] = EXPECTED_IMAGE_SIZE,
) -> Tensor:
    """Decodifica un PNG gris en un tensor float32 [H, W] dentro de [0, 1]."""

    opened_source, close_source = _open_source(source)
    try:
        try:
            with Image.open(opened_source) as image:
                image.load()
                if image.format != "PNG":
                    raise DataValidationError(
                        f"Formato invalido: {image.format}; se esperaba PNG"
                    )
                if image.mode != "L":
                    raise DataValidationError(
                        f"Modo invalido: {image.mode}; se esperaba escala de grises L"
                    )
                if image.size != expected_size:
                    raise DataValidationError(
                        f"Tamano invalido: {image.size}; se esperaba {expected_size}"
                    )
                array = np.asarray(image, dtype=np.uint8).copy()
        except (OSError, UnidentifiedImageError) as exc:
            raise DataValidationError(f"No se pudo decodificar la imagen: {exc}") from exc
    finally:
        if close_source:
            opened_source.close()

    tensor = torch.from_numpy(array).to(dtype=torch.float32).div_(255.0)
    if tensor.shape != (expected_size[1], expected_size[0]):
        raise DataValidationError(f"Forma inesperada tras decodificar: {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise DataValidationError("La imagen contiene valores no finitos")
    return tensor.contiguous()


def load_dce_triplet(
    sources: Mapping[str, ImageSource],
    *,
    expected_size: tuple[int, int] = EXPECTED_IMAGE_SIZE,
) -> Tensor:
    """Apila PRE, EARLY y LATE en un tensor [3, H, W].

    Esta funcion sirve tanto para rutas del dataset como para bytes recibidos por
    la futura aplicacion web. Exige exactamente las tres fases y fija su orden.
    """

    provided = set(sources)
    expected = set(PHASES)
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing or extra:
        raise DataValidationError(
            f"Fases invalidas; ausentes={missing}, adicionales={extra}"
        )

    channels = [
        decode_phase_png(sources[phase], expected_size=expected_size) for phase in PHASES
    ]
    image = torch.stack(channels, dim=0)
    expected_shape = (len(PHASES), expected_size[1], expected_size[0])
    if image.shape != expected_shape:
        raise DataValidationError(
            f"Tensor DCE con forma {tuple(image.shape)}; se esperaba {expected_shape}"
        )
    return image.contiguous()


def row_sources(row: Any, root: str | Path) -> dict[str, Path]:
    """Convierte las rutas de una fila de samples.csv en fuentes seguras."""

    return {
        phase: resolve_inside_root(root, getattr(row, column))
        for phase, column in PHASE_COLUMNS.items()
    }


class BreastDCEDataset(Dataset[dict[str, Any]]):
    """Dataset trazable: imagen, etiqueta e identificadores de cada corte.

    ``transform`` recibe el tensor completo [3, H, W]. Esta decision hace
    imposible aplicar por accidente una transformacion geometrica distinta a
    cada fase desde dentro del Dataset.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        root: str | Path = "breastdcedl",
        transform: TensorTransform | None = None,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.root = Path(root).resolve()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        image = load_dce_triplet(row_sources(row, self.root))
        if self.transform is not None:
            image = self.transform(image)
        self._validate_transformed_tensor(image)

        return {
            "image": image,
            "target": torch.tensor(float(row.pCR), dtype=torch.float32),
            "patient_id": str(row.patient_id),
            "sample_id": str(row.sample_id),
            "slice_index": int(row.slice_index),
            "split": str(row.split),
            "fold": int(row.fold),
        }

    @staticmethod
    def _validate_transformed_tensor(image: Tensor) -> None:
        if not isinstance(image, Tensor):
            raise DataValidationError("transform debe devolver un torch.Tensor")
        if image.shape != (3, 256, 256):
            raise DataValidationError(
                f"transform devolvio forma {tuple(image.shape)}; se esperaba (3, 256, 256)"
            )
        if image.dtype != torch.float32:
            raise DataValidationError(
                f"transform devolvio dtype {image.dtype}; se esperaba torch.float32"
            )
        if not torch.isfinite(image).all():
            raise DataValidationError("transform introdujo valores no finitos")


def seed_worker(worker_id: int) -> None:
    """Sincroniza semillas de NumPy y random en cada proceso del DataLoader."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    splits: DatasetSplits,
    root: str | Path = "breastdcedl",
    config: LoaderConfig = LoaderConfig(),
    train_transform: TensorTransform | None = None,
) -> DataLoaders:
    """Crea loaders: train aleatorio; validacion y test secuenciales."""

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers,
        "worker_init_fn": seed_worker,
    }
    train_dataset = BreastDCEDataset(splits.train, root, transform=train_transform)
    validation_dataset = BreastDCEDataset(splits.validation, root)
    test_dataset = BreastDCEDataset(splits.test, root)

    return DataLoaders(
        train=DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            **common,
        ),
        validation=DataLoader(validation_dataset, shuffle=False, **common),
        test=DataLoader(test_dataset, shuffle=False, **common),
    )


def cut_level_pos_weight(rows: pd.DataFrame) -> float:
    """Calcula N0/N1 solo sobre las filas de entrenamiento recibidas."""

    positives = int((rows["pCR"] == 1).sum())
    negatives = int((rows["pCR"] == 0).sum())
    if positives == 0 or negatives == 0:
        raise DataValidationError("Se necesitan ambas clases para calcular pos_weight")
    return negatives / positives


def inspect_pipeline(
    root: str | Path,
    validation_fold: int,
    loader_config: LoaderConfig,
) -> dict[str, Any]:
    """Ejecuta un smoke test sin leer imagenes del conjunto test."""

    samples = load_samples(root)
    splits = split_by_patient_fold(samples, validation_fold)
    loaders = create_dataloaders(splits, root, loader_config)

    train_batch = next(iter(loaders.train))
    validation_batch = next(iter(loaders.validation))
    direct = loaders.validation.dataset[0]
    same_tensor = torch.equal(validation_batch["image"][0], direct["image"])
    same_sample = validation_batch["sample_id"][0] == direct["sample_id"]
    if not (same_tensor and same_sample):
        raise RuntimeError("Validacion no conserva el orden secuencial esperado")

    return {
        "root": str(Path(root).resolve()),
        "validation_fold": validation_fold,
        "loader_config": asdict(loader_config),
        "patients": splits.patient_counts(),
        "samples": splits.sample_counts(),
        "train_batch": {
            "image_shape": list(train_batch["image"].shape),
            "image_dtype": str(train_batch["image"].dtype),
            "target_shape": list(train_batch["target"].shape),
            "min": float(train_batch["image"].min()),
            "max": float(train_batch["image"].max()),
        },
        "validation_batch": {
            "image_shape": list(validation_batch["image"].shape),
            "first_sample_id": validation_batch["sample_id"][0],
            "sequential_identity_check": True,
        },
        "test_pixels_read": False,
        "cut_level_pos_weight": cut_level_pos_weight(splits.train),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("breastdcedl"))
    parser.add_argument("--validation-fold", type=int, default=0, choices=range(5))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/data_pipeline/summary.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = LoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        seed=args.seed,
    )
    summary = inspect_pipeline(args.root, args.validation_fold, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Pipeline de datos: OK")
    print(f"Forma batch train: {summary['train_batch']['image_shape']}")
    print(f"Pacientes: {summary['patients']}")
    print(f"Cortes: {summary['samples']}")
    print(f"pos_weight de train interno: {summary['cut_level_pos_weight']:.4f}")
    print("Test: indexado, pero sus pixeles no se han leido")
    print(f"Resumen: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
