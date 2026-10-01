"""Genera fichas y diagramas desde las capas reales, sin leer el dataset.

Ejecutar: python -m src.document_architectures
Las figuras SVG y PNG son documentación propia versionable.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
import torch
from torch import nn

from src.architectures import BreastPCRNet, trainable_parameter_count
from src.experiment_config import load_experiment_config
from src.model import MinimalCNN
from src.phase_representation import PhaseRepresentation


ROOT = Path(__file__).resolve().parents[1]


def trace(model: nn.Module) -> list[dict]:
    """Captura el orden real de ejecución, también cuando el pool es intermedio."""
    steps = [{"label": "Entrada\n3 fases DCE", "shape": (3, 256, 256),
              "kind": "input", "parameters": 0, "rf": 1, "jump": 1}]
    handles = []
    rf, jump = 1, 1

    def record(layer, inputs, output):
        nonlocal rf, jump
        shape = tuple(output.shape[1:])
        parameters = sum(p.numel() for p in layer.parameters(recurse=False))
        if isinstance(layer, PhaseRepresentation):
            if layer.mode == "raw":
                return
            label, kind = "PRE\nEARLY−PRE\nLATE−EARLY", "input"
            detail = "restas fijas; sin clipping, pesos, kernel ni cambio espacial"
        elif isinstance(layer, nn.Conv2d):
            kernel, stride, padding = layer.kernel_size[0], layer.stride[0], layer.padding[0]
            rf += (kernel - 1) * jump
            jump *= stride
            label = f"Conv {kernel}×{kernel}\nBN + ReLU" if has_bn else f"Conv {kernel}×{kernel}\nReLU"
            kind = "conv"
            detail = f"kernel={kernel}, padding={padding}, stride={stride}"
        elif isinstance(layer, nn.MaxPool2d):
            kernel, stride = layer.kernel_size, layer.stride
            rf += (kernel - 1) * jump
            jump *= stride
            label, kind = "MaxPool\n2×2", "pool"
            detail = "kernel=2, padding=0, stride=2"
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            label = "GAP\npromedio global" if shape[-1] == 1 else "AvgPool\nadaptativo 4×4"
            kind, detail = "head", f"salida espacial={shape[-2:]}"
        elif isinstance(layer, nn.Flatten):
            # El vector se documenta en la tabla; no añade un volumen redundante.
            label, kind, detail = "Flatten", "flatten", "reorganiza; sin pesos"
        elif isinstance(layer, nn.Dropout):
            label, kind, detail = f"Dropout\np={layer.p:g}", "head", "solo durante entrenamiento"
        elif isinstance(layer, nn.Linear):
            label, kind = "Lineal\n1 logit", "head"
            detail = f"{layer.in_features} → {layer.out_features}"
        else:
            return
        steps.append(dict(label=label, shape=shape, kind=kind, parameters=parameters,
                          rf=rf, jump=jump, detail=detail))

    has_bn = any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
    for layer in model.modules():
        if isinstance(layer, (PhaseRepresentation, nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d,
                              nn.Flatten, nn.Dropout, nn.Linear)):
            handles.append(layer.register_forward_hook(record))
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(1, 3, 256, 256))
    finally:
        for handle in handles:
            handle.remove()
    if isinstance(model, MinimalCNN):
        # La CNN mínima realiza flatten con torch.flatten, sin módulo independiente.
        linear_index = next(i for i, step in enumerate(steps) if step["label"].startswith("Lineal"))
        steps.insert(linear_index, dict(label="Flatten", shape=(512,), kind="flatten",
                                      parameters=0, rf=rf, jump=jump,
                                      detail="32 × 4 × 4 = 512 valores"))
    return steps


def dimensions(shape: tuple) -> str:
    if len(shape) == 3:
        return f"{shape[0]} @ {shape[1]}²" if shape[1] == shape[2] else f"{shape[0]} @ {shape[1]}×{shape[2]}"
    return " × ".join(map(str, shape))


def draw(steps: list[dict], title: str, subtitle: str, params: int, path: Path) -> None:
    visible = [step for step in steps if step["kind"] != "flatten"]
    n = len(visible)
    fig, ax = plt.subplots(figsize=(max(22, n * 1.65), 10))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(-0.7, n + 1.7), ylim=(-2.4, 5.0))
    ax.axis("off")
    navy, muted = "#17283f", "#61748c"
    palette = {"input": ("#e7efff", "#83ade9"), "conv": ("#e3f3ed", "#75bda7"),
               "pool": ("#fff0df", "#e8a358"), "head": ("#efe7f6", "#a188c1")}
    ax.text(-0.5, 4.55, title, fontsize=27, color=navy, fontfamily="DejaVu Serif")
    ax.text(-0.5, 4.12, subtitle, fontsize=15, color=muted)
    # Cada ancho indica canales de forma esquemática; altura disminuye con H/W.
    for i, step in enumerate(visible):
        shape = step["shape"]
        channels = shape[0]
        side = shape[-1] if len(shape) == 3 else 1
        height = 0.65 + 2.05 * math.log2(max(side, 1) + 1) / math.log2(257)
        width = 0.18 + 0.40 * min(math.log2(channels + 1) / 8, 1)
        if len(shape) != 3:
            height = 0.55 + 0.85 * min(channels / 128, 1)
        x, y, dx, dy = i - width / 2, 1.50 - height / 2, 0.17, 0.23
        fill, edge = palette[step["kind"]]
        faces = [[(x, y), (x + width, y), (x + width, y + height), (x, y + height)],
                 [(x, y + height), (x + dx, y + height + dy),
                  (x + width + dx, y + height + dy), (x + width, y + height)],
                 [(x + width, y), (x + width + dx, y + dy),
                  (x + width + dx, y + height + dy), (x + width, y + height)]]
        for face in faces:
            ax.add_patch(Polygon(face, closed=True, facecolor=fill, edgecolor=edge, linewidth=1.8))
        ax.text(i + 0.05, 3.43, step["label"], ha="center", va="center", fontsize=11, color=navy, linespacing=1.5)
        ax.text(i + 0.05, -0.28, dimensions(shape), ha="center", fontsize=12, color=muted)
        if i < n - 1:
            ax.annotate("", xy=(i + 0.62, 1.48), xytext=(i + width / 2 + dx + 0.03, 1.48),
                        arrowprops=dict(arrowstyle="->", color="#b3beca", lw=1))
    ax.annotate("", xy=(n + 0.10, 1.48), xytext=(n - 0.55, 1.48),
                arrowprops=dict(arrowstyle="-|>", color=navy, lw=2.4))
    ax.text(n + 0.13, 1.8, "Sigmoide", fontsize=13, color=navy)
    ax.text(n + 0.13, 1.28, "Probabilidad\ndel corte", fontsize=13, color=muted, linespacing=1.6)
    ax.text(n + 0.13, 0.42, "↓ Media de cortes\npor paciente", fontsize=12, color=muted, linespacing=1.5)
    ax.add_patch(FancyBboxPatch((-0.5, -2.15), n + 1.85, 1.25, boxstyle="round,pad=0.12",
                               facecolor="#edf3fc", edgecolor="#aac5ed", linewidth=1.5))
    ax.text(-0.22, -1.15, f"{params:,} parámetros entrenables".replace(",", "."), fontsize=15, color=navy)
    ax.text(-0.22, -1.5, "Azul: entrada · Verde: convolución · Naranja: reducción espacial · Violeta: cabeza de salida", fontsize=13, color=navy)
    ax.text(-0.22, -1.84, "Volúmenes esquemáticos; dimensiones exactas bajo cada capa. BCEWithLogitsLoss recibe el logit. Uso educativo, no clínico.", fontsize=12, color=muted)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_sheet(directory: Path, title: str, model: nn.Module, explanation: str,
                configs: list[str], steps: list[dict]) -> None:
    params = trainable_parameter_count(model)
    table = ["| Operación | Salida por corte | Parámetros de la operación | Detalle |",
             "|---|---|---:|---|"]
    for step in steps:
        table.append(f"| {step['label'].replace(chr(10), ' · ')} | `{step['shape']}` | {step['parameters']} | {step.get('detail', '3 fases temporales')} |")
    local_rf = next(step["rf"] for step in reversed(steps) if step["kind"] in {"conv", "pool"})
    bn_note = "Los parámetros de BatchNorm se incluyen en el total, pero no en las filas de convolución." if any(isinstance(m, nn.BatchNorm2d) for m in model.modules()) else "Esta red no usa Batch Normalization."
    content = f"# {title}\n\n![Diagrama de la arquitectura](arquitectura.png)\n\n" + explanation
    content += f"\n\nTotal: **{params:,} parámetros entrenables**.\n\n".replace(",", ".")
    content += "Configuraciones asociadas:\n\n" + "\n".join(f"- [{name}](../../configs/experiments/{name})" for name in configs)
    if not configs:
        content += "- [CNN mínima](../../src/model.py), utilizada por la prueba de memorización."
    content += "\n\n## Recorrido de las capas\n\n" + "\n".join(table)
    content += f"\n\nLas formas omiten el batch `B`. {bn_note}\n"
    content += f"\nAntes del promedio adaptativo, cada posición tiene un campo receptivo teórico de **{local_rf}×{local_rf} píxeles**. El promedio combina posiciones espaciales; no equivale a localizar un tumor.\n"
    content += "\nLas convoluciones tienen stride 1 y padding 1; cada MaxPool 2×2 tiene stride 2. ReLU aporta no linealidad. La sigmoide se aplica para evaluar, después del logit. La media de probabilidades por paciente es la agregación inicial, pendiente de selección con validación.\n"
    content += "\n## Imágenes y regeneración\n\n[SVG vectorial](arquitectura.svg) · [PNG](arquitectura.png). Las etiquetas y dimensiones se obtienen ejecutando la red con una entrada ficticia; no se leen imágenes de pacientes.\n\n```bash\npython -m src.document_architectures\n```\n\nUso educativo y de investigación, sin validez clínica.\n"
    (directory / "README.md").write_text(content, encoding="utf-8")


def main(only: str | None = None) -> None:
    catalog = [
        ("A01_minima", "A01 · CNN mínima de depuración", MinimalCNN(), [],
         "Tres bloques Conv–ReLU–Pool, canales 8/16/32. Resume a 4×4 y conecta 512 valores a un logit. Su objetivo fue memorizar un subconjunto pequeño para comprobar el cableado; no es una estimación de generalización."),
    ]
    specifications = [
        ("E13_phase_differences", "E13 · Realce explícito entre fases", "E13_phase_differences.json", ["E13_phase_differences.json"],
         "La entrada externa sigue siendo PRE/EARLY/LATE en [0,1], con forma [3,256,256]. Dentro del modelo una operación fija conserva PRE y calcula EARLY−PRE y LATE−EARLY: tres canales, cero parámetros nuevos y sin recortar negativos. Las ocho convoluciones, cuatro MaxPool, BatchNorm, GAP y dropout son los de A04/E05; total 294.129 parámetros y campo receptivo local 106×106. Se conserva LR inicial 0,001 y todos los ajustes de E05 Mac. Es un cambio de representación invertible, no información nueva ni una CNN más grande: EARLY=PRE+(EARLY−PRE), LATE=EARLY+(LATE−EARLY). Hipótesis: presentar directamente los cambios temporales podría facilitar optimización. La primera convolución original ya podría aprender restas, por lo que no hay mejora garantizada. Las escalas/correlaciones de entrada cambian y pueden afectar la optimización y BatchNorm; esto forma parte del experimento. E13 completó diez épocas en Mac MPS: mejor época 10, ROC-AUC 0,572480 y AP 0,350860; 421,48 segundos. No supera la referencia E05 Mac (ROC-AUC 0,586895). El mejor checkpoint al final no demuestra convergencia ni que más épocas ayuden; revisar diagnóstico train/validación antes de decidir. Ver [protocolo E13](../REALCE_E13.md)."),
        ("A02_dos_convoluciones", "A02 · Dos convoluciones antes del pooling", "E02_base_normal.json", ["E02_base_normal.json", "E03_base_weighted.json"],
         "Cada bloque hace Conv–BN–ReLU → Conv–BN–ReLU → MaxPool. La hipótesis es combinar características antes de reducir resolución. E02 y E03 comparten esta arquitectura y cambian únicamente la pérdida. En el primer fold y semilla E02 obtuvo ROC-AUC 0,6204 y E03 0,6029; no basta para declarar una arquitectura ganadora."),
        ("A03_una_convolucion", "A03 · Una convolución por bloque", "E04_one_conv_normal.json", ["E04_one_conv_normal.json"],
         "Cada bloque hace Conv–BN–ReLU → MaxPool. E04 elimina la segunda convolución para estudiar si reducir capacidad ayuda a generalizar. Obtuvo ROC-AUC 0,5998 en el primer fold y semilla: fue más rápido, pero no mejoró el resultado observado de E02. La diferencia requiere confirmación con más folds o semillas."),
        ("A04_pool_intermedio", "A04 · Pooling entre las convoluciones", "E05_pool_between_convs.json", ["E05_pool_between_convs.json"],
         "Cada bloque hace Conv–BN–ReLU → MaxPool → Conv–BN–ReLU. Se mueve el único pooling del bloque; no se añade otro al final. Conserva los parámetros y las dimensiones finales de A02, pero la segunda convolución trabaja a menor resolución. Hipótesis: estudiar el efecto de reducir antes de refinar características. E05 completó diez épocas: el checkpoint elegido por ROC-AUC fue el de la época 1, con ROC-AUC por paciente 0,6140 y PR-AUC 0,3910. A umbral 0,5 predijo todas las pacientes como negativas; no hay mejora demostrada de generalización. Cambiar el presupuesto de nueve a diez épocas frente a los baselines limita la comparación: confirmar candidatos con igual presupuesto."),
        ("A05_cinco_bloques", "A05 · Cinco bloques con pooling intermedio", "E06_five_blocks_normal.json", ["E06_five_blocks_normal.json"],
         "Se añade un quinto bloque de 128 canales a A04/E05: Conv–BN–ReLU → MaxPool → Conv–BN–ReLU. Son diez convoluciones y cinco MaxPool, con resolución 256 → 128 → 64 → 32 → 16 → 8. No se amplía a 256 canales para no cambiar profundidad y anchura a la vez. Hipótesis: el bloque extra combina información de una región mayor antes del promedio global. El campo receptivo local aumenta de 106×106 a 218×218; es teórico, no una medida de atención ni una explicación causal. El riesgo es aumentar sobreajuste o perder detalle espacial. Se mantienen todos los ajustes de entrenamiento de E05, incluido learning rate inicial 0,001, BCE normal, seed 42, fold 0 y diez épocas. El scheduler puede reducir el learning rate durante el entrenamiento. E06 completó diez épocas en universidad: mejor época 6, ROC-AUC 0,574647 y AP 0,367224, sin superar el resultado observado de E05; más capas no garantizan alcanzar ROC-AUC 0,7. Comparación y criterio de confirmación en [el protocolo](../PROTOCOLO_COMPARACION.md)."),
        ("E07_dropout_050", "E07 · Cuatro bloques y dropout 0,5", "E07_dropout_050.json", ["E07_dropout_050.json"],
         "Variante de regularización de A04/E05: se cambia únicamente el dropout de 0,3 a 0,5 en el vector de 128 características, después del promedio global. Durante entrenamiento anula aleatoriamente componentes de ese vector; en evaluación está desactivado. No elimina convoluciones, no reduce canales y no añade parámetros. Hipótesis: dificultar la dependencia de características concretas puede reducir el sobreajuste; también podría empeorar por regularización excesiva. Se conserva weight decay 0,0001, learning rate inicial 0,001, BCE normal, seed 42, fold 0 y diez épocas. E07 completó diez épocas en universidad: mejor época 1, ROC-AUC 0,595565 y AP 0,368886; no superó el resultado observado de E05. Ver [comparación E05/E07/E08](../REGULARIZACION.md)."),
        ("E08_weight_decay_001", "E08 · Cuatro bloques y weight decay 0,001", "E08_weight_decay_001.json", ["E08_weight_decay_001.json"],
         "La arquitectura es exactamente A04/E05, con dropout 0,3; el dibujo se repite aquí para que el experimento tenga su ficha completa. Solo se cambia el weight decay de AdamW de 0,0001 a 0,001. No es una capa nueva: modifica la actualización de los pesos durante entrenamiento mediante decaimiento desacoplado. Hipótesis: una regularización mayor de los pesos puede reducir el sobreajuste; su utilidad se mide en validación, no por lograr una menor loss de train. Se mantienen learning rate inicial 0,001, BCE normal, seed 42, fold 0 y diez épocas. E08 no combina este cambio con el dropout 0,5 de E07. E08 completó diez épocas en Mac MPS: mejor época 3, ROC-AUC 0,582661 y AP 0,355411. Frente a E05 Mac, el intervalo pareado global incluye cero; no hay mejora clara. Ver [comparación E05/E07/E08](../REGULARIZACION.md)."),
        ("E09_shared_affine", "E09 · CNN de E05 con aumentos solo en train", "E09_shared_affine.json", ["E09_shared_affine.json"],
         "La CNN es exactamente A04/E05: ocho convoluciones, cuatro MaxPool y dropout 0,3. El único cambio es un aumento aleatorio de entrada durante entrenamiento: rotaciones entre −5° y +5° y traslaciones de hasta 3 % por eje (7,68 píxeles). Se aplica una única rejilla bilineal a PRE/EARLY/LATE, con relleno cero; no se normalizan las fases por separado. No se crean pacientes nuevas ni se guardan PNG aumentados. Validación, test e inferencia no aplican el aumento; mantienen la misma carga determinista en [0,1]. El presupuesto sigue siendo diez épocas, LR inicial 0,001, weight decay 0,0001, BCE normal, fold 0 y seed 42. Hipótesis: aprender patrones menos dependientes de la posición. Puede empeorar si pierde detalle; no se garantiza reducir el sobreajuste ni alcanzar ROC-AUC 0,7. E09 completó diez épocas en universidad: mejor época 1, ROC-AUC 0,610282 y AP 0,375773; no superó el resultado observado de E05. Ver [protocolo y lámina local](../AUMENTOS_E09.md)."),
        ("A06_mitad_canales", "A06 · Misma profundidad, mitad de canales", "E10_half_width.json", ["E10_half_width.json"],
         "E10 mantiene el orden Conv–BN–ReLU → MaxPool → Conv–BN–ReLU de E05, pero reduce los canales de 16/32/64/128 a 8/16/32/64. Conserva ocho convoluciones y cuatro MaxPool, las resoluciones espaciales y el campo receptivo local de 106×106. La cabeza recibe 64 valores tras GAP y devuelve un logit. Reduce capacidad sin eliminar etapas de procesamiento: no es E04, que quitaba convoluciones. Hipótesis: menos parámetros pueden limitar la adaptación a peculiaridades de train; también podrían causar infraajuste. El diagnóstico de E05 muestra AUC train/validación 0,587/0,614 en época 1 y 0,870/0,580 en época 10. E10 conserva diez épocas, LR inicial 0,001, weight decay 0,0001, dropout 0,3, BCE normal, fold 0 y seed 42, sin aumentos. E10 completó diez épocas en Mac MPS: mejor época 6, ROC-AUC 0,590625 y AP 0,379061. E05 Mac obtuvo 0,586895: delta +0,003730 con IC pareado global que incluye cero, sin ventaja clara. Ver [protocolo y ejecución en Mac](../ANCHURA_E10.md)."),
        ("E11_lr_0003", "E11 · Learning rate inicial 0,0003", "E11_lr_0003.json", ["E11_lr_0003.json"],
         "La arquitectura es A04/E05: cuatro bloques Conv–BN–ReLU → MaxPool → Conv–BN–ReLU, canales 16/32/64/128 y 294.129 parámetros. El único ajuste modificado respecto a E05 Mac es el learning rate inicial de AdamW: 0,001 → 0,0003. No es una capa ni un cambio en los kernels. Se mantienen diez épocas y el scheduler, que puede reducir la tasa según AUC de validación. Hipótesis: actualizaciones menores podrían estabilizar aprendizaje; también podrían necesitar más épocas. E11 completó diez épocas en Mac MPS: mejor época 2, ROC-AUC 0,550202 y AP 0,344140; 399,15 segundos. No superó el ROC-AUC de E05 Mac (0,586895) y mantuvo señales de sobreajuste. Ver [protocolo de learning rate](../LEARNING_RATE.md)."),
        ("E12_lr_0001", "E12 · Learning rate inicial 0,0001", "E12_lr_0001.json", ["E12_lr_0001.json"],
         "La arquitectura es A04/E05, idéntica a E11. El único ajuste modificado respecto a E05 Mac es el learning rate inicial de AdamW: 0,001 → 0,0001. Se mantienen datos completos, fold 0, seed 42, batch 16, diez épocas, BCE normal, dropout 0,3, weight decay 0,0001 y scheduler. No hay aumentos ni pesos preentrenados. La comparación inicial usa igual presupuesto, pero no supone convergencia idéntica; decidir una prueba más larga por separado si aún mejora al terminar. E12 completó diez épocas en Mac MPS: mejor época 3, ROC-AUC 0,557964 y AP 0,386688; 481,87 segundos. No superó el ROC-AUC de E05 Mac (0,586895). La AP algo mayor no cambia nuestro criterio de selección; el sobreajuste persiste. Ver [protocolo de learning rate](../LEARNING_RATE.md)."),
    ]
    for folder, title, config, configs, explanation in specifications:
        loaded = load_experiment_config(ROOT / "configs/experiments" / config)
        catalog.append((folder, title, BreastPCRNet(loaded.model), configs, explanation))
    if only is not None and only not in {entry[0] for entry in catalog}:
        raise ValueError(f"Arquitectura desconocida: {only}")
    for folder, title, model, configs, explanation in catalog:
        if only is not None and folder != only:
            continue
        directory = ROOT / "experimentos" / folder
        directory.mkdir(parents=True, exist_ok=True)
        steps = trace(model)
        draw(steps, title, "La resolución se reduce y los canales aumentan; cada operación se muestra en su orden real.",
             trainable_parameter_count(model), directory / "arquitectura")
        write_sheet(directory, title, model, explanation, configs, steps)
        print(f"Documentado: {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Regenerar solo una carpeta, por ejemplo A05_cinco_bloques")
    main(parser.parse_args().only)
