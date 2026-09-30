# Catálogo de arquitecturas

Cada carpeta reúne una arquitectura distinta, su explicación y un dibujo del recorrido real de sus capas. El código ejecutable continúa en `src/` y los experimentos de entrenamiento en `configs/experiments/`.

| Arquitectura | Experimentos | Orden dentro del bloque | Parámetros |
|---|---|---|---:|
| [A01 · Mínima](A01_minima/README.md) | Memorización de depuración | Conv → ReLU → Pool | 6.545 |
| [A02 · Dos convoluciones](A02_dos_convoluciones/README.md) | E02 y E03 | Conv → Conv → Pool | 294.129 |
| [A03 · Una convolución](A03_una_convolucion/README.md) | E04 | Conv → Pool | 97.809 |
| [A04 · Pool intermedio](A04_pool_intermedio/README.md) | E05 | Conv → Pool → Conv | 294.129 |
| [A05 · Cinco bloques](A05_cinco_bloques/README.md) | E06 | Conv → Pool → Conv, quinto bloque añadido | 589.553 |

En A02, A03, A04 y A05 cada convolución va seguida de BatchNorm y ReLU. E02 y E03 usan la misma red con pérdidas distintas, así que no duplicamos su dibujo. Las alturas y anchuras de los volúmenes son esquemáticas: las dimensiones exactas aparecen bajo cada capa.

Abre la vista previa del README de cada carpeta en Visual Studio Code. Cada ficha incluye `arquitectura.png` para presentaciones y `arquitectura.svg` para ampliar sin perder calidad.

Para regenerar todas las imágenes y fichas a partir de las redes:

```bash
python -m src.document_architectures
```

Para regenerar únicamente una ficha: `python -m src.document_architectures --only A05_cinco_bloques`.

[Protocolo E05/E06 y siguientes pasos hacia ROC-AUC 0,7](PROTOCOLO_COMPARACION.md).

Toda arquitectura nueva debe incorporarse a este catálogo con hipótesis, orden de operaciones, dimensiones, parámetros, campo receptivo y configuraciones asociadas. Los resultados por paciente se guardan en `reports/`, que permanece fuera de Git.

Uso educativo y de investigación, sin validez clínica.
