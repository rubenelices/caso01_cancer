# E10: menos anchura sin perder profundidad

## Evidencia que motiva la prueba

Diagnóstico de E05 compartido por el estudiante desde la universidad:

| Checkpoint | AUC train | AUC validación | AUC validación con copia BN recalculada |
|---|---:|---:|---:|
| Época 1 | 0,586984 | 0,614012 | 0,617339 |
| Época 10 | 0,869524 | 0,579738 | 0,590121 |

Es una brecha de generalización compatible con sobreajuste. La intervención BN
no la resuelve; sus intervalos pareados de mejora incluyen cero. No identifica
qué señales o sesgos aprende la red. Los resultados son por paciente, fold 0,
seed 42, usando solo train público y validación interna; no se consulta test.

## Hipótesis y única variable principal

Reducir a la mitad los canales puede limitar la capacidad de adaptar filtros a
peculiaridades del entrenamiento. Es una hipótesis, no una mejora demostrada.
También puede perder señal útil y causar infraajuste.

| Decisión | E05 | E10 |
|---|---|---|
| Canales por bloque | 16 / 32 / 64 / 128 | 8 / 16 / 32 / 64 |
| Parámetros entrenables | 294.129 | 73.913 |
| Convoluciones / MaxPool | 8 / 4 | 8 / 4 |
| Orden de cada bloque | Conv-BN-ReLU, Pool, Conv-BN-ReLU | Igual |
| Resolución tras los bloques | 128, 64, 32, 16 | Igual |
| Campo receptivo local teórico final | 106 × 106 | Igual |
| Vector tras promedio global | 128 | 64 |

La reducción de parámetros es aproximadamente 74,9 %, pero no garantiza una
reducción equivalente de tiempo. E04 eliminaba convoluciones: este experimento
reduce anchura manteniendo profundidad y contexto espacial.

No se cambian el preprocesamiento PRE/EARLY/LATE en [0,1], BN, dropout 0,3,
AdamW, LR inicial 0,001, weight decay 0,0001, BCE normal, batch 16, seed 42,
fold 0, diez épocas, scheduler ni criterio de mejor checkpoint por AUC paciente.
No hay aumentos. E08 sigue siendo una prueba independiente de regularización.

[Ficha, dibujo y dimensiones capa a capa](A06_mitad_canales/README.md).

## Mac: antes del entrenamiento completo

No es necesario traer el checkpoint universitario: E10 empieza con pesos nuevos.
Sí hacen falta los datos locales y el entorno Python con las dependencias.

Comprobar la aceleración de Apple disponible en el entorno elegido:

```bash
python -c "import torch; print('MPS disponible:', torch.backends.mps.is_available())"
```

Si sale `True`, `--device auto` elige MPS si no hay CUDA. Si sale `False`, elige
CPU. No usar `--device cuda` en un Mac sin CUDA. El runner solo activa precisión
mixta en CUDA: MPS/CPU usan float32. No confundir Mac MPS con ROCm universitario.

Prueba técnica pequeña de dos épocas (ejecuta en CPU y no mide generalización):

```bash
python -m src.smoke_training --config configs/experiments/SMOKE_half_width.json
```

Antes de invertir tiempo, medir el dispositivo real con el benchmark local:

```bash
python -m src.benchmark_training --config configs/experiments/E10_half_width.json --steps 10 --device auto --output reports/benchmarks/E10_half_width_local.json
```

El benchmark corregido todavía tiene cambios locales independientes del commit
del diagnóstico. Su duración es una estimación, no una garantía: validación,
gráficos, escrituras, calentamiento y carga del ordenador añaden variabilidad.
No interpretar sus desgloses asíncronos de MPS como latencias exactas; confirmar
el tiempo con una época real si se decide entrenar allí.

Entrenamiento completo cuando se acepte el tiempo estimado:

```bash
python -m src.train --config configs/experiments/E10_half_width.json --device auto
```

No se ha lanzado automáticamente. No usar `--overwrite`: las rutas de E10 son
propias y los resultados anteriores se conservan. Una ejecución incompleta no
se reanuda automáticamente; antes de repetirla hay que decidir cómo conservarla.

## Verificaciones locales

Las 68 pruebas del repositorio y el smoke integral de dos épocas han pasado.
El smoke usó ocho pacientes de train y cuatro de validación en CPU y completó
el entrenamiento en 2,54 segundos; ese tiempo y sus métricas no representan
un entrenamiento completo ni prueban una mejora científica. El dibujo PNG/SVG
se ha revisado visualmente. MPS no está disponible en este entorno de ejecución;
conviene comprobarlo desde el entorno de terminal del estudiante.

Benchmark CPU local con batch 16, cero workers, tres pasos de calentamiento,
diez medidos y OMP/MKL limitados a dos hilos: 67,27 cortes/s, aproximadamente
2,17 minutos de entrenamiento por época y 21,7 minutos para diez épocas,
sin incluir validación ni escritura. Es una medida corta y específica de este
entorno, no una promesa para cualquier Mac. Informe regenerable e ignorado en
`reports/benchmarks/E10_half_width_cpu.json`.

## Qué vamos a comparar

Estado actualizado: el estudiante ya completó E10 en Mac MPS (198,61 s), mejor
época 6, ROC-AUC paciente 0,590625 y AP 0,379061. E05 Mac completo obtuvo
0,586895 y 0,374219. Delta AUC +0,003730, IC pareado exploratorio
[−0,076872; 0,077167]: no hay ventaja global clara. La reducción de anchura
no elimina el patrón de pérdidas compatible con sobreajuste. El registro,
notebook y análisis completo por cohorte están actualizados; no se entrena otra
variante por ahora.

Guardar historia, configuración, entorno, mejor/último checkpoint y curvas en
las rutas de E10. Comparar con E05 el mejor AUC por paciente bajo el mismo
criterio y presupuesto, PR-AUC, evolución de las pérdidas y brecha train/val.
Una menor loss de train o una menor brecha no basta: podría ser infraajuste.

Si resulta prometedor, repetir con semillas y folds por paciente antes de
afirmar una mejora: ya hemos consultado repetidamente este fold de desarrollo.
Si vuelve a fallar, investigar señal de entrada y sesgo de cohorte antes de
continuar aumentando o reduciendo capas sin hipótesis.

Uso educativo y de investigación, sin validez clínica.
