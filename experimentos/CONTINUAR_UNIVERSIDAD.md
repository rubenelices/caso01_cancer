# Retomar las pruebas en la GPU universitaria

Este paquete incluye los experimentos preparados hasta E13, el benchmark
corregido, evaluación interna por paciente, cohortes, diagnóstico y notebook
con resultados agregados. No incluye web, imágenes ni checkpoints. Los pesos
entrenados en el Mac no aparecen al hacer pull: permanecen locales.

La copia exacta del paquete seleccionado pasó 96 tests, el smoke integral E13
de dos épocas en CPU (4,20 s) y un benchmark técnico corto CPU. Estas pruebas
no miden calidad científica ni sustituyen la comprobación ROCm universitaria.

## 1. Actualizar sin perder cambios locales

Desde el repositorio de la universidad, revisar primero:

```bash
git status --short
```

Si no hay cambios locales que conservar, actualizar:

```bash
git pull --ff-only
```

Si Git avisa que sobrescribiría archivos, detenerse y revisar los archivos
afectados. No usar reset ni descartar cambios automáticamente.

Activar el entorno que ya tiene PyTorch ROCm:

```bash
conda activate cancer
```

No reinstalar PyTorch sin revisar el entorno: podría sustituir la instalación
ROCm que ya funciona. Los datos siguen en `breastdcedl/`, locales e ignorados.

## 2. Comprobar la GPU

```bash
python -c 'import torch; print("GPU disponible:", torch.cuda.is_available()); print("HIP:", torch.version.hip); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ninguna")'
```

PyTorch ROCm usa el nombre de dispositivo `cuda`, aunque la tarjeta sea AMD.
Conservar la configuración ROCm que ya permitió entrenar; no modificar
variables del sistema a ciegas. Las advertencias MIOpen conocidas no demuestran
por sí solas que el entrenamiento falle, pero revisar errores y tiempos reales.

## 3. Benchmark corregido pendiente

```bash
python -m src.benchmark_training --config configs/experiments/E05_pool_between_convs.json --batch-size 16 --steps 10 --device cuda
```

Separa lectura, transferencia y cálculo, con calentamiento y precisión según
configuración. Usa un modelo nuevo temporal; no modifica checkpoints entrenados
ni usa el test para comparar calidad. La estimación no incluye todo el coste
de validación/escritura. Enviar su salida antes de decidir pruebas largas.

## 4. Prueba pequeña de compatibilidad, si hace falta

```bash
python -m src.train --config configs/experiments/SMOKE_phase_differences.json --device cuda
```

Dos épocas con 8 pacientes de train y 4 de validación; resultados no científicos.
Si la carpeta ya existe, conservarla y consultar antes de sobrescribir.
`src.smoke_training` fuerza CPU; por eso esta comprobación específica usa el
runner normal con configuración pequeña y `--device cuda` explícito.

## 5. Elegir la siguiente prueba

E10, E11, E12 y E13 ya se probaron en Mac; sus cifras están en el notebook.
E13 obtuvo train AUC 0,788197 y validación 0,572480, y recalibrar BatchNorm
no ayudó. No repetirlos como pendientes ni alargar automáticamente por usar GPU.
Cambiar MPS/float32 a ROCm/AMP limita la comparación con resultados anteriores;
una nueva ablación necesita referencia comparable en la universidad e identidades
separadas si se repite una ejecución existente.

La propuesta siguiente es regularización dentro de bloques convolucionales;
todavía no hay E14 implementado. Acordar configuración e hipótesis antes de
entrenar, cambiando una variable principal. Mantener test cerrado y confirmar
candidatos con otras semillas/folds. No hay modelo definitivo ni garantía de 0,7.

Uso educativo y de investigación, sin validez clínica.
