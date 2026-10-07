# Repetir E19, E48 y E51 en la GPU de la universidad

Tres entrenamientos desde cero, uno después de otro. Es una comprobación en
otro dispositivo, no una garantía de mejorar AUC ni tres arquitecturas nuevas.
Entrada PRE/EARLY/LATE, objetivo pCR. Resultados educativos, no clínicos.

Se conserva exactamente el protocolo Mac: CNN propia294.129parámetros,
fold0/seed42,878/219pacientes, batch16, LR/WD0,001, BCE normal,
máximo50épocas/paciencia10, float32 SIN precisión mixta, workers0/pinningfalse.
No aumentar batch/workers ni activar AMP en esta comparación inicial.
E48 activa EMA0,99; E51 suaviza objetivos train0,05/0,95. No se combinan.
Test cerrado, sin transfer learning ni pesos iniciales externos.

Las tres redes comparten cuatro bloques de dos convoluciones3×3 con BN/ReLU,
MaxPool2×2 entre ambas, canales16→32→64→128, GAP, densa64/dropout0,3 y
un logit. Dentro del modelo: PRE, EARLY−PRE, LATE−EARLY; entrada externa
[3,256,256]. Los canales son fases, no RGB. La sigmoide final da una
probabilidad por corte; evaluación principal agrega por paciente con la media.

## Preparación: cada bloque es un comando independiente

Desde el repositorio de la universidad, con datos locales en `breastdcedl/`:

```bash
conda activate cancer
```

```bash
git status --short
```

Si hay cambios locales, no hacer reset/stash/restore a ciegas: consultarlos antes
de actualizar. Con el árbol limpio:

```bash
git pull --ff-only
```

```bash
python -c "import torch; print('PyTorch:',torch.__version__); print('HIP:',torch.version.hip); print('GPU disponible:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DISPONIBLE')"
```

En PyTorch ROCm se usa `cuda` también con AMD. No reinstalar torch desde pip:
conservar el entorno ROCm que ya funciona. Si la GPU no está disponible, parar.
Si torch no detecta GPU o el smoke falla, no lanzar el entrenamiento completo.

## 1. E19: referencia

```bash
python -m src.train --config configs/gpu/SMOKE_E19_reference_cuda.json --device cuda
```

Solo si termina correctamente:

```bash
python -m src.train --config configs/gpu/E19_reference_cuda.json --device cuda
```

## 2. E48: EMA

```bash
python -m src.train --config configs/gpu/SMOKE_E48_ema_cuda.json --device cuda
```

Solo si termina correctamente:

```bash
python -m src.train --config configs/gpu/E48_ema_cuda.json --device cuda
```

## 3. E51: objetivos suaves

```bash
python -m src.train --config configs/gpu/SMOKE_E51_label_smoothing_cuda.json --device cuda
```

Solo si termina correctamente:

```bash
python -m src.train --config configs/gpu/E51_label_smoothing_cuda.json --device cuda
```

No usar overwrite ni ejecutar a la vez. Paciencia10 significa diez épocas sin
mejorar AUC, no una parada obligatoria en10. Si una ruta ya está completa,
conservarla y revisar antes de repetir. No elegir la mejor entre Mac/GPU como
confirmación estadística: mismo fold/pacientes, aritmética y trayectoria distintas.
Registrar también versiones: si PyTorch/ROCm difieren del Mac, esta repetición
compara entornos completos, no aísla causalmente solo el hardware.

## Dónde queda todo

Cada run guarda historial, resumen, curvas y predicciones en
`reports/gpu/EXPERIMENTO/fold_0_seed_42/`; best/last en
`checkpoints/EXPERIMENTO/fold_0_seed_42/`. Smokes en rutas propias.
No hay que copiar toda la terminal: enviar `summary.json` e `history.csv` basta
para la primera revisión. Los archivos quedan locales e ignorados por Git;
las predicciones individuales no se publican ni se hacen push como resultados.
El Mac no recibe automáticamente los archivos de otro ordenador por una ruta:
habrá que adjuntarlos o transferirlos explícitamente.

Código/configs/tests verificados localmente. No se ha validado la compatibilidad
actual en RX6700XT: eso lo comprueban los tres smokes de la universidad.
