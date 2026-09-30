# E09: aumentos geométricos compartidos

## Cambio respecto a E05

Misma CNN (294.129 parámetros), pérdida normal, dropout 0,3, weight decay
0,0001, batch 16, fold 0, seed 42, LR inicial 0,001 y diez épocas. Solo se
activa el aumento de datos de train. No se combinan los cambios de E07/E08.

Cada vez que se carga un corte de train se muestrean un giro uniforme de
−5° a +5° y desplazamientos uniformes de −7,68 a +7,68 píxeles por eje.
Los tres canales se transforman con la misma rejilla. No son tres giros
independientes. Se usa interpolación bilineal y relleno cero donde no hay
imagen de origen. No hay flips, zoom, recortes aleatorios, ruido ni aumentos RGB.

```text
TRAIN: PRE/EARLY/LATE → carga compartida [0,1] → afín compartida → CNN
VAL / TEST / WEB: PRE/EARLY/LATE → misma carga [0,1] → CNN (eval)
```

La rotación y la interpolación modifican algunos píxeles; pueden perder detalle
o contenido de los bordes. Son aumentos moderados propuestos, no parámetros
óptimos demostrados. Su utilidad se decidirá por validación por paciente.
Siguen siendo las mismas 878 pacientes de train, no más pacientes independientes.
Los cortes aumentados conservan patient_id, sample_id y etiqueta. Los PNG de
origen permanecen intactos. Las variaciones se generan al vuelo, no se acumulan
en el disco ni se cuentan como nuevas muestras del dataset.

## Reproducibilidad y compatibilidad

- Parámetros en `data.augmentation`, registrados en configuración y checkpoint.
- Configuraciones antiguas: aumento desactivado por defecto. No hay que editar E05.
- Las semillas de PyTorch y los workers controlan el muestreo. La reproducibilidad
  se comprueba con el mismo entorno y número de workers; no se promete igualdad
  bit a bit entre dispositivos o versiones diferentes.
- Implementación propia mediante `affine_grid` y `grid_sample`, sin añadir
  torchvision ni modelos importados. Ambas llamadas usan `align_corners=False`.
  [Rejilla afín de PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.affine_grid.html)
  y [muestreo bilineal](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.grid_sample.html).

## Inspección visual local

```bash
python -m src.preview_augmentation
```

Genera `reports/data_augmentation/E09_shared_affine/original_vs_augmented.png`
y `preview.json`, con parámetros e identificadores. Solo lee un corte de train
de cada clase, seis PNG originales en total; no mira imágenes de test.
La lámina muestra original y dos variaciones, con la misma escala [0,1] para
todas las imágenes. Es una inspección de dos ejemplos, no una garantía para
todo el dataset. Estos archivos incluyen imágenes docentes y permanecen locales
y fuera de Git. El diagrama de la CNN en esta carpeta sí es documentación propia.

## Pruebas y entrenamiento

Las pruebas verifican alineación de fases, identidad sin transformaciones,
dirección de la traslación, rango/dtype, parámetros acotados, semilla,
conservación de etiquetas y ausencia de aumentos en validación/test sintético.
La prueba integral usa solo ocho pacientes de train y cuatro de validación:

```bash
python -m src.smoke_training --config configs/experiments/SMOKE_shared_affine.json
```

Sus métricas no estiman calidad y sus rutas son independientes de E09. Una
vez sincronizado el código, el entrenamiento completo se ejecuta en la GPU:

```bash
python -m src.train --config configs/experiments/E09_shared_affine.json --device cuda
```

Resultados en `reports/experiments/E09_shared_affine/fold_0_seed_42/` y pesos
en `checkpoints/E09_shared_affine/fold_0_seed_42/`. No usar `--overwrite` para
el ensayo científico; conservar mejor/último checkpoint, historial y curvas.

Comparar con E05 por ROC-AUC, PR-AUC, evolución train/validación y matriz de
confusión por paciente. El train loss aumentado no es directamente equivalente
al de imágenes originales: el problema de entrenamiento ahora es más difícil.
Confirmar candidatos con otras semillas/folds; test permanece cerrado.
Sigue pendiente diagnosticar BatchNorm y probabilidades en checkpoints de la
universidad. Los aumentos no sustituyen esa revisión.

No se ha medido el tiempo con aumentos en la RX 6700 XT; hay un coste adicional
en carga/transformación. El benchmark anterior no representa E09 si no aplica
estos aumentos. No se lanza aquí el entrenamiento completo en CPU.

Uso educativo y de investigación, sin validez clínica.
