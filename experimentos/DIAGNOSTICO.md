# Diagnóstico antes de seguir buscando arquitecturas

Este análisis no es E10 ni un entrenamiento nuevo: estudia un checkpoint
existente. Empieza por el `best.pt` de E05. No optimiza pesos, no ajusta umbral,
no selecciona un modelo automáticamente y no consulta test.

## Qué compara

| Condición | Imágenes | Dropout | Estadísticas BatchNorm |
|---|---|---|---|
| Train original | Train interno sin aumentos | Desactivado | Las del checkpoint |
| Validación original | Validación interna sin aumentos | Desactivado | Las del checkpoint |
| Train copia BN | Mismas pacientes y cortes | Desactivado | Recalculadas solo con train |
| Validación copia BN | Mismas pacientes y cortes | Desactivado | Recalculadas solo con train |

La pérdida por corte se calcula con la BCE del checkpoint; las métricas
principales son por paciente, agregando probabilidades con la misma media.
Se incluyen ROC-AUC, PR-AUC (average precision), sensibilidad, especificidad,
precisión, F1, balanced accuracy, matriz de confusión y Brier por paciente.
No se compara una train loss con dropout/aumentos con una val loss sin ellos.

BatchNorm mantiene estadísticas de entrenamiento que se usan en evaluación.
El experimento hace una copia en memoria, reinicia únicamente esos buffers y
los estima pasando train sin aumentos ni dropout. Solo BatchNorm está en modo
train durante esa pasada, con `momentum=None`; no hay gradientes ni optimizador.
Después toda la copia vuelve a eval. Se conservan los pesos de las convoluciones
y los parámetros aprendidos de BatchNorm. [Documentación oficial de BatchNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html).

La recalibración es una media acumulada de estadísticas por minibatch,
no una estimación exacta de momentos de toda la población. Depende del batch,
el orden y la composición de pacientes. No se usan estadísticas de validación,
ni siquiera sin etiquetas. No se cambia la web ni se guarda un checkpoint nuevo.
Se verifica la igualdad de parámetros y del estado original, y el SHA-256 del
archivo original antes y después.

## Ejecución universitaria

Una vez sincronizado este código, estando en la raíz del repositorio y con
el entorno `cancer` activo:

```bash
python -m src.diagnose_training --checkpoint checkpoints/E05_pool_between_convs/fold_0_seed_42/best.pt --device cuda
```

No hace falta volver a entrenar E05: necesita el checkpoint que ya guardó
aquella ejecución. Solo cargar checkpoints propios y confiables. Si falta,
el programa lo indica y no intenta entrenar ni descargar ningún modelo.
La configuración, arquitectura, fold y semilla se leen del propio checkpoint;
los formatos anteriores sin `data.augmentation` también son compatibles.

Por defecto conserva batch/workers del checkpoint, y usa AMP al evaluar en
CUDA/ROCm si así se entrenó. La recalibración BN se hace en FP32. La sigmoide se
calcula en FP32; pequeñas diferencias respecto al resumen de entrenamiento
pueden deberse a precisión numérica, no necesariamente a otro conjunto de datos.
Para diagnosticar también precisión numérica se puede hacer una ejecución
separada con `--float32 --output-dir reports/diagnostics/E05_float32`.

Es una pasada de train y otra de validación por cada condición, más una pasada
de train para BN: cinco pasadas sin backward. No se ha medido el tiempo en la
RX 6700 XT; debería presupuestarse como inferencia, no como 10 épocas nuevas.
El programa muestra en la terminal cada etapa.

## Qué guardar y compartir

Guarda todo en `reports/diagnostics/<experiment_id>/<sha256_corto>/`:

- `diagnosis.json`: métricas, distribuciones por clase, cambios de BN,
  configuración, entorno, tiempos, comprobaciones y comparaciones.
- `diagnostic_dashboard.png`: ROC train/validación y probabilidades por clase.
  Los histogramas amplían el rango observado para ver separaciones pequeñas.
- `README.md`: informe legible con la figura.
- CSV por corte y por paciente para las cuatro condiciones.

La terminal imprime un resumen con AUC train/validación antes y después de BN.
Compartir ese resumen y la figura, o `diagnosis.json`, para decidir el siguiente
paso. Los CSV, informes y checkpoints quedan fuera de Git. Una carpeta existente
no se sobrescribe; para repetir el diagnóstico, elegir otra con `--output-dir`.

## Cómo interpretar sin saltar a conclusiones

- Un AUC de train alto junto a uno de validación bajo es compatible con falta
  de generalización. No demuestra por sí solo cuál es la causa.
- Si ambos son modestos, investigar representación, optimización y fuentes
  de señal; no concluir automáticamente que faltan más capas.
- Un cambio tras recalibrar BN indica sensibilidad a sus estadísticas, no un
  fallo demostrado ni una mejora asegurada. Confirmar con otras semillas/folds.
- Probabilidades todas menores de 0,5 explican sensibilidad cero a ese umbral,
  pero no implican probabilidades constantes ni ROC-AUC necesariamente 0,5.
- Brier y distribución orientan calibración; bajar un umbral no eleva ROC-AUC.

El intervalo del cambio de AUC usa 500 remuestreos estratificados y pareados
por paciente (mismas pacientes en ambas condiciones), no por corte. Es
exploratorio: no corrige selección del mejor checkpoint ni múltiples ensayos
sobre el mismo fold, y no permite prometer superar 0,7.

El mejor checkpoint puede ser de la primera época. Un AUC de train bajo ahí
no descarta que la última época haya memorizado. Si es necesario, repetir sobre
`last.pt` con otra carpeta para estudiar evolución, sin reemplazar el criterio
de selección ni consultar test. La recalibración BN no es calibración de la
probabilidad, y no modifica la normalización compartida de PRE/EARLY/LATE.

## Verificación local

Se prueba primero con datos sintéticos y con un checkpoint del smoke test
existente, no con una ejecución científica universitaria. Se reconstruyen los
subconjuntos originales de los smoke checkpoints y los informes quedan
marcados como técnicos: no estiman calidad.

```bash
python -m pytest tests/test_diagnose_training.py -q
```

Uso educativo y de investigación, sin validez clínica.
