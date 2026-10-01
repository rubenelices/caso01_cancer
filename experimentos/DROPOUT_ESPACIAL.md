# E14/E15: regularización en los mapas convolucionales

## Por qué esta prueba

E13 en Mac terminó con ROC-AUC train **0,788197** y validación **0,572480**
al evaluar ambos sin aumentos y con `eval()`. Recalibrar BatchNorm no ayudó.
Esto sugiere un problema de generalización; no demuestra qué patrón concreto
está aprendiendo la CNN. Añadir capas no resolvió el problema en E06.

La referencia nueva es **E13_50epochs**, que el estudiante ha dejado corriendo
en la GPU universitaria. E14/E15 conservan su presupuesto y cambian únicamente
`model.spatial_dropout`. No comparamos 50 épocas contra 10 como si fuera una
ablación de regularización pura. No hay resultados de estas variantes todavía.

| Ejecución | Dropout2d por bloque | Dropout final | Épocas / paciencia |
|---|---:|---:|---:|
| E13_50epochs | 0 | 0,3 | 50 / 51 |
| E14_spatial_dropout_010 | 0,1 | 0,3 | 50 / 51 |
| E15_spatial_dropout_020 | 0,2 | 0,3 | 50 / 51 |

Hipótesis: dificultar que el clasificador dependa de mapas particulares podría
mejorar la generalización. Un valor mayor no tiene por qué ser mejor y puede
provocar infraajuste. El número de parámetros no disminuye: **294.129**.

## Qué hace la capa, explicado para defenderla

Dentro de cada bloque: **Conv–BN–ReLU → MaxPool → Conv–BN–ReLU → Dropout2d**.
Las convoluciones son 3×3, padding 1, stride 1; el pool es 2×2, stride 2,
padding 0. Dropout2d no tiene kernel, padding, stride ni pesos aprendibles.

Cada mapa de características se anula con probabilidad `p`, independientemente
por muestra durante entrenamiento. Los mapas supervivientes se escalan por
`1/(1-p)`. No se exige que exactamente el 10 % o 20 % desaparezca en cada lote.
Se anulan mapas **aprendidos**, no canales PRE/EARLY/LATE de las imágenes.
No es aumento de datos ni borrado permanente de filtros. En evaluación es
identidad, como especifica la [documentación oficial de PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout2d.html).

| Bloque | Entrada | Salida antes y después de Dropout2d | Parámetros Dropout2d |
|---|---|---|---:|
| 1 | 3×256×256 | 16×128×128 | 0 |
| 2 | 16×128×128 | 32×64×64 | 0 |
| 3 | 32×64×64 | 64×32×32 | 0 |
| 4 | 64×32×32 | 128×16×16 | 0 |

El campo receptivo local final sigue siendo **106×106**. GAP produce 128 valores,
el dropout final 0,3 los regulariza y la capa lineal produce un logit. La entrada
externa sigue siendo PRE/EARLY/LATE en [0,1]; la representación fija de E13 se
aplica una sola vez dentro del modelo. No cambiamos normalización ni agregación.

## Controles y compatibilidad

- BCE normal, AdamW, LR inicial 0,001, weight decay 0,0001, batch 16.
- Fold 0 agrupado por paciente, seed 42, media de probabilidades, umbral 0,5.
- Mismo scheduler; paciencia 51 impide early stopping durante las 50 épocas.
- Misma inicialización y claves de pesos; configuraciones antiguas usan `p=0`
  e identidad, sin consumir números aleatorios adicionales.
- La configuración se guarda en el checkpoint y permite reconstruir la capa.
- Con dropout activo la secuencia aleatoria de entrenamiento cambia; igual
  semilla no implica trayectorias idénticas. Tampoco garantiza determinismo ROCm.
- La salida enmascarada puede influir en BatchNorm del siguiente bloque durante
  train. No lo presentamos como una solución demostrada al problema de BN.
- Los pesos, informes por paciente y dataset siguen locales e ignorados.

## Ejecución ordenada, cuando termine el entrenamiento actual

Primero revisar E13_50epochs completo y conservar sus resultados. Después,
probar **E14**; E15 queda como contraste de intensidad, no como ganador supuesto.
No lanzar entrenamientos simultáneos ni usar `--overwrite` sobre ejecuciones previas.

```bash
python -m src.train --config configs/experiments/E14_spatial_dropout_010.json --device cuda
```

En una ejecución posterior independiente:

```bash
python -m src.train --config configs/experiments/E15_spatial_dropout_020.json --device cuda
```

En Mac usar `--device mps`, con nuevas identidades/rutas si ya existen resultados.
Para comparar directamente, repetir la referencia con la misma plataforma y
precisión; no mezclar MPS/float32 y ROCm/AMP como si fueran condiciones idénticas.

Pruebas técnicas separadas (CPU, dos épocas, 8/4 pacientes train/validación):

```bash
python -m src.smoke_training --config configs/experiments/SMOKE_spatial_dropout_010.json
python -m src.smoke_training --config configs/experiments/SMOKE_spatial_dropout_020.json
```

`src.smoke_training` permite sobrescribir solo su propia ejecución técnica
pequeña; nunca pasarle la configuración científica. Un AUC alto del smoke no
prueba que el modelo generalice.

## Cómo decidir

Comparar el **mejor checkpoint por ROC-AUC por paciente** de cada ejecución,
curvas de validación, PR-AUC, sensibilidad/especificidad y calibración; no elegir
la última época por defecto ni buscar otra época solo porque su AP sea mayor.
Para la brecha train/validación usar el diagnóstico con `eval()` y sin aumentos.
Una menor loss de train o un menor gap por empeorar train no bastan.

Seleccionar como máximo un candidato provisional y confirmarlo frente a la
referencia en otras semillas/folds con el mismo protocolo antes de decidir.
Los intervalos sobre el fold ya explorado no corrigen todos los ensayos ni la
selección de checkpoints. Test sigue cerrado. No se promete alcanzar AUC 0,7.

Fichas y diagramas verificados: [E14](E14_spatial_dropout_010/README.md) y
[E15](E15_spatial_dropout_020/README.md).

Uso educativo y de investigación, sin validez clínica.
