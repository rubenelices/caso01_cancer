# E11/E12: actualizaciones más pequeñas

Referencia: E05 Mac, AUC paciente 0,586895, AP 0,374219, mejor época 1.
El learning rate inicial 0,001 del profesor es una recomendación, no un requisito
fijo, según aclaración del estudiante. Probamos valores menores sin cambiar capas.

| Ejecución | LR inicial | Épocas | Arquitectura |
|---|---:|---:|---|
| E05 Mac ya entrenado | 0,001 | 10 | A04, 294.129 parámetros |
| E11 | 0,0003 | 10 | La misma |
| E12 | 0,0001 | 10 | La misma |

Hipótesis: actualizaciones más pequeñas pueden estabilizar el aprendizaje y
mejorar validación. No es una regularización garantizada ni prueba de que el
LR explique el sobreajuste. Cambiar LR también afecta el decaimiento efectivo
de AdamW aunque conservemos su coeficiente weight decay: es un matiz de la
comparación, no una segunda modificación deliberada.

## Control experimental

Mantener datos completos, fold 0, seed 42, batch 16, BCE normal, dropout 0,3,
weight decay 0,0001, sin aumentos y la misma inicialización, arquitectura y
agregación mean. Usar el mismo entorno Mac MPS de E05/E08/E10. Las carpetas
de informes y checkpoints son nuevas; no cargar pesos anteriores ni usar
`--overwrite`. El test sigue cerrado.

El scheduler sigue siendo ReduceLROnPlateau sobre AUC paciente, patience 3,
factor 0,5. Son tasas **iniciales**, no constantes: consultar learning_rate
en history.csv. El comportamiento del scheduler puede diferir por la trayectoria
de cada entrenamiento; comparamos las políticas completas con distinto LR inicial.

Las fichas [E11](E11_lr_0003/README.md) y [E12](E12_lr_0001/README.md) repiten
el dibujo de A04 para facilitar la consulta; no hay una arquitectura nueva.

## Ejecutar una después de otra en el Mac

```bash
python -m src.train --config configs/experiments/E11_lr_0003.json --device auto
```

Cuando termine E11:

```bash
python -m src.train --config configs/experiments/E12_lr_0001.json --device auto
```

No se han lanzado entrenamientos completos automáticamente. Una referencia de
coste es E05 Mac, que tardó unos siete minutos en total: cambiar LR no reduce
directamente las operaciones por época. Temperatura/carga/dispositivo pueden
cambiar el tiempo; ejecutar ambas simultáneamente dificulta esa comparación.

Smoke tests independientes de dos épocas, CPU y subconjunto por paciente:

```bash
python -m src.smoke_training --config configs/experiments/SMOKE_lr_0003.json
python -m src.smoke_training --config configs/experiments/SMOKE_lr_0001.json
```

Sus métricas solo comprueban funcionamiento, no calidad científica.

Verificación local: 89 tests superados y ambos smoke tests completos en CPU,
con ocho pacientes de train y cuatro de validación, sin evaluar test. Cada smoke
entrenó dos épocas en aproximadamente cuatro segundos. El notebook ejecutado
registra los resultados completos de E11/E12; las figuras se comprobaron contra las capas.

## Qué decidir después

Guardar todas las épocas; elegir el checkpoint únicamente por AUC paciente.
Comparar AUC/AP, curvas, sensibilidad/especificidad a umbral 0,5 y evolución
del learning rate. No cambiar el criterio a posteriori buscando la cifra favorable.
Las entradas E11/E12 del notebook ya incorporan los resultados locales completos.

## Resultados observados en Mac MPS

| Ejecución | Mejor época por ROC-AUC | ROC-AUC paciente | AP paciente | Tiempo total |
|---|---:|---:|---:|---:|
| E05 Mac | 1 | 0,586895 | 0,374219 | 414,01 s |
| E11 | 2 | 0,550202 | 0,344140 | 399,15 s |
| E12 | 3 | 0,557964 | 0,386688 | 481,87 s |

Las tres ejecuciones usan diez épocas, fold 0, seed 42 y 219 pacientes de
validación. E11/E12: MPS, Python 3.13.5, PyTorch 2.11.0. Fuente: summary.json
e history.csv locales contrastados con las salidas del estudiante. Test no evaluado.

A umbral 0,5, E11 detecta 1 de las 64 positivas (2 falsos positivos), y E12
8 de 64 (7 falsos positivos). No ajustar el umbral para fingir una mejora del
ROC-AUC: este no depende de un único umbral. E12 tiene AP algo mayor que E05,
pero no cambiamos a posteriori el criterio principal de selección.

La loss train/validación de la última época es 0,1557/1,0753 en E11 y
0,2900/0,9228 en E12. Los mejores checkpoints fueron tempranos y las curvas
mantienen señales de sobreajuste. Estas dos ejecuciones no apoyan bajar LR
como solución suficiente; no demuestran que toda tasa menor sea inútil.
No se prolongan ni sobrescriben automáticamente. Las instrucciones de arriba
quedan como referencia reproducible, no como entrenamientos pendientes.

Diez épocas fijan el primer presupuesto, no aseguran que cada LR haya convergido.
Si una variante todavía mejora al terminar, acordar una ejecución más larga con
identidad y rutas nuevas, manteniendo las otras variables. No descartar una tasa
menor solo porque aprenda más despacio ni alargar todas por defecto.
Confirmar propuestas prometedoras con más semillas/folds: este fold se ha
consultado repetidamente. No se garantiza AUC 0,7.

Uso educativo y de investigación, sin validez clínica.
