# E06: ¿ayuda un quinto bloque?

## Hipótesis y control

E06 añade un bloque de dos convoluciones y un MaxPool a E05. No cambia los
primeros cuatro bloques, los canales finales (128), la pérdida ni los ajustes
de entrenamiento. El cambio principal es la profundidad: el bloque extra
incluye tanto procesamiento como reducción espacial. No permite separar el
efecto individual de las dos convoluciones del del quinto pooling.

| Propiedad | E05 | E06 |
|---|---:|---:|
| Convoluciones | 8 | 10 |
| MaxPool 2×2 | 4 | 5 |
| Mapa antes del promedio global | 128×16×16 | 128×8×8 |
| Campo receptivo local teórico | 106×106 | 218×218 |
| Parámetros | 294.129 | 589.553 |
| Épocas máximas | 10 | 10 |
| Learning rate inicial | 0,001 | 0,001 |

Ambos usan BCE normal, AdamW, dropout 0,3, weight decay 0,0001, batch 16,
fold 0, seed 42 y el mismo scheduler. El learning rate puede bajar según la
validación. La unidad de evaluación es la paciente y se agregan probabilidades
con la media. La selección de checkpoint sigue siendo por ROC-AUC por paciente,
no por la métrica que resulte más favorable al terminar.

## Referencia observada, no objetivo garantizado

Salida de E05 facilitada por el estudiante: 878 pacientes de train, 219 de
validación, 10 épocas, mejor checkpoint en época 1, ROC-AUC 0,614012,
PR-AUC 0,390992, sensibilidad 0 y especificidad 1 a umbral 0,5. Tiempo total
205,27 s en la RX 6700 XT. Train loss pasó de 0,6148 a 0,4727 sin mejora
sostenida en validación. El test no fue evaluado.

E06 está pendiente de entrenamiento completo. No se afirma que sea mejor.
El criterio de interés del profesor, ROC-AUC 0,7, se tratará como objetivo
experimental: no se garantiza ni se consigue ajustando el umbral (el ROC-AUC
se calcula con las probabilidades, no con la clase binaria a umbral 0,5).

## Ejecución ordenada

1. Pruebas de formas, parámetros, orden de capas, gradientes y carga de pesos.
2. Smoke test de dos épocas sobre 8 pacientes de train y 4 de validación.
   Sus métricas no estiman generalización. Usa rutas separadas.
3. En la universidad, ejecutar una vez E06 con su configuración de diez épocas.
   Guardar `summary.json`, `history.csv`, curvas y predicciones. No sobrescribir
   resultados anteriores para quedarnos solo con una ejecución favorable.
4. Comparar con E05: mejor ROC-AUC, PR-AUC, evolución train/validación, sensibilidad
   y especificidad, tiempo y memoria. Mayor accuracy no basta; revisar el colapso
   de predicciones a una clase. Una diferencia pequeña no prueba superioridad.
5. Confirmar candidatos prometedores con semillas 42/43/44 y después con cinco
   folds por paciente. Cada ejecución necesita su identidad y rutas propias.
   No seleccionar solo la mejor semilla: reportar dispersión y rendimiento por
   cohorte, e intervalos por paciente. Comparar sobre las mismas pacientes.

Probar muchas redes sobre el mismo fold puede sobreajustar las decisiones a esa
validación. Una cifra superior a 0,7 en este primer fold es un resultado de
desarrollo, no una garantía sobre pacientes nuevas. Reservar test para la
evaluación final cuando todas las decisiones estén cerradas.

## Si añadir profundidad no ayuda

No seguir añadiendo bloques sin diagnóstico. Priorizar la revisión de la
distribución de probabilidades, la estabilidad de BatchNorm entre train y eval,
y la evolución del learning rate. Luego plantear por separado regularización,
aumentos geométricos sincronizados o una representación explícita del realce.
Son hipótesis futuras, no cambios incluidos en E06. Ningún nuevo preprocesamiento
se incorporará solo al entrenamiento: debe compartirlo la inferencia y la web.

## Órdenes

Smoke local (CPU, subconjunto pequeño):

```bash
python -m src.smoke_training --config configs/experiments/SMOKE_five_blocks_end_to_end.json
```

Entrenamiento completo en la GPU universitaria, una vez sincronizado el código:

```bash
python -m src.train --config configs/experiments/E06_five_blocks_normal.json --device cuda
```

No se ha medido todavía el tiempo de E06 en esa GPU. Aunque casi duplica los
parámetros, las capas nuevas trabajan a 16×16 y 8×8; no debe extrapolarse el
tiempo multiplicando por la relación de parámetros. Usar el tiempo real de
las primeras épocas para decidir si continuar.

Uso educativo y de investigación, sin validez clínica.
