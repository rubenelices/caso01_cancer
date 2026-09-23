# Como interpretar el entrenamiento

No existe una cifra garantizada que la CNN deba alcanzar antes de entrenarla.
La documentacion del caso no fija un ROC-AUC objetivo. Si podemos definir
referencias que permiten detectar rapidamente un entrenamiento roto.

## Que significa «si en la epoca 5 no baja, esta mal»

Normalmente se refiere a la **loss de entrenamiento**. La loss cuantifica el
error que el optimizador intenta minimizar mediante backpropagation.

- Si entre las epocas 1 y 5 la loss de train muestra una tendencia descendente,
  la red esta recibiendo gradientes y aprendiendo algo de las muestras.
- No necesita bajar en todas las epocas: puede oscilar.
- Si permanece practicamente plana, crece o es `NaN`, se revisan datos,
  etiquetas, learning rate, loss, gradientes y preprocesamiento.
- La loss de validacion puede fluctuar. Si train baja y validacion sube de forma
  sostenida, la red esta sobreajustando.

La epoca 5 es una regla de diagnostico temprano, no una demostracion de que el
modelo generalice bien.

## Referencias del problema

En train, aproximadamente el 70,6 % de los casos son `pCR=0`. Un predictor que
siempre diga cero consigue:

| Metrica | Referencia trivial |
|---|---:|
| Accuracy | 70,6 % |
| Balanced accuracy | 50 % |
| Sensibilidad pCR=1 | 0 % |
| Especificidad pCR=0 | 100 % |
| ROC-AUC | 0,50 |
| PR-AUC | prevalencia positiva, aproximadamente 0,29 |

Por eso una accuracy cercana al 70 % puede esconder un modelo inutil. Hay que
mirar siempre la matriz de confusion y comprobar que predice ambas clases.

Con logits iniciales cercanos a cero, la BCE normal suele comenzar alrededor de
`0,69`, aunque la inicializacion y los primeros batches pueden cambiarla. La BCE
ponderada tiene otra escala —alrededor de `0,98` para un predictor neutro con
`pos_weight≈2,4`—, por lo que E02 y E03 no se comparan usando solo el valor
absoluto de su loss.

## Que queremos observar en E02

1. La loss de train baja claramente durante las primeras epocas.
2. La loss de validacion baja inicialmente o, al menos, no diverge enseguida.
3. ROC-AUC por paciente supera 0,50 de forma estable, no en una epoca aislada.
4. PR-AUC supera aproximadamente 0,29.
5. Sensibilidad y especificidad son ambas mayores que cero.
6. Balanced accuracy supera 0,50.
7. El mejor checkpoint no tiene una matriz de confusion colapsada en una clase.

Estas son señales de aprendizaje, no umbrales definitivos para aprobar el
modelo. La estabilidad entre los cinco folds sera mas importante que una cifra
afortunada en un solo fold.

## Archivos que apareceran al terminar

Dentro de `reports/experiments/<experimento>/fold_0_seed_42/`:

- `training_curves.png`: panel visual de loss, AUC, sensibilidad/especificidad y
  matriz de confusion.
- `training_diagnostics.json`: comprobacion de epoca 5, colapso de clase,
  posible sobreajuste y baselines.
- `history.csv`: todas las metricas de cada epoca.
- `summary.json`: resumen del mejor checkpoint.
- `environment.json`: dispositivo y versiones utilizadas.

Los pesos `best.pt` y `last.pt` se guardan en `checkpoints/`, fuera de Git.
