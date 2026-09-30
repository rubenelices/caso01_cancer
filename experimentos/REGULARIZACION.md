# E07 y E08: regularización sobre E05

Son dos experimentos independientes, no modificaciones del experimento E05.
Cada uno conserva su configuración, identidad y rutas de resultados/pesos.

| Ajuste | E05 (referencia) | E07 | E08 |
|---|---:|---:|---:|
| Dropout después del promedio global | 0,3 | **0,5** | 0,3 |
| Weight decay de AdamW | 0,0001 | 0,0001 | **0,001** |
| Learning rate inicial | 0,001 | 0,001 | 0,001 |
| Épocas máximas | 10 | 10 | 10 |
| Parámetros | 294.129 | 294.129 | 294.129 |

En ambos: cuatro bloques Conv–BN–ReLU → Pool → Conv–BN–ReLU, canales
16/32/64/128, entrada PRE/EARLY/LATE en [0,1], BCE normal, batch 16, fold 0,
seed 42, misma media de probabilidades por paciente y mismo scheduler.
El learning rate inicial respeta 10⁻³, pero el scheduler puede reducirlo.

E07 estudia si descartar más componentes del vector de características durante
entrenamiento evita dependencia excesiva de algunos de ellos. E08 estudia un
decaimiento mayor de los pesos de AdamW. Ningún cambio garantiza mejorar;
incluso podría llevar a infraajuste. No se espera necesariamente una loss de
entrenamiento menor: interesa la generalización por paciente.

## Referencias y estado

| Experimento | Mejor ROC-AUC por paciente | PR-AUC del mismo checkpoint | Mejor época | Estado |
|---|---:|---:|---:|---|
| E05 | 0,614012 | 0,390992 | 1 | Ejecutado, fold 0 / seed 42 |
| E06 | 0,574647 | 0,367224 | 6 | Ejecutado, fold 0 / seed 42; no mejoró en esta ejecución |
| E07 | — | — | — | Configurado; pendiente de entrenamiento completo |
| E08 | — | — | — | Configurado; pendiente de entrenamiento completo |

Los resultados E05/E06 proceden de las salidas compartidas por el estudiante.
E06 no es el control directo de estos ensayos: tiene un quinto bloque y
589.553 parámetros. Sus losses train/validación pasaron de 0,6090/0,6057 a
0,1355/1,5563, compatibles con un sobreajuste marcado. Ambas ejecuciones
respetaron la separación por paciente y no evaluaron test.

## Antes de sacar conclusiones

- Superar pruebas y smoke tests separados. No interpretar sus AUC como calidad.
- Sigue pendiente comprobar las distribuciones de probabilidades y la estabilidad
  de BatchNorm entre entrenamiento/evaluación en los checkpoints universitarios.
  Estos nuevos ensayos no sustituyen esa comprobación ni demuestran que haya un fallo.
- Mantener el mejor checkpoint por ROC-AUC de validación por paciente. No cambiar
  a posteriori el criterio para escoger una época que parezca más favorable.
- Revisar PR-AUC, pérdidas, sensibilidad, especificidad y matriz de confusión,
  además de tiempo y memoria. No seleccionar solo por accuracy ni por train loss.
- Confirmar candidatos con más semillas y folds; comparar repetidamente sobre
  el mismo fold puede sobreajustar la selección. Test permanece cerrado.

## Ejecución en la GPU universitaria

Cuando se hayan sincronizado estos archivos, ejecutar una orden cada vez:

```bash
python -m src.train --config configs/experiments/E07_dropout_050.json --device cuda
```

Al terminar, ejecutar el ensayo independiente E08:

```bash
python -m src.train --config configs/experiments/E08_weight_decay_001.json --device cuda
```

Cada resultado se guarda en `reports/experiments/<ID>/fold_0_seed_42/` y cada
checkpoint en `checkpoints/<ID>/fold_0_seed_42/`. Conservar `history.csv`,
`summary.json`, curvas y checkpoints mejor/último. Estos directorios siguen
fuera de Git. El runner rechaza una ruta que ya contenga `history.csv`, salvo
que se fuerce con `--overwrite`: **no usar esa opción para estos ensayos**.
Para otra semilla o fold, crear una nueva configuración con nuevas rutas.

La CNN tiene el mismo coste estructural que E05; no se ha medido el tiempo de
E07/E08 en la GPU y no se ejecutará aquí el entrenamiento completo en CPU.

## Smoke tests reproducibles

Dos épocas, 8 pacientes de train y 4 de validación, sin consultar test:

```bash
python -m src.smoke_training --config configs/experiments/SMOKE_dropout_050.json
```

```bash
python -m src.smoke_training --config configs/experiments/SMOKE_weight_decay_001.json
```

Los smoke tests tienen sus propias rutas y no sustituyen al entrenamiento real.
El comando de smoke sobrescribe solo su ejecución técnica si se repite.

Uso educativo y de investigación, sin validez clínica.
