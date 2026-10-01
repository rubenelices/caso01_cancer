# Referencia E05 en Mac y diagnóstico por cohorte

## Por qué esta comprobación

Los resultados observados no justifican seguir modificando capas sin diagnóstico:

| Ejecución | Mejor época | AUC paciente | AP paciente | Dispositivo |
|---|---:|---:|---:|---|
| E05 original | 1 | 0,614012 | 0,390992 | ROCm/CUDA |
| E05 referencia Mac | 1 | 0,586895 | 0,374219 | Mac MPS |
| E08, weight decay 0,001 | 3 | 0,582661 | 0,355411 | Mac MPS |
| E10, mitad de canales | 6 | 0,590625 | 0,379061 | Mac MPS |

E05 Mac completó diez épocas en 414,01 segundos, E08 en 402,88 y E10 en 198,61.
El dispositivo y la precisión cambian entre universidad y Mac; la misma semilla
no garantiza trayectorias idénticas entre plataformas. No atribuir toda diferencia
a arquitectura o regularización. E02 también obtuvo 0,6204: E05 es nuestra
referencia para estas ablaciones, no la ganadora definitiva de todo el proyecto.

## 1. Repetir E05 sin sobrescribirlo

`E05_mac_reference.json` conserva todos los ajustes de E05, incluidos fold 0,
seed 42, batch 16, diez épocas, canales 16/32/64/128, ocho convoluciones,
pooling intermedio, LR inicial 0,001, dropout 0,3, weight decay 0,0001 y BCE normal.
Solo cambia identidad, descripción y carpetas. No usa aumentos ni pesos anteriores.

```bash
python -m src.train --config configs/experiments/E05_mac_reference.json --device auto
```

En el entorno del estudiante, `auto` ha elegido MPS: GPU integrada de Apple,
computación local, no un servicio de pago. Comprobar que `environment.device`
vuelve a ser `mps`, con el mismo Python/PyTorch que E08/E10. MPS usa float32 en
este runner; CUDA podía usar precisión mixta. No forzar CUDA en el Mac.

Guardar por separado en:

- `reports/experiments/E05_mac_reference/fold_0_seed_42/`
- `checkpoints/E05_mac_reference/fold_0_seed_42/`

Ejecución completa ya realizada por el estudiante; el comando queda como
documentación, no como instrucción de repetirla ahora. No usar `--overwrite`.
Su coste puede ser del orden de E08 (unos siete minutos en la ejecución
compartida), pero cambia con temperatura, carga y entorno. No es un plazo garantizado.
No hay una nueva arquitectura: reutiliza [el dibujo A04](A04_pool_intermedio/README.md).

## 2. Analizar cohortes sin volver a inferir

`src/analyze_cohorts.py` lee las predicciones ya guardadas del mejor checkpoint.
Anota por `patient_id` la cohorte de `metadata/patients.csv`; no usa la cohorte
como entrada de la red ni entrena un nuevo clasificador. No lee PNG ni checkpoints.

El informe E08/E10 permanece en `reports/cohorts/E08_E10_mac/`. La comparación
completa E05/E08/E10 ya se ha generado en `reports/cohorts/E05_E08_E10_mac/`.
Este es el comando utilizado (no repetir sobre el mismo destino):

```bash
python -m src.analyze_cohorts \
  --experiment-dir reports/experiments/E05_mac_reference/fold_0_seed_42 \
  --experiment-dir reports/experiments/E08_weight_decay_001/fold_0_seed_42 \
  --experiment-dir reports/experiments/E10_half_width/fold_0_seed_42 \
  --output-dir reports/cohorts/E05_E08_E10_mac
```

El primer experimento es la referencia para las diferencias pareadas. La barra
inversa al final permite pegar todo el bloque en terminal como un solo comando.
Si el destino ya existe, el script lo rechaza: elegir un nombre nuevo.

## Controles y archivos

- Solo ejecuciones completas sin debug ni test; validación interna completa.
- Se comprueba cada corte contra `samples.csv`: identidad, etiqueta, fold y
  cobertura exacta. La media por paciente debe coincidir con el CSV guardado.
- Las cohortes deben existir, sin duplicados ni etiquetas contradictorias.
- Las comparaciones requieren mismas pacientes, etiquetas, cohortes, fold,
  semilla, agregación y umbral. No cambia el umbral ni selecciona otro checkpoint.
- Total y por cohorte: tamaños/clases, prevalencia, AUC, AP, sensibilidad,
  especificidad, precisión, F1, balanced accuracy, Brier y matriz de confusión.
- Bootstrap estratificado por paciente: IC 95 % de AUC/AP y diferencias pareadas
  de AUC. Las diferencias remuestrean las mismas pacientes dentro de cohorte/clase.
- Estadísticas de probabilidades por clase, gráficos ROC por cohorte,
  distribución de probabilidades y comparación de prevalencia/media predicha.

Produce `README.md`, `cohort_dashboard.png`, `cohort_metrics.csv`,
`cohort_summary.json` y CSV anotados por experimento. Incluye configuración,
entorno y hashes de los informes originales para trazabilidad.
Informes e identificadores permanecen locales y fuera de Git.

## Cómo interpretar

### Estado observado al cerrar esta comparación

E05 Mac completó diez épocas y seleccionó época 1, AUC 0,586895 y AP 0,374219.
La pérdida train pasó de 0,612911 a 0,4213 y la de validación de 0,597712 a
0,7559 (los valores finales están redondeados de la salida comunicada): sigue
un patrón compatible con sobreajuste. Su sensibilidad a umbral 0,5 es cero.

| Candidato frente a E05 Mac | Delta AUC | IC pareado exploratorio 95 % |
|---|---:|---|
| E08 | −0,004234 | [−0,040867; 0,032825] |
| E10 | +0,003730 | [−0,076872; 0,077167] |

Ninguna ventaja global clara. AUC de E05 por cohorte: Duke 0,381232,
I-SPY1 0,617647 e I-SPY2 0,615106. E10−E05 en Duke tiene delta 0,178886 e
intervalo nominal [0,004326; 0,376906], pero es un análisis exploratorio sin
corrección por consultas múltiples: no confirma una mejora general ni explica
la causa. La cohorte Duke tiene 42 pacientes (11 positivas), I-SPY1 21 (4) e
I-SPY2 156 (49). Conservar resultados y no seleccionar una red por un subgrupo.

El notebook `notebooks/05_resultados_experimentos.ipynb` recoge esta comparación.
`experimentos/cohortes_registradas.json` guarda una copia agregada seleccionada
de los informes, sin identificadores: permite consultar las cifras sin los
archivos locales. La procedencia se muestra al ejecutar el cuaderno.
Por petición del estudiante, se actualiza el registro y no se lanza otro
experimento ni se modifica la arquitectura en esta etapa.

1. Si E05 en Mac mantiene una ventaja, E08/E10 no mejoran esta referencia en
   esta ejecución. Un IC del delta que cruza cero no permite afirmar ventaja.
2. AUC global y AUC dentro de cohortes responden a comparaciones distintas.
   Una discrepancia puede sugerir estudiar la composición del dataset; no
   demuestra que la CNN haya aprendido el hospital ni identifica una causa.
3. AP depende de prevalencia; mirar clases/tamaños, no solo ordenar cohortes
   por esa cifra. Grupos con menos de diez pacientes en alguna clase llevan
   aviso. Con una única clase, AUC/AP quedan como no definidos, no como cero.
4. Bootstrap exploratorio condicionado a la composición observada. No corrige
   consultas repetidas al fold ni selección del mejor checkpoint. Confirmar
   propuestas en otras semillas y folds antes de conclusiones fuertes.

No modificar preprocesamiento, agregación o representación del realce basándose
solo en un subgroup pequeño. Cada propuesta posterior requiere configuración e
hipótesis propias. Test permanece cerrado. Uso educativo, no clínico.
