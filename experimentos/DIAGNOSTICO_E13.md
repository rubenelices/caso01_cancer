# Diagnóstico E13: aprende train, generaliza poco

Evaluación del mejor checkpoint E13, época 10, en Mac MPS float32.
Batch 16, seed 42, fold 0, sin aumentos ni dropout; 878 pacientes de train
y 219 de validación. No se entrenó ni se evaluó test. La configuración y
pesos originales no cambiaron; tampoco se guardaron nuevos pesos.

| Condición | ROC-AUC paciente | AP paciente | Sensibilidad (umbral 0,5) |
|---|---:|---:|---:|
| Train original | 0,788197 | 0,620341 | 0,492248 |
| Validación original | 0,572480 | 0,350860 | 0,218750 |
| Train copia BN | 0,803645 | 0,636333 | 0,379845 |
| Validación copia BN | 0,557964 | 0,344283 | 0,218750 |

La diferencia train−validación original es 0,215717 de ROC-AUC. Es evidencia
de un problema de generalización, no de ausencia de aprendizaje incluso en train.
No demuestra que se memoricen literalmente los píxeles ni identifica una causa
única; pueden contribuir correlación entre cortes, señal limitada y sesgos de cohorte.

Recalcular estadísticas BatchNorm en una copia usando solo train no ayuda al
ROC-AUC de validación: delta −0,014516, intervalo bootstrap pareado exploratorio
95 % [−0,029791; −0,000955], 500 repeticiones estratificadas por clase/paciente.
Este intervalo no corrige selección del checkpoint ni múltiples ensayos.
No demuestra que BatchNorm sea universalmente inadecuada; esta recalibración
depende del lote y su composición. A umbral 0,5 mejora especificidad, pero
el criterio principal sigue siendo ROC-AUC, no la métrica más favorable.

No prolongar E13 automáticamente ni añadir capas basándose solo en que la mejor
época era la última. La próxima hipótesis debe abordar generalización y conservar
una referencia comparable. Candidatos prometedores requieren otras semillas/folds.

El diagnóstico completo válido está en
`reports/diagnostics/E13_phase_differences/mps_epoch10/` (ignorado en Git).
Un intento CPU anterior se interrumpió para usar MPS: su carpeta
`reports/diagnostics/E13_phase_differences/0d3fd4ba9dde/` es **parcial**, no
un informe completo y no se usa en estas conclusiones. No se borraron archivos.

[Snapshot agregado sin identificadores de pacientes](diagnostico_E13_registrado.json).

Uso educativo y de investigación, sin validez clínica.
