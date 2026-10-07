# E56/E57 · Media de características antes de decidir

Preparación solicitada el 7 de octubre de 2026. No son los canales constantes
propuestos inicialmente como E56: esa propuesta no se implementó. Aquí E56
es el control de cabeza no lineal, y E57 el ensayo de agregación de características.
Sin transfer learning, pesos ajenos, Git, web, despliegue o test. Científicos
pendientes por el estudiante; no reanuda la campaña autónoma terminada.

## Corrección conceptual y motivación

Con una cabeza lineal, `Linear(mean(features)) == mean(Linear(features))`
en evaluación: mover la media no aprende relaciones nuevas. La media de logits
NO es la media de probabilidades. No vender este cambio como una solución segura.
Usamos una cabeza propia no lineal128→32→1, pequeña y entrenada desde cero.
La media de features sigue siendo una suma invariante al orden: no aprende
interacciones por pares, posición axial, selección de cortes ni atención.
Puede combinar señales de diferentes cortes antes de una decisión no lineal,
pero también puede diluir señal. No garantiza superar E19 ni alcanzar0,7.

## Comparación fijada antes de resultados

- E56: extractor compartido→MLP por corte→media probabilidades por paciente→BCE.
- E57: extractor compartido→media features por paciente→MLP una vez→BCE.

Ambos tienen exactamente298161 parámetros: extractor294000 y cabeza4161.
Misma inicialización/semilla/fold/lotes de DOS pacientes completas: aproximadamente
20 cortes variables por paso, no batch16 efectivo. Se reutiliza el sampler E26/E27.
Todo train interno878/219 pacientes,8759/2186 cortes, AdamW LR/WD0,001,
BCE normal por paciente, Plateau factor0,5/patiencia3, máximo50/paciencia10,
MPSfloat32, sin aumentos/EMA/GeM/label smoothing/SAM. Cada paciente una vez por época.
Son439 actualizaciones/época, no548. Ambos usan las mismas fuentes de datos.
E57−E56 aísla la política de agregación incluyendo la granularidad de dropout
(por corte vs por bolsa); no solo una operación aritmética en entrenamiento.
E57−E19 cambia varios elementos: no atribuir esa diferencia solo a pooling.

## Evaluación y defensa con un corte

`forward([B,3,256,256])` clasifica cada corte como una bolsa de tamaño uno.
`forward_patients` acepta números variables de cortes; con N=1 y eval es idéntico.
La evaluación concatena todas las features antes de agrupar: no divide pacientes
cuando atraviesan varios lotes de lectura. No se usa test ni se ajustan umbrales.

**Criterio primario fijo:** AUC por paciente de la media de probabilidades de
los cortes evaluados individualmente, como E19. Guía scheduler/checkpoint/early stopping.
Es un criterio comparable con el historial, NO simula literalmente un único corte
por paciente de la defensa. Guardar también métricas por corte y un corte central
por paciente como diagnóstico prefijado, sin afirmar que represente toda selección
aleatoria del profesor. El modo bolsa completa tiene su propio AUC/CSV secundarios;
no sustituir el monitor si solo ese sube. Requerimos evaluar el modo de un corte
antes de adopción. AUC de bolsa alto no garantiza funcionamiento en defensa.

Train loss E57 se refiere a bolsas; val_loss principal a media de probabilidades
de cortes individuales. Diferentes objetivos: no interpretar su diferencia como
una comparación pura de sobreajuste. `bag_val_loss` compara train/val del objetivo
de bolsa en E57, con la salvedad train/dropout/BN frente eval.

## Comandos Mac, secuenciales

```bash
python -u -m src.train_patient_features --config configs/patient_features/E56_nonlinear_cut_control_mac.json --device mps
python -u -m src.train_patient_features --config configs/patient_features/E57_patient_feature_mean_mac.json --device mps
```

Esperar a que termine el primero antes de ejecutar el segundo. No iniciar ambos
en terminales separadas. No combinar con E55 aún activo. Sin fallback CPU silencioso.
No overwrite/resume: se rechaza cualquier carpeta ya existente, incluso parcial.
Runner dedicado y esquema distinto al legacy; no usar `src.train` para estos JSON.
Resultados `reports/patient_features/<experimento>/fold_0_seed_42`, pesos en
`checkpoints/<experimento>/fold_0_seed_42`. Guarda curvas, configuración, entorno,
mejor/último peso, predicciones y ambos modos de evaluación. Conservar los dos ensayos.

Checkpoints etiquetados PatientFeatureNet_v1, restaurar con
`src.patient_feature_model.load_feature_checkpoint`. Usan el preprocesamiento
compartido actual pero **no están integrados con la web/loader antiguo**.
La integración web requerirá un cambio explícito cuando se elija un modelo.
No introducir estos informes en comparadores legacy que exigen BreastPCRNet
sin adaptar su validación de esquema; serie nueva separada del catálogo científico
legacy, con fichas en experimentos. Guardar ausencia de resultado como pendiente.

Uso educativo y de investigación, sin validez clínica.
