# Plan maestro del proyecto BreastDCEDL

## 1. Objetivo

Construir, entrenar, evaluar y desplegar una CNN 2D propia que reciba las fases
PRE, EARLY y LATE de un corte DCE-MRI realizado antes del tratamiento y estime
la probabilidad de que la paciente alcance respuesta patologica completa (pCR).

El proyecto tiene dos objetivos inseparables:

1. Obtener un pipeline reproducible, robusto y metodologicamente correcto.
2. Comprender y poder defender cada decision de la arquitectura de la CNN.

No se esta detectando la presencia de un tumor. Las imagenes ya estan centradas
en su entorno; se intenta anticipar la respuesta posterior al tratamiento.

## 2. Restricciones no negociables

- CNN 2D implementada con PyTorch y entrenada desde cero.
- Prohibido transfer learning, pesos preentrenados y arquitecturas tomadas como
  ResNet, VGG, EfficientNet o DenseNet.
- Una muestra contiene tres canales temporales en orden PRE, EARLY y LATE; no
  son canales RGB.
- Toda transformacion geometrica debe aplicarse de forma identica a las tres
  fases.
- La separacion estadistica se realiza por paciente, nunca por corte.
- El conjunto test no se usa para elegir arquitectura, hiperparametros, umbral,
  calibracion ni metodo de agregacion.
- Se comparan `BCEWithLogitsLoss` normal y ponderada con `pos_weight=N0/N1`.
- Las metricas finales se calculan principalmente por paciente.
- La aplicacion y el evaluador deben compartir exactamente el mismo codigo de
  preprocesamiento e inferencia.
- La validacion privada del profesor y sus etiquetas nunca se publican.

## 3. Datos disponibles

| Elemento | Cantidad |
|---|---:|
| Pacientes publicas | 1.273 |
| Pacientes train | 1.097 |
| Pacientes test | 176 |
| Cortes | 12.703 |
| Imagenes PNG | 38.109 |
| Canales por corte | 3 |
| Resolucion | 256 x 256 |

La clase pCR representa aproximadamente el 29 % de las pacientes de train, por
lo que la accuracy no puede ser la unica metrica. Las cohortes Duke, I-SPY1 e
I-SPY2 tienen prevalencias y procesos de seleccion de cortes diferentes; se
debe estudiar el posible sesgo de cohorte.

## 4. Principio de trabajo

Cada decision arquitectonica debe responder:

1. Que tensor entra y que tensor sale.
2. Cuantos parametros introduce.
3. Que transformacion realiza.
4. Que problema pretende resolver.
5. Que experimento permitira comprobar si ayuda.

Se cambia una variable principal por experimento. Todas las ejecuciones guardan
configuracion, semilla, metricas, curvas, tiempos y checkpoint.

## 5. Fases del proyecto

### Fase 0 - Fundamentos de arquitectura

**Objetivo:** dominar tensor, batch, canal, kernel, filtro, feature map, padding,
stride, pooling, activacion, logit, sigmoide, perdida y backpropagation.

**Resultado esperado:** poder calcular a mano las dimensiones y los parametros
de una convolucion y explicar por que aumentan los canales mientras disminuye la
resolucion espacial.

- [ ] Calcular varias salidas de `Conv2d` a mano.
- [ ] Distinguir dimensiones espaciales, canales y profundidad de la red.
- [ ] Entender que los pesos de los kernels se aprenden y no se dibujan a mano.
- [ ] Entender el recorrido logit -> sigmoide -> probabilidad.

### Fase 1 - Auditoria reproducible

**Objetivo:** convertir las comprobaciones iniciales en codigo repetible.

- [x] Verificar las 38.109 rutas.
- [x] Validar PNG, escala de grises y resolucion 256 x 256.
- [x] Comprobar unicidad de `sample_id`.
- [x] Comprobar etiqueta constante por paciente.
- [x] Comprobar ausencia de pacientes compartidas entre splits y folds.
- [x] Analizar clases, cohortes, cortes por paciente y datos ausentes.
- [x] Visualizar PRE, EARLY, LATE y EARLY-PRE por clase y cohorte.
- [x] Generar un informe y tests de integridad.

### Fase 2 - Pipeline de datos

**Objetivo:** producir tensores fiables y trazables.

- [x] Implementar carga `[3, 256, 256]` en orden PRE, EARLY, LATE.
- [x] Mantener valores `float32` y rango esperado.
- [x] Preservar `patient_id`, `sample_id`, etiqueta, split y fold.
- [x] Crear DataLoaders de train y evaluacion.
- [x] Usar `shuffle=True` solo en entrenamiento.
- [x] Centralizar preprocesamiento para reutilizarlo en la web.
- [x] Probar que rutas locales y bytes de la futura aplicacion generan tensores identicos.

### Fase 3 - CNN minima de depuracion

**Objetivo:** validar el proceso completo antes de buscar generalizacion.

- [x] Construir una CNN pequena desde cero.
- [x] Comprobar formas de todos los tensores.
- [x] Hacer forward y backward sin errores.
- [x] Sobreajustar deliberadamente 24 muestras de pacientes distintas.
- [x] Confirmar que la perdida baja y que todos los parametros reciben gradiente.

Si la red no puede memorizar ese subconjunto, se depura antes de continuar.

### Fase 4 - Arquitectura base

Arquitectura candidata inicial, todavia no definitiva:

```text
[B, 3, 256, 256]
  -> bloque 3 a 16     -> [B, 16, 128, 128]
  -> bloque 16 a 32    -> [B, 32, 64, 64]
  -> bloque 32 a 64    -> [B, 64, 32, 32]
  -> bloque 64 a 128   -> [B, 128, 16, 16]
  -> Global Average Pooling
  -> Dropout
  -> Linear 128 a 1
  -> un logit por corte
```

Cada bloque candidato contiene dos secuencias
`Conv2d -> BatchNorm2d -> ReLU` seguidas de reduccion espacial. Se comparara
MaxPool con convolucion de stride 2.

- [x] Dibujar la arquitectura.
- [x] Calcular dimensiones capa por capa.
- [x] Calcular parametros capa por capa.
- [x] Calcular o estimar el campo receptivo.
- [x] Justificar Global Average Pooling frente a `flatten` masivo.

### Fase 5 - Entrenamiento base

Configuracion inicial que debe validarse experimentalmente:

- Inicializacion Kaiming.
- Optimizador AdamW.
- Learning rate inicial aproximado `1e-3`.
- Batch 16 o 32 segun memoria.
- Weight decay y dropout.
- Maximo aproximado de 50 epocas.
- Scheduler de learning rate y early stopping.
- Semillas fijas y entorno registrado.
- Checkpoint elegido exclusivamente con validacion interna.

- [x] Superar un smoke test integral sobre un subconjunto por paciente.
- [x] Registrar perdida y metricas por epoca.
- [x] Medir tiempo por epoca, tiempo total y memoria GPU cuando esté disponible.
- [x] Guardar mejor y ultimo checkpoint por separado.
- [x] Crear graficas de aprendizaje y diagnosticar overfitting.

### Fase 6 - Experimentos controlados

Orden propuesto:

| ID | Cambio principal | Pregunta |
|---|---|---|
| E00 | Predictor mayoritario | Cuanto obtiene un modelo que no aprende? |
| E01 | CNN minima | Funciona el pipeline? |
| E02 | CNN base | Cual es la referencia seria? |
| E03 | Perdida ponderada | Mejora la sensibilidad de pCR? |
| E04 | Una/dos convoluciones | Compensa aumentar profundidad? |
| E05 | Menos/mas canales | Falta o sobra capacidad? |
| E06 | Pooling/stride | Como conviene reducir resolucion? |
| E07 | Dropout y weight decay | Se reduce el overfitting? |
| E08 | Aumentos moderados | Mejora la generalizacion? |
| E09 | Ablacion de fases | Que aportan EARLY y LATE? |
| E10 | Realce explicito | Ayudan EARLY-PRE y LATE-EARLY? |

Solo las configuraciones prometedoras pasan a validacion completa de cinco
folds. No se realizan barridos masivos sin una hipotesis clara.

La tabla anterior era la propuesta inicial; las identidades realmente utilizadas
y sus resultados E02-E10 se consultan en `experimentos/resultados_registrados.json`
y `notebooks/05_resultados_experimentos.ipynb`. No confundir los ID propuestos
en ese esquema con los archivos de configuración definitivos.

### Fase 7 - Aumentos de datos

- [ ] Establecer baseline sin aumentos.
- [ ] Probar rotaciones y traslaciones pequenas.
- [ ] Evaluar volteo horizontal tras justificarlo anatomicamente.
- [ ] Probar ruido o intensidad compartida entre fases.
- [ ] Garantizar transformaciones geometricas sincronizadas.
- [ ] Evitar `ColorJitter`, normalizacion ImageNet y cambios independientes.

### Fase 8 - Validacion y seleccion

1. Desarrollar y depurar con un fold interno de train.
2. Crear una lista corta de arquitecturas.
3. Evaluarlas sobre los cinco folds por paciente.
4. Elegir arquitectura, perdida, aumentos, agregacion y umbral sin usar test.
5. Fijar el numero final de epocas usando los resultados de validacion.
6. Reentrenar con todo train.
7. Abrir test una sola vez y no volver a modificar el modelo.

- [ ] Reportar media y desviacion entre folds.
- [ ] Mantener un registro de decisiones y descartes.
- [ ] Congelar configuracion final antes de test.

### Fase 9 - Evaluacion

El modelo genera probabilidades por corte, pero pCR pertenece a la paciente. Se
compararan media, mediana, maximo y voto para agregar cortes; el metodo se elige
con validacion interna.

Metricas principales:

- ROC-AUC y PR-AUC.
- Sensibilidad y especificidad.
- Precision, F1 y balanced accuracy.
- Matriz de confusion.
- Brier score y curva de calibracion.
- Intervalos de confianza mediante bootstrap por paciente.
- Resultados por cohorte y, si hay soporte suficiente, por subgrupos clinicos.

- [ ] Comparar umbral 0,5 con umbral elegido en validacion.
- [ ] Estudiar temperature scaling si la calibracion lo necesita.
- [ ] Analizar falsos positivos y falsos negativos.
- [x] Separar claramente metricas por corte y por paciente.
- [x] Implementar predicciones trazables, curvas e intervalos bootstrap por
  paciente sobre validacion interna, pendientes de ejecutar con E02 y E03.
- [x] Implementar comparacion pareada y reproducible de E02 frente a E03 sobre
  exactamente las mismas pacientes.

### Fase 10 - Interpretabilidad

- [ ] Visualizar kernels de la primera capa antes y despues de entrenar.
- [ ] Visualizar feature maps de varios bloques.
- [ ] Generar mapas tipo Grad-CAM.
- [ ] Comparar ejemplos correctos e incorrectos.
- [ ] Evitar interpretar atencion visual como causalidad clinica.

### Fase 11 - Aplicacion web

- [x] Cargar exactamente PRE, EARLY y LATE.
- [x] Validar formato, firma, tamano, dimensiones y duplicados.
- [x] Mostrar las tres fases y el mapa EARLY-PRE.
- [x] Ejecutar `model.eval()` sin gradientes.
- [x] Mostrar probabilidad, clase y umbral.
- [x] Mostrar checksum del modelo, dispositivo y latencia.
- [ ] Mostrar configuracion y metricas internas.
- [x] Incluir aviso educativo y ausencia de validez clinica.
- [ ] Desplegar en una URL accesible desde otro equipo.
- [ ] Probar cinco cargas consecutivas sin reinicio ni cambios de codigo.

La interfaz se implementa en HTML/CSS/JavaScript por preferencia del estudiante.
Un servidor Python local conecta las cargas con el modelo PyTorch; el destino de
despliegue se decidira tras validar el modelo.

### Fase 12 - Entrega y defensa

- [ ] Repositorio reproducible sin dataset ni datos privados.
- [ ] Pesos distribuidos fuera del historial normal de Git si son grandes.
- [ ] Informe con decisiones, resultados y limitaciones.
- [ ] Exactamente cinco diapositivas.
- [ ] Simulacion de la prueba privada.
- [ ] Ensayo de preguntas sobre dimensiones, parametros y entrenamiento.

## 6. Calendario de 30 dias

| Dias | Objetivo |
|---|---|
| 1-3 | Fundamentos y calculo manual de arquitectura |
| 4-5 | Auditoria reproducible |
| 6-7 | Pipeline de datos y tests |
| 8-9 | CNN minima y sobreajuste controlado |
| 10-12 | CNN base y entrenamiento inicial |
| 13-16 | Arquitectura, regularizacion y desbalance |
| 17-19 | Aumentos y ablaciones |
| 20-22 | Cinco folds y seleccion final |
| 23 | Entrenamiento definitivo con todo train |
| 24 | Evaluacion unica sobre test |
| 25-27 | Aplicacion, despliegue y pruebas |
| 28 | Informe |
| 29 | Cinco diapositivas |
| 30 | Simulacion de defensa y margen |

## 7. Criterio de finalizacion

El proyecto se considera terminado cuando:

- Una instalacion limpia puede cargar el modelo y reproducir la inferencia.
- No existe solapamiento de pacientes ni contaminacion de test.
- Arquitectura y configuracion estan guardadas y justificadas.
- Evaluador y web producen el mismo tensor y la misma probabilidad.
- Existen metricas por paciente, calibracion e intervalos de confianza.
- La web procesa cinco casos consecutivos y gestiona errores.
- La URL funciona desde un equipo distinto.
- El estudiante puede dibujar la CNN y explicar cada capa sin leer el codigo.
- Las limitaciones clinicas, eticas y de generalizacion estan documentadas.

## 8. Registro de decisiones

Esta tabla se actualizara durante el proyecto. Una decision no se borra: si se
revierte, se anade una nueva fila que explique por que.

| Fecha | Decision | Evidencia | Estado |
|---|---|---|---|
| 2026-09-22 | Mantener `breastdcedl/dataset/` fuera de Git | 38.109 PNG y aproximadamente 1,3 GB | Aceptada |
| 2026-09-22 | Dar prioridad al aprendizaje y defensa de la arquitectura | Objetivo formativo del estudiante | Aceptada |
| 2026-09-22 | Separar siempre por paciente y reservar test | Requisito del enunciado y prevencion de leakage | Aceptada |
| 2026-09-23 | Completar la auditoria antes de implementar la CNN | 38.109 imagenes validadas, 0 errores criticos | Completada |
| 2026-09-23 | Usar una unica funcion para Dataset y futura web | Rutas y bytes producen tensores identicos en tests | Aceptada |
| 2026-09-23 | Validar el aprendizaje con una CNN minima de 6.545 parametros | 100 % sobre 24 muestras, loss final 0,0502 y gradientes finitos | Completada |
| 2026-09-25 | Hacer bootstrap y comparacion E02/E03 de forma pareada por paciente | La paciente es la unidad independiente y ambos modelos comparten fold | Implementada; pendiente de resultados |
| 2026-09-25 | Reutilizar `load_dce_triplet` tambien desde la web | Evita discrepancias PRE/EARLY/LATE entre evaluacion y despliegue | Implementada |
| 2026-09-25 | Usar HTML/CSS/JavaScript para la interfaz web | Preferencia explicita del estudiante; Flask conserva el pipeline PyTorch compartido | Implementada localmente; despliegue pendiente |
| 2026-09-30 | Repetir E05 en Mac con rutas separadas antes de comparar ablaciones entre dispositivos | E08/E10 en MPS no mejoraron el resultado observado de E05 en ROCm; precisión y plataforma distintas | Configurada; entrenamiento pendiente |
| 2026-09-30 | Analizar validación por cohorte sin alterar umbral, agregación ni CNN | E08/E10: delta AUC 0,007964 con IC pareado que incluye cero; spy1 solo tiene cuatro positivas | Informe exploratorio local y 77 pruebas; test cerrado |
| 2026-09-30 | Cerrar la comparación E05/E08/E10 en Mac y conservar métricas | E05 Mac AUC 0,586895; diferencias E08/E10 frente a referencia con IC globales que incluyen cero | Referencia completa, cohortes y notebook actualizados; no se inicia otro experimento |
| 2026-10-01 | Probar LR inicial 0,0003 y 0,0001 manteniendo E05 Mac | LR 0,001 es recomendación, no requisito; hipótesis de actualizaciones menores | E11/E12 completos: AUC 0,550202/0,557964, mejores épocas 2/3; no superan E05 Mac y persiste sobreajuste |
| 2026-10-01 | Probar diferencias firmadas entre fases en E13 | Petición de continuar: misma CNN E05 Mac y mismos ajustes; solo cambiar representación | Implementado dentro del modelo con cero parámetros nuevos; smoke completo superado, entrenamiento científico pendiente |

## 9. Siguiente accion

El registro de experimentos, el notebook de resultados y la comparación por
cohorte están actualizados. E05 Mac, E08 y E10 han terminado y no muestran
una mejora global clara en este fold. El objetivo ROC-AUC 0,7 sigue sin
alcanzarse; no hay arquitectura definitiva ni test evaluado.

El estudiante ha retomado la mejora y acordado probar dos learning rates
iniciales menores: E11/E12, sin cambiar otra variable principal de E05 Mac.
Ambas ejecuciones ya han terminado y se han registrado con sus curvas locales.
La reducción del LR no mejoró el ROC-AUC observado; no prolongar por defecto.
E13 está preparado: PRE, EARLY−PRE, LATE−EARLY dentro del modelo, conservando
la CNN y ajustes E05 Mac. Protocolo en `experimentos/REALCE_E13.md` y ficha
PNG/SVG en `experimentos/E13_phase_differences/`. Smoke de dos épocas CPU
con 8/4 pacientes train/validación completado en 4,10 s, sin evaluar test.
El estudiante completó E13: diez épocas MPS, mejor época 10, ROC-AUC
0,572480, AP 0,350860 y 421,48 s; no supera E05 Mac. Registro/notebook
actualizados. Ha autorizado diagnosticar train/validación antes de elegir
prolongación o cambio estructural. Test cerrado y sin commit/push.
Diagnóstico ya completado en MPS: train AUC 0,788197 vs validación 0,572480;
copia BN validación 0,557964. No resuelve la generalización; documentación
en `experimentos/DIAGNOSTICO_E13.md`. No hay autorización para otro modelo
ni entrenamiento largo; acordar una hipótesis controlada de generalización.
