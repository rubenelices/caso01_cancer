# AGENTS.md

## Proposito

Este repositorio corresponde a un proyecto individual de Aprendizaje Automatico
sobre BreastDCEDL. El objetivo es predecir respuesta patologica completa (pCR)
a partir de tres fases de DCE-MRI anteriores al tratamiento y desplegar una
demostracion web reproducible.

El estudiante quiere alcanzar un nivel tecnico alto, pero tambien comprender y
poder defender cada decision. Al colaborar en este repositorio, prioriza codigo
claro, explicaciones pedagogicas y experimentos con una hipotesis explicita.

Lee `PLAN.md`, `breastdcedl/GUIA.md` y el enunciado oficial antes de cambiar el
protocolo experimental.

## Hechos del problema

- Entrada por corte: tensor `[3, 256, 256]`.
- Orden de canales: PRE, EARLY, LATE.
- Los canales son fases temporales, no RGB.
- Objetivo binario: `pCR=1` frente a `pCR=0`.
- La unidad estadistica es la paciente, aunque la CNN procese cortes.
- Hay aproximadamente diez cortes correlacionados por paciente.
- Train publico: 1.097 pacientes.
- Test publico: 176 pacientes.
- Validacion privada: solo profesorado.
- Las cohortes Duke, I-SPY1 e I-SPY2 pueden introducir sesgo.

## Reglas obligatorias

1. Implementar una CNN 2D desde cero con PyTorch.
2. No usar transfer learning, pesos preentrenados ni arquitecturas importadas
   como ResNet, VGG, EfficientNet o DenseNet.
3. No dividir cortes aleatoriamente. Usar folds agrupados por paciente.
4. No usar test para seleccionar arquitectura, hiperparametros, umbral,
   calibracion, epocas ni agregacion.
5. Comparar `BCEWithLogitsLoss` normal y ponderada.
6. Aplicar cualquier aumento geometrico a las tres fases de forma sincronizada.
7. No aplicar normalizacion ImageNet ni aumentos RGB.
8. Evaluar y reportar principalmente por paciente.
9. Mantener identico el preprocesamiento entre entrenamiento, evaluacion y web.
10. No publicar validacion privada, etiquetas ocultas ni datos sensibles.
11. Incluir siempre el aviso de que el resultado es educativo y no clinico.

## Politica de Git y archivos grandes

- Nunca anadir ninguna ruta de `breastdcedl/` al indice de Git. El paquete
  docente completo permanece local por decision del usuario.
- No versionar checkpoints, pesos, ejecuciones, uploads ni secretos.
- Antes de un commit, revisar `git status` y comprobar que no aparecen PNG del
  dataset ni ficheros privados.
- No ejecutar `git add .` sin revisar primero los ficheros candidatos.
- No hacer commit, push, crear ramas ni reescribir historial salvo peticion
  explicita del usuario.
- Versionar solo nuestro codigo, configuraciones, tests, notebooks y
  documentacion propia. Los informes regenerables permanecen locales.
- Distribuir pesos grandes mediante releases o almacenamiento externo cuando se
  decida el mecanismo de despliegue.

## Forma de trabajar

- Mantener la logica principal en modulos, no solo en notebooks.
- Usar notebooks para exploracion y comunicacion, no como unica fuente de verdad.
- Centralizar carga, transformaciones e inferencia para evitar discrepancias.
- Guardar configuraciones de experimentos en archivos legibles y versionados.
- Fijar semillas y registrar versiones, dispositivo, tiempos y memoria.
- Anadir tests para integridad de datos, formas de tensores e inferencia.
- Ejecutar primero smoke tests pequenos antes de entrenamientos largos.
- Cambiar una variable principal por experimento siempre que sea posible.
- No afirmar que una tecnica mejora el modelo sin evidencia de validacion.
- Preservar cambios existentes del usuario y evitar operaciones destructivas.

## Convenciones conceptuales

- No describir el proyecto como deteccion de tumores. Predice pCR futura.
- No llamar RGB a PRE/EARLY/LATE.
- No confundir `slice_index` con un instante temporal.
- No llamar backtesting a la evaluacion; distinguir train, validacion y test.
- No interpretar una probabilidad como supervivencia, curacion o recomendacion
  de omitir cirugia.
- No presentar Grad-CAM o feature maps como explicaciones causales.

## Arquitectura y pedagogia

Para cada capa o bloque nuevo, documentar:

- Forma de entrada y salida.
- Numero de parametros.
- Kernel, padding y stride.
- Cambio de resolucion y canales.
- Campo receptivo cuando sea relevante.
- Motivo de la decision.
- Experimento que medira su utilidad.

Cuando se implemente la primera CNN, empezar por una red minima capaz de
sobreajustar 20-50 muestras. No avanzar a busquedas de arquitectura hasta que
esa prueba funcione.

## Evaluacion

- Mantener metricas por corte separadas de metricas por paciente.
- Agregar probabilidades por paciente solo con un metodo elegido en validacion.
- Incluir ROC-AUC, PR-AUC, sensibilidad, especificidad, precision, F1, balanced
  accuracy y matriz de confusion.
- Estudiar calibracion e intervalos de confianza por paciente.
- Analizar rendimiento por cohorte sin sacar conclusiones fuertes de subgrupos
  pequenos.

## Estado actual

- Enunciado, guia, metadatos y dataset local disponibles.
- Integridad basica comprobada: 38.109 imagenes presentes y sin solapamiento de
  pacientes entre train y test.
- El remoto privado solo contiene el commit inicial.
- `breastdcedl/dataset/` esta ignorado por Git.
- La auditoria reproducible esta implementada en `src/audit_data.py` y ha
  validado las 38.109 imagenes sin errores criticos.
- El pipeline unico esta implementado en `src/data.py`; rutas y bytes producen
  el mismo tensor PRE/EARLY/LATE y los splits son disjuntos por paciente.
- La CNN minima de `src/model.py` tiene 6.545 parametros y ha memorizado 24
  muestras equilibradas con 100 % de accuracy y loss final 0,0502. Es una
  prueba de cableado, no una estimacion de generalizacion.
- El notebook ejecutado `notebooks/03_cnn_minima.ipynb` documenta formas,
  parametros, forward, backward y la curva de memorizacion.
- La arquitectura candidata `BreastPCRNet` tiene cuatro bloques de dos
  convoluciones, 294.129 parametros y esta documentada en
  `notebooks/04_arquitectura_base.ipynb`.
- E02 (BCE normal) y E03 (BCE ponderada) estan configurados. El runner registra
  metricas por corte y paciente, early stopping, checkpoints, tiempos y memoria.
- El benchmark local con batch 16 estima unos 347 segundos por epoca y 4,82
  horas para 50 epocas, sin validacion; se ha decidido esperar a la GPU.
- El smoke test integral ha recorrido 2 epocas con 8 pacientes de train y 4 de
  validacion en 5,26 s. Ha verificado metricas, curvas, mejor/ultimo checkpoint
  y que test no se evalua. Sus cifras estan marcadas como no cientificas.
- No se ha entrenado todavia una arquitectura candidata sobre train completo ni
  se ha consultado el test.

## Proxima tarea recomendada

Revisar pedagogicamente la arquitectura base y ejecutar E02 y E03 en la GPU de
la universidad. Comparar validacion por paciente antes de proponer cambios.
