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
11. Mantener el aviso educativo y no clinico en web, documentación y artefactos;
    por petición del estudiante, no repetirlo como coletilla en cada mensaje.

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

Por peticion del usuario, toda arquitectura nueva debe tener una carpeta en
`experimentos/` con README explicativo, diagrama de volumenes PNG y SVG,
dimensiones y parametros comprobados contra el modelo real. Mantener el
catalogo `experimentos/README.md` y el generador `src/document_architectures.py`.
Experimentos que solo cambian la perdida comparten la misma ficha arquitectonica.

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
- El remoto privado contiene las arquitecturas E02-E09 y el diagnóstico hasta
  c6f23d7; se está preparando la publicación del paquete de continuación E10-E13.
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
- El benchmark inicial mezclaba lectura y calculo, forzaba `num_workers=0` y no
  usaba precision mixta, por lo que su estimacion no era comparable con el
  entrenamiento real. Se ha sustituido por un benchmark que separa lectura,
  transferencia y calculo puro, usa calentamiento y replica la configuracion.
- La GPU universitaria es una RX 6700 XT de 12 GB. ROCm la expone como
  `gfx1030` mediante `HSA_OVERRIDE_GFX_VERSION=10.3.0`, aunque su arquitectura
  nativa es `gfx1031`; falta ejecutar alli el benchmark corregido.
- El smoke test integral ha recorrido 2 epocas con 8 pacientes de train y 4 de
  validacion en 5,26 s. Ha verificado metricas, curvas, mejor/ultimo checkpoint
  y que test no se evalua. Sus cifras estan marcadas como no cientificas.
- La evaluacion interna guarda predicciones por corte y paciente, ROC,
  precision-recall, calibracion, matriz de confusion e intervalos bootstrap por
  paciente. El comparador E02/E03 usa bootstrap pareado y exige las mismas
  pacientes, fold, agregacion y umbral.
- `src/inference.py` centraliza la carga de checkpoints y la inferencia desde
  PRE, EARLY y LATE reutilizando `load_dce_triplet`. La interfaz elegida por el
  usuario es HTML/CSS/JavaScript en `web/`, servida localmente por `app.py` con
  Flask para conectar con PyTorch. Incluye validacion, previsualizacion, realce,
  trazabilidad y aviso no clinico; falta conectarla a un checkpoint cientifico
  y decidir el despliegue.
- El estudiante ha ejecutado E02/E03/E04/E05 en la GPU universitaria con el fold
  interno 0. E05 completo 10 epocas, mejor checkpoint en epoca 1: ROC-AUC por
  paciente 0,614012, PR-AUC 0,390992; todas negativas a umbral 0,5. No se ha
  consultado el test ni elegido una arquitectura definitiva.
- E06 anade un quinto bloque de 128 canales a E05, sin cambiar el resto del
  entrenamiento: 589.553 parametros, diez convoluciones, cinco MaxPool, mapa
  final 128x8x8 y campo receptivo local teorico 218x218. Configuracion en
  `configs/experiments/E06_five_blocks_normal.json`, ficha PNG/SVG en
  `experimentos/A05_cinco_bloques/`. Pruebas de arquitectura y smoke de dos
  epocas superados. El estudiante completo el entrenamiento de E06: mejor epoca
  6, ROC-AUC por paciente 0,574647 y PR-AUC 0,367224, con sobreajuste marcado.
  No mejoro E05 en este fold/semilla. El AUC del smoke no es
  evidencia de calidad. Ver `experimentos/PROTOCOLO_COMPARACION.md`.
- E07 (`E07_dropout_050.json`) cambia solo dropout a 0,5 respecto a E05;
  E08 (`E08_weight_decay_001.json`) cambia solo weight decay a 0,001 y mantiene
  dropout 0,3. Ambos conservan 10 epocas, LR inicial 0,001, BCE normal, fold 0,
  seed 42 y rutas propias. Fichas y PNG/SVG en `experimentos/E07_dropout_050/`
  y `experimentos/E08_weight_decay_001/`; comparacion en
  `experimentos/REGULARIZACION.md`. Las 49 pruebas y ambos smoke tests han
  pasado. El estudiante completo E07 (10 epocas, mejor epoca 1): ROC-AUC por
  paciente 0,595565 y PR-AUC 0,368886; todas negativas a umbral 0,5. E08 ya
  tiene resultados en Mac documentados abajo. No combinar cambios sin crear
  otro experimento ni sobrescribir resultados con `--overwrite`.
- E09 (`E09_shared_affine.json`) mantiene modelo/entrenamiento de E05 y activa
  solo en train rotacion hasta 5 grados y traslacion hasta 3 % por eje. Una
  rejilla bilineal compartida transforma las tres fases; relleno cero. No cambia
  la carga [0,1] de validacion, test o web. Configuraciones antiguas mantienen
  aumentos desactivados. Implementacion en `src/augmentation.py`, conexion en
  `src/train.py` y configuracion serializable en `src/experiment_config.py`.
  59 pruebas y smoke integral de dos epocas superados. Ficha con diagrama en
  `experimentos/E09_shared_affine/`; protocolo en `experimentos/AUMENTOS_E09.md`.
  `src/preview_augmentation.py` genera una lamina de dos cortes de train en
  `reports/data_augmentation/E09_shared_affine/`, inspeccionada visualmente.
  Las imagenes de pacientes y los pesos permanecen locales e ignorados.
- El estudiante completo E09: diez epocas, mejor checkpoint en epoca 1,
  ROC-AUC por paciente 0,610282 y PR-AUC 0,375773. No mejoro E05 en este
  fold/semilla; no se ha consultado test.
- `src/diagnose_training.py` prepara un diagnostico sin entrenamiento: compara
  train/validacion sin aumentos, guarda probabilidades y curvas, y recalcula
  estadisticas BatchNorm en una copia usando solo train. No modifica parametros
  aprendidos ni el checkpoint original, no guarda pesos nuevos y no evalua test.
  Documentado en `experimentos/DIAGNOSTICO.md`; 65 pruebas superadas y ejecucion
  tecnica local verificada con el checkpoint del smoke E09. El diagnostico
  cientifico de E05 ya fue ejecutado en universidad para épocas 1 y 10, con
  resultados comunicados por el estudiante y registrados en el notebook.
  Diagnostico subido en el commit c6f23d7: solo cinco archivos propios, sin
  web/evaluacion/benchmark. La copia exacta del indice paso sus 54 pruebas.

## Proxima tarea recomendada

Nueva petición de mejora autorizada: E11/E12 cambian solo LR inicial de E05 Mac
a 0,0003/0,0001; el estudiante aclaró que 0,001 es recomendación, no requisito.
Configuraciones `E11_lr_0003.json` y `E12_lr_0001.json`, diez épocas, mismo
scheduler y demás ajustes. Smokes separados `SMOKE_lr_0003.json` y
`SMOKE_lr_0001.json`. Fichas PNG/SVG reutilizando A04 en carpetas E11/E12;
protocolo `experimentos/LEARNING_RATE.md`. El estudiante completó ambas en
Mac MPS: E11 mejor época 2, AUC 0,550202, AP 0,344140, 399,15 s; E12 mejor
época 3, AUC 0,557964, AP 0,386688, 481,87 s. Ninguna supera el ROC-AUC de
E05 Mac (0,586895); persisten señales de sobreajuste. No cambiar el criterio
a AP por ser más favorable en E12 ni pedir repetirlas como pendientes.
Registro, notebook y fichas conservan resultados completos, no cifras del smoke.
Test sigue cerrado; sin commit/push. El estudiante autorizó continuar con E13.
`src/phase_representation.py` transforma dentro de BreastPCRNet las fases
originales a PRE, EARLY−PRE y LATE−EARLY, conservando negativos y cero
parámetros nuevos. `model.input_representation` se serializa; por defecto raw
para configuraciones/checkpoints previos. La carga de PNG no cambia ni se deben
pasar diferencias ya calculadas desde la web. E13 replica E05 Mac salvo esa
representación: 294.129 parámetros, LR inicial 0,001, diez épocas, sin aumentos.
Config `E13_phase_differences.json`, smoke `SMOKE_phase_differences.json`, ficha
PNG/SVG `experimentos/E13_phase_differences/`, protocolo `REALCE_E13.md`.
Smoke integral completado en 4,10 s CPU; métricas no científicas y test cerrado.
El estudiante completó E13 en MPS: diez épocas, mejor época 10, ROC-AUC
0,572480, AP 0,350860, 421,48 s. No supera E05 Mac; test cerrado.
Registro/notebook actualizados; diagnóstico train/validación autorizado.
Diagnóstico E13 completado en MPS: train AUC 0,788197, validación 0,572480,
copia BN validación 0,557964; delta BN −0,014516 con IC exploratorio
[−0,029791; −0,000955]. Checkpoint y modelo originales sin cambios, test cerrado.
Informe válido `reports/diagnostics/E13_phase_differences/mps_epoch10/`;
el intento CPU `0d3fd4ba9dde/` está incompleto, no usarlo como diagnóstico final.
Snapshot agregado sin pacientes en `experimentos/diagnostico_E13_registrado.json`
y ficha `DIAGNOSTICO_E13.md`. No alargar automáticamente ni crear más modelos
sin petición; siguiente hipótesis debe centrarse en generalización.
Las 103 pruebas del repositorio pasan y el notebook se ejecutó sin errores.
Diagrama E13 inspeccionado visualmente, con dimensiones tomadas del modelo.

El notebook ejecutado `notebooks/05_resultados_experimentos.ipynb` reúne los
resultados E02-E10, historiales locales, diagnóstico BN de E05 y cohortes.
`experimentos/resultados_registrados.json` conserva solo cifras agregadas
comunicadas, con procedencia y valores desconocidos como null; no fabrica
historiales universitarios ausentes. `src/results_notebook.py` carga/valida
summary/config/history locales y actualiza la vista sin entrenar, inferir ni
leer imágenes, pesos o identificadores. E05 Mac ya está terminado e incorporado.
`experimentos/cohortes_registradas.json` conserva snapshots agregados por cohorte
sin pacientes individuales; el notebook los usa si faltan informes locales y
señala su procedencia. El notebook tiene salidas agregadas seguras para versionar;
los informes completos siguen ignorados. No se ha hecho commit/push.
Notebook actualizado y ejecutado sin errores, con figuras inspeccionadas: 82 pruebas del
repositorio superadas. El cuaderno funciona también sin informes/dataset,
conservando cifras transcritas y snapshots agregados por cohorte, marcando sus
limitaciones. Fichas E06/E07/E08/E09/E10 ya reflejan resultados observados;
no confundir las instrucciones históricas de ejecución con una tarea pendiente.

E08 y E10 ya se han entrenado completos en el Mac del estudiante con MPS,
float32, PyTorch 2.11.0 y Python 3.13.5. E08: mejor epoca 3, AUC 0,582661,
AP 0,355411, 402,88 s; E10: mejor epoca 6, AUC 0,590625, AP 0,379061,
198,61 s. Ninguno mejora el resultado observado de E05 universitario; cambiar
dispositivo/precision limita esa comparacion. Ambos conservan test cerrado.
Se ha preparado `configs/experiments/E05_mac_reference.json`: replica exacta
de los ajustes E05, con identidad y rutas nuevas. El estudiante completó esa
ejecución en MPS: diez épocas, mejor época 1, AUC 0,586895 y AP 0,374219,
414,01 segundos. Sensibilidad cero a umbral 0,5. No lanzar ni sobrescribir
ejecuciones completas sin acordarlo.
`src/analyze_cohorts.py` analiza CSV del mejor checkpoint sin leer PNG ni pesos,
valida contra el fold completo y anota cohortes por paciente, con metricas,
bootstrap e intervalos pareados. Informe E08/E10 en `reports/cohorts/E08_E10_mac/`:
delta AUC E10-E08 0,007964, IC [-0,078183; 0,079259]; no ventaja clara.
Duke 42 pacientes (11 positivas), spy1 21 (4), spy2 156 (49). No interpretar
subgrupos como evidencia causal. Protocolo en
`experimentos/COHORTES_Y_REFERENCIA_MAC.md`; 77 pruebas superadas. Cambios
locales, sin commit/push. Informe completo ya creado en
`reports/cohorts/E05_E08_E10_mac/`: E08-E05 delta −0,004234 con IC
[−0,040867; 0,032825]; E10-E05 delta +0,003730 con IC [−0,076872; 0,077167].
Ninguna ventaja global clara; mantener incertidumbre y cautela en subgrupos.
Tras actualizar registros, el estudiante retomó las pruebas y autorizó preparar
E11/E12 de learning rate, ya completados. Después autorizó preparar E13 de
realce explícito. No inferir permiso para más variantes ni entrenamientos largos.

E10 reduce solo anchura de E05 a 8/16/32/64: 73.913 parametros, ocho
convoluciones, cuatro pools, campo receptivo local 106x106. Configuracion en
`configs/experiments/E10_half_width.json`, smoke separado, ficha PNG/SVG en
`experimentos/A06_mitad_canales/` y protocolo en `experimentos/ANCHURA_E10.md`.
Las 68 pruebas y el smoke integral de dos epocas pasaron. No se ha lanzado
entrenamiento cientifico desde el agente ni hecho commit/push de E10. MPS no se detecta en el
entorno actual; medir CPU/MPS desde el entorno del estudiante antes de invertir
tiempo. Diagnostico universitario E05 epoca 10: AUC train 0,869524 frente a
validacion 0,579738; copia BN 0,590121 con intervalo de mejora que incluye cero.
Conservar E05 epoca 1 como referencia y confirmar candidatos en otros folds.

E05 y el catalogo inicial de arquitecturas se subieron en el commit 4439545.
E06, sus configuraciones, pruebas y documentacion se subieron en el commit
68267cf. Se verificaron 15 tests de arquitectura y el smoke completo sobre una
copia exacta del indice, sin cambios locales de web/evaluacion/benchmark.
E07/E08, sus fichas y pruebas se subieron en el commit 9072374, dejando fuera
cambios pendientes de web/evaluacion/benchmark. Se verificaron 22 pruebas de
arquitectura/configuracion y ambos smoke tests sobre una copia exacta del indice.
E09 se subio en el commit a7bb7ff, incluyendo solo los hunks de aumentos de
`src/train.py`, sin los de evaluacion/web. La copia exacta del indice paso sus
48 pruebas y el smoke integral. E09 ya se ha probado en GPU y no mejoro E05.
E08 ya tiene resultados compartidos y analisis local. El diagnostico ya esta subido en
c6f23d7, dejando fuera cambios pendientes de web/evaluacion/benchmark. Diagnóstico
de E05 universitario completado; no pedir repetirlo como si siguiera pendiente.
Cuando se retome la mejora, confirmar candidatos con más semillas y folds;
no prometer AUC 0,7. El benchmark corregido tambien sigue local; el tiempo real
de E05 fue aproximadamente 20 segundos por epoca.

El estudiante ha vuelto a la universidad y pidió commitear lo necesario para
continuar allí. Paquete seleccionado: código CNN/evaluación/benchmark, variantes
E10-E13, pruebas, fichas propias y notebook agregado. Web, Flask, inference/app,
datos, pesos e informes completos quedan fuera. Pasos en
`experimentos/CONTINUAR_UNIVERSIDAD.md`. Recordar el benchmark corregido
pendiente; no reinstalar PyTorch ROCm ni entrenar modelos largos automáticamente.
La copia exacta del índice pasó 96 tests, smoke E13 completo (4,20 s CPU) y
benchmark técnico corto CPU. Excluidos web, inference/app, Flask y datasets.
