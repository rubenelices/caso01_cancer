# Prediccion de pCR con DCE-MRI

Proyecto individual de Aprendizaje Automatico para predecir la respuesta
patologica completa (pCR) a partir de tres fases de una DCE-MRI previa al
tratamiento: PRE, EARLY y LATE.

La solucion debe emplear una CNN 2D disenada y entrenada completamente desde
cero. No se permite transfer learning ni el uso de arquitecturas o pesos
preentrenados.

## Documentacion del proyecto

- [Plan de trabajo](PLAN.md)
- [Contexto y reglas para asistentes](AGENTS.md)
- [Guia suministrada con los datos](breastdcedl/GUIA.md)
- [Enunciado oficial](breastdcedl/documentation/caso_breastdcedl.pdf)

El repositorio incluye una configuracion ligera de Visual Studio Code que evita
indexar continuamente las imagenes del dataset, sin ocultarlas ni impedir que el
codigo las utilice.

## Datos

El dataset contiene 38.109 imagenes PNG y se mantiene exclusivamente en local
dentro de `breastdcedl/dataset/`. Esa carpeta esta ignorada por Git. Los
metadatos publicos y la documentacion si pueden versionarse porque permiten
reproducir la estructura y los experimentos sin incluir las imagenes.

Este proyecto es exclusivamente educativo y de investigacion. No constituye un
dispositivo medico ni puede emplearse para diagnostico o decisiones terapeuticas.

## Primera fase: auditoria

La auditoria completa valida metadatos y abre las 38.109 imagenes sin modificar
los datos originales:

```bash
python -m src.audit_data
```

Durante el desarrollo puede ejecutarse una comprobacion mas rapida:

```bash
python -m src.audit_data --image-check sample --sample-size 200
```

Los resultados se generan en `reports/audit/`: informe HTML, JSON estructurado,
catalogo de fuentes y figuras descriptivas.

## Segunda fase: pipeline de datos

El smoke test del pipeline crea las particiones por paciente, carga un batch de
train y otro de validacion y comprueba forma, tipo, rango y orden secuencial:

```bash
python -m src.data
```

La funcion `src.data.load_dce_triplet` es compartida por el Dataset y por la
futura aplicacion. Acepta rutas o bytes y siempre devuelve PRE, EARLY y LATE en
un tensor `float32` de forma `[3, 256, 256]` y rango `[0, 1]`.

## Tercera fase: CNN minima

La CNN educativa y su prueba de memorizacion se ejecutan con:

```bash
python -m src.overfit_smoke
```

El recorrido explicado paso a paso esta en
`notebooks/03_cnn_minima.ipynb`. El notebook reutiliza `src/model.py` y
`src/overfit_smoke.py`; la logica del proyecto no depende de celdas ocultas ni
de su orden de ejecucion.

## Cuarta fase: arquitectura base y experimentos

La primera arquitectura candidata esta implementada desde cero en
`src/architectures.py`. Recibe `[B, 3, 256, 256]`, contiene cuatro bloques
convolucionales y genera un logit por corte. Su explicacion capa por capa,
incluidos dimensiones, parametros y campo receptivo, esta en
`notebooks/04_arquitectura_base.ipynb`.

Antes de ocupar una maquina durante horas puede hacerse un cronometraje corto:

```bash
python -m src.benchmark_training \
  --config configs/experiments/E02_base_normal.json \
  --batch-size 16 --steps 4
```

En una maquina con GPU CUDA, los dos primeros experimentos se lanzan por
separado:

```bash
python -m src.train --config configs/experiments/E02_base_normal.json --device cuda
python -m src.train --config configs/experiments/E03_base_weighted.json --device cuda
```

E02 utiliza la perdida binaria normal y E03 cambia unicamente a perdida
ponderada. Cada ejecucion guarda su configuracion resuelta, entorno, historial
por epoca, curvas y resumen en `reports/experiments/`. Los checkpoints mejor y
ultimo se guardan aparte en `checkpoints/`, que esta excluida de Git. Durante
estos experimentos solo se consulta train y validacion interna; test permanece
cerrado.

### Smoke test integral

Antes del entrenamiento largo, el ensayo general reducido se ejecuta con:

```bash
python -m src.smoke_training
```

Usa 4 pacientes de cada clase para train, 2 de cada clase para validacion y 2
epocas. Recorre carga de imagenes, forward, loss, backward, optimizador,
validacion, metricas, curvas y checkpoints. Sus cifras predictivas no miden la
calidad de la red y quedan marcadas como `scientific_result: false`.
