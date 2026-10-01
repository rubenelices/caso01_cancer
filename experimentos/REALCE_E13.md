# E13: presentar explícitamente los cambios entre fases

## Hipótesis y referencia

La referencia es **E05 Mac**, no E12 elegido por AP: ROC-AUC paciente 0,586895,
AP 0,374219 y mejor época 1. E11/E12 no mejoraron su ROC-AUC observado.
E13 cambia una variable principal: la representación que ve la primera
convolución. No añade capas aprendidas, pacientes, aumentos ni pesos externos.

La entrada del sistema sigue siendo PRE/EARLY/LATE, tres PNG cargados juntos
en float32 [0,1]. Dentro del modelo, `PhaseRepresentation` hace:

| Canal interno | Fórmula | Rango posible | Idea |
|---|---|---|---|
| 0 | PRE | [0,1] | Mantener la imagen anterior al contraste |
| 1 | EARLY − PRE | [-1,1] | Cambio temprano de intensidad |
| 2 | LATE − EARLY | [-1,1] | Cambio posterior, incluido descenso |

Por ejemplo, un píxel con PRE=0,2, EARLY=0,7 y LATE=0,5 se convierte en
[0,2; 0,5; −0,2]. No hacemos valor absoluto, clipping, ReLU ni normalización
independiente antes de la primera convolución: perderíamos el signo.

Es una transformación lineal **invertible**: EARLY=PRE+canal1,
LATE=PRE+canal1+canal2. No añade información nueva. La primera convolución
de E05 ya podría aprender restas entre canales. La hipótesis es que hacerlas
explícitas facilite el aprendizaje con este presupuesto; también puede empeorar.
Las diferencias son intensidades de PNG normalizados, no mediciones físicas
cuantitativas de perfusión ni pruebas de respuesta futura.

## Arquitectura y compatibilidad

- Transformación: [B,3,256,256] → [B,3,256,256], cero parámetros, sin kernel,
  padding o stride; no cambia el campo receptivo.
- CNN E05: cuatro bloques Conv–BN–ReLU → MaxPool → Conv–BN–ReLU,
  canales 16/32/64/128; GAP, dropout 0,3 y una salida logit.
- Ocho convoluciones 3×3, padding 1, stride 1 y cuatro MaxPool 2×2 stride 2.
- 294.129 parámetros, mapa final 128×16×16, campo receptivo local 106×106.
- La configuración guarda `model.input_representation="pre_differences"`.
  Configuraciones anteriores sin ese campo usan `raw` y mantienen sus pesos.

La operación vive **dentro del modelo**, antes del primer bloque, no en un
script especial de entrenamiento. Train, evaluación, diagnóstico e inferencia
desde la web la ejecutan una sola vez. El usuario sigue subiendo las fases
originales; no debe preparar ni subir PNG de diferencias. No se modifica el
dataset ni se vuelven a guardar imágenes. La ReLU posterior a la convolución
actúa sobre características aprendidas, no directamente sobre las restas.

Los pesos iniciales, número de parámetros y política de inicialización se
conservan. No obstante, las escalas/correlaciones de entrada cambian, y con ellas
la optimización y las estadísticas BatchNorm. Es el efecto que medimos, no una
comparación con funciones iniciales idénticas.

## Protocolo de comparación

Igual que E05 Mac: fold 0, seed 42, 878 pacientes de train y 219 de validación,
batch 16, diez épocas, AdamW LR inicial 0,001, weight decay 0,0001, BCE normal,
sin aumentos. Mismo scheduler y parada; agregación mean y umbral 0,5.
MPS mantiene float32 como la referencia. Checkpoint elegido por ROC-AUC
paciente, no por una época favorable en otra métrica. Test permanece cerrado.

Informes: `reports/experiments/E13_phase_differences/fold_0_seed_42/`.
Pesos: `checkpoints/E13_phase_differences/fold_0_seed_42/`.
No reutilizar pesos E05 ni sobrescribir sus resultados. No pasar datos ya
transformados a un modelo E13: aplicaría las diferencias dos veces.

Si resulta prometedor, comparar predicciones sobre las mismas pacientes y
confirmar con otras semillas/folds. El fold de desarrollo ya ha sido consultado
muchas veces; una subida aislada no demuestra generalización ni garantiza 0,7.

## Ejecución

Prueba técnica pequeña (CPU, dos épocas, 8 pacientes train y 4 validación):

```bash
python -m src.smoke_training --config configs/experiments/SMOKE_phase_differences.json
```

Entrenamiento científico, cuando el estudiante decida, desde la raíz del repo:

```bash
python -m src.train --config configs/experiments/E13_phase_differences.json --device auto
```

El coste debería ser próximo a E05 Mac (~7 minutos en su ejecución), pues las
capas aprendidas son idénticas y solo se añaden dos restas. No es un benchmark
ni una garantía: temperatura/carga pueden cambiar el tiempo. No se ha lanzado
el entrenamiento completo desde el agente. Las métricas del smoke no evalúan
la calidad del modelo.

Verificación: smoke integral de dos épocas completado en CPU en 4,10 segundos,
con 80 cortes de train y 40 de validación. Se generaron curvas, predicciones,
configuración y checkpoints mejor/último; todas las comprobaciones pasaron,
incluida ausencia de evaluación de test. Las pruebas verifican restas firmadas,
inversión, gradientes, dimensiones, parámetros y mismos pesos iniciales que E05,
además de carga desde checkpoint e inferencia idéntica desde rutas y bytes.
Las 103 pruebas del repositorio pasan; notebook actualizado y ejecutado sin
errores. El diagrama E13 se ha inspeccionado visualmente.

[Ficha con capas y PNG/SVG](E13_phase_differences/README.md).

## Resultado científico observado

Ejecución del estudiante en MPS, Python 3.13.5 y PyTorch 2.11.0, diez épocas:
mejor checkpoint en época 10, ROC-AUC paciente 0,5724798387 y AP 0,3508598157.
Tiempo total 421,48 s. No supera E05 Mac (ROC-AUC 0,586895).
A umbral 0,5: TN=132, FP=23, FN=50, TP=14; sensibilidad 0,21875 y
especificidad 0,851613. Loss train final 0,527474, validación 0,656577.
El LR registrado permanece en 0,001 durante las diez épocas. Fuente:
summary.json/history.csv locales y salida compartida por el estudiante.
No se ha evaluado test. Mejor época final no implica convergencia ni garantiza
una mejora al prolongar; revisar diagnóstico train/validación antes de decidir.

Diagnóstico completado en MPS: train AUC 0,788197 frente a validación 0,572480;
copia BN validación 0,557964. Recalibrar no mejora el criterio principal.
Ver [diagnóstico E13](DIAGNOSTICO_E13.md) y snapshot agregado asociado.

Uso educativo y de investigación, sin validez clínica.
