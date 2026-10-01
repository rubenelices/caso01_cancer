# E13 durante 50 épocas, sin parada temprana

Petición explícita del estudiante en la universidad: probar un entrenamiento
más largo y dejar la GPU trabajando. Se conserva la arquitectura, representación,
datos y ajustes de E13; no es E14 de dropout espacial, que sigue sin implementar.

Cambios: `epochs=50`, `early_stopping_patience=51`, identidad `E13_50epochs`
y carpetas propias. La paciencia es mayor que el presupuesto, por lo que el
early stopping no puede actuar. El scheduler sigue activo y puede reducir LR.
El mejor checkpoint se elige por ROC-AUC paciente; `last.pt` conserva época 50.
Se entrena **desde cero**, no se retoma un checkpoint: el runner no ofrece resume.

Datos completos, fold 0, seed 42, batch 16, LR inicial 0,001, BCE normal,
sin aumentos. La CNN y su [diagrama E13](E13_phase_differences/README.md)
son idénticos: 294.129 parámetros y fases internas PRE, EARLY−PRE, LATE−EARLY.
Test permanece cerrado. No hay resultados científicos de esta ejecución todavía.

Desde el repositorio universitario con el entorno cancer activo, una orden
después de otra:

```bash
git pull --ff-only
```

```bash
python -m src.train --config configs/experiments/E13_50epochs.json --device cuda
```

No usar `--overwrite`. Informes en
`reports/experiments/E13_50epochs/fold_0_seed_42/`; checkpoints en
`checkpoints/E13_50epochs/fold_0_seed_42/`.
Con ~20–30 segundos por época, estimar unos 17–25 minutos más trabajo final;
el tiempo real puede variar. Mantener el equipo encendido, sin suspensión y
la sesión abierta. No es un entrenamiento en servicios externos.

El diagnóstico E13 mostró train AUC 0,7882 y validación 0,5725; más épocas
pueden agravar ese problema. Esta prueba estudia la trayectoria, no promete
mejora. Comparar MPS/float32 con ROCm/AMP añade un cambio de entorno: no atribuir
automáticamente una diferencia al presupuesto. Confirmar candidatos en otros
folds/semillas antes de concluir; consultar más épocas también amplía selección.

Uso educativo y de investigación, sin validez clínica.
