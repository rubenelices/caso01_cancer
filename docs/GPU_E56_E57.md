# E56/E57 en la GPU de la universidad

Primero `git status --short`. Si hay cambios locales, conservarlos y resolverlos
antes de actualizar; no usar reset/overwrite. Con el árbol limpio:

```bash
git pull --ff-only
conda activate cancer
```

Desde la raíz del repositorio, ejecutar uno por uno:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u -m src.train_patient_features --config configs/patient_features/gpu/E56_nonlinear_cut_control_cuda.json --device cuda
```

Esperar a que termine y después:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u -m src.train_patient_features --config configs/patient_features/gpu/E57_patient_feature_mean_cuda.json --device cuda
```

`cuda` es también el nombre del dispositivo PyTorch en ROCm/AMD. No reinstalar
PyTorch ni cambiar drivers. Datos locales en breastdcedl, nunca se incluyen en Git.
Dos hilos CPU para la preparación; CNN en GPU. Float32, sin AMP, como Mac.

E56 controla una cabeza no lineal128→32→1 con agregación de probabilidades;
E57 combina características antes de la misma cabeza. Ambos desde cero,
mismos lotes de dos pacientes completas y configuraciones científicas Mac,
con identidades/rutas GPU distintas. Mismo monitor AUC por paciente de media
de probabilidades de cortes solos. Bolsa completa y corte central secundarios.
Máximo50/paciencia10. No garantiza una mejora ni alcanzar0,7.

Resultados `reports/gpu/E56_nonlinear_cut_control_cuda/fold_0_seed_42` y
`reports/gpu/E57_patient_feature_mean_cuda/fold_0_seed_42`. Pesos en checkpoints
con los mismos nombres. Se rechazan carpetas existentes, sin overwrite/resume.
No mezclar resultados GPU con Mac ni elegir solo semillas favorables.

Guardar summary.json/history.csv y predicciones antes de salir de la universidad.
No se sincronizan con el Mac por hacer push del código; no versionar pesos/informes
individuales. [Protocolo detallado](../experimentos/CARACTERISTICAS_PACIENTE.md).
Checkpoints nuevos requieren loader propio; web antigua aún no integrada.

Uso educativo y de investigación, sin validez clínica.
