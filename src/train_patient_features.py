"""Ensayo pareado de cabeza no lineal: probabilidades vs features por paciente.

Runner separado para no cambiar ejecuciones antiguas ni fuentes de E55 activo.
Monitor prefijado: AUC por paciente de MEDIA DE PROBABILIDADES de cortes solos.
La bolsa completa se informa como secundaria: no cambiar selección tras ver picos.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.architectures import parameter_breakdown, trainable_parameter_count
from src.data import (LoaderConfig, create_dataloaders, limit_splits_for_debug,
                      load_samples, split_by_patient_fold)
from src.experiment_config import experiment_config_from_dict
from src.learning_rate_schedule import LearningRateSchedule
from src.metrics import binary_metrics, cut_and_patient_metrics, aggregate_by_patient
from src.patient_training import pool_patient_logits
from src.patient_feature_model import PatientFeatureNet, group_features
from src.train import set_reproducibility, _environment, _write_history, _history_row
from src.training_report import save_training_dashboard
from src.evaluation import (EvaluationConfig, build_evaluation_summary,
                            save_evaluation_figure)


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    if (spec.get('schema') != 'patient_features_v1'
            or spec.get('pool_stage') not in {'probabilities', 'features'}):
        raise ValueError('Esquema o punto de agregación inválido')
    cfg = experiment_config_from_dict(spec['experiment'])
    if (cfg.training.objective != 'patient_mean_probability'
            or cfg.data.train_patients_per_batch != 2
            or cfg.training.mixed_precision or cfg.training.loss != 'normal'
            or cfg.training.ema.enabled or cfg.training.sam.enabled
            or cfg.training.mixup.enabled or cfg.training.label_smoothing.enabled
            or cfg.data.augmentation.enabled or cfg.data.intensity_augmentation.enabled
            or cfg.data.horizontal_flip.enabled
            or cfg.training.monitor != 'patient_roc_auc'):
        raise ValueError('Protocolo prefijado: dos pacientes, BCE normal, float32, sin aumentos')
    if (cfg.data.train_patient_fraction != 1 or cfg.model.input_representation != 'pre_differences'
            or cfg.model.input_normalization != 'none'):
        raise ValueError('No añadir selección o representación diferente a este ensayo')
    return spec, cfg


def train_epoch(model, loader, optimizer, device, stage):
    model.train()
    total, count = 0., 0
    criterion = nn.BCEWithLogitsLoss()
    for batch in loader:
        x, y = batch['image'].to(device), batch['target'].to(device)
        optimizer.zero_grad(set_to_none=True)
        if stage == 'features':
            logits, labels, _ = model.forward_patients(x, y, batch['patient_id'])
        else:
            logits, labels = pool_patient_logits(model(x), y, batch['patient_id'])
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def evaluate_features(model, loader, device, threshold):
    """Agrupa DESPUÉS de recorrer todo val: un lote nunca define una bolsa."""
    model.eval()
    features, labels, pids, sids, indices, cut_logits = [], [], [], [], [], []
    for batch in loader:
        f = model.extract_features(batch['image'].to(device))
        features.append(f.cpu())
        labels.extend(batch['target'].tolist())
        pids.extend(batch['patient_id'])
        sids.extend(batch['sample_id'])
        indices.extend(batch['slice_index'].tolist())
        cut_logits.append(model.classify_features(f).cpu())
    y = torch.tensor(labels, device=device)
    f = torch.cat(features).to(device)
    cut_z = torch.cat(cut_logits).to(device)
    means, bag_y, bag_ids = group_features(f, y, pids)
    bag_z = model.classify_features(means)
    single_z, single_y = pool_patient_logits(cut_z, y, pids)
    probs = torch.sigmoid(cut_z).cpu().tolist()
    metrics = cut_and_patient_metrics(pids, labels, probs, threshold, aggregation='mean')
    bag_probs = torch.sigmoid(bag_z).cpu().tolist()
    bag_metrics = binary_metrics(bag_y.cpu().tolist(), bag_probs, threshold)
    cut_frame = pd.DataFrame(dict(sample_id=sids, patient_id=pids, slice_index=indices,
                                 target=labels, probability=probs))
    bag_frame = pd.DataFrame(dict(patient_id=bag_ids, target=bag_y.cpu().tolist(), probability=bag_probs))
    counts = pd.Series(pids).value_counts()
    bag_frame['cuts'] = bag_frame.patient_id.map(counts)
    single_loss = float(nn.functional.binary_cross_entropy_with_logits(single_z, single_y).cpu())
    bag_loss = float(nn.functional.binary_cross_entropy_with_logits(bag_z, bag_y).cpu())
    return single_loss, bag_loss, metrics, bag_metrics, cut_frame, bag_frame


def central_cut_predictions(cuts):
    """Un corte por paciente: mediana superior axial, sin consultar etiquetas/scores."""
    ordered = cuts.sort_values(['patient_id', 'slice_index', 'sample_id'])
    indices = [group.index[len(group) // 2] for _, group in ordered.groupby('patient_id', sort=True)]
    return ordered.loc[indices].reset_index(drop=True)


def run(path, device_name):
    spec, cfg = load_spec(path)
    device = torch.device(device_name)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS no disponible; no hacer fallback silencioso a CPU')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA/ROCm no disponible')
    if device.type not in {'cpu', 'mps', 'cuda'}:
        raise ValueError('Dispositivo no admitido')
    out, weights = Path(cfg.output_dir), Path(cfg.checkpoint_dir)
    for folder, prefix in ((out, 'reports'), (weights, 'checkpoints')):
        if folder.is_absolute() or '..' in folder.parts or not folder.is_relative_to(prefix):
            raise ValueError('Ruta fuera de reports/checkpoints')
        if folder.exists():
            raise FileExistsError(f'Conservar ejecución existente: {folder}; sin overwrite ni resume')
    set_reproducibility(cfg.training.seed)
    splits = limit_splits_for_debug(split_by_patient_fold(load_samples(cfg.data.root), cfg.data.validation_fold),
        cfg.data.train_patients_per_class, cfg.data.validation_patients_per_class, cfg.training.seed)
    loaders = create_dataloaders(splits, cfg.data.root, LoaderConfig(
        batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
        seed=cfg.training.seed, train_patients_per_batch=2))
    model = PatientFeatureNet(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)
    scheduler = LearningRateSchedule(optimizer, cfg.training)
    out.mkdir(parents=True)
    weights.mkdir(parents=True)
    (out / 'config_resolved.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    environment = _environment(device)
    source_names = ['patient_feature_model.py', 'train_patient_features.py', 'architectures.py',
                    'patient_training.py', 'data.py', 'phase_representation.py', 'experiment_config.py']
    environment['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in source_names}
    environment['input_config_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (out / 'environment.json').write_text(json.dumps(environment, indent=2))
    history, best, best_epoch, stale = [], -float('inf'), 0, 0
    start = time.perf_counter()
    for epoch in range(1, cfg.training.epochs + 1):
        epoch_start = time.perf_counter()
        lr = scheduler.start_epoch(epoch)
        train_loss = train_epoch(model, loaders.train, optimizer, device, spec['pool_stage'])
        single_loss, bag_loss, metrics, bag_metrics, _, _ = evaluate_features(
            model, loaders.validation, device, cfg.training.threshold)
        score = metrics['patient']['roc_auc']
        scheduler.finish_epoch(score)
        elapsed = time.perf_counter() - epoch_start
        # Val loss aquí es BCE por paciente en modo corte solo/media probabilidades.
        row = _history_row(epoch, train_loss, single_loss, lr, elapsed, metrics)
        row.update(bag_patient_roc_auc=bag_metrics['roc_auc'], bag_val_loss=bag_loss)
        history.append(row)
        _write_history(out / 'history.csv', history)
        checkpoint = dict(model_class='PatientFeatureNet_v1', epoch=epoch, config=spec,
            model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(), validation_metrics=metrics,
            bag_patient_metrics_secondary=bag_metrics)
        torch.save(checkpoint, weights / 'last.pt')
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save(checkpoint, weights / 'best.pt')
        else:
            stale += 1
        print(f'epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={single_loss:.4f} '
              f'patient_roc_auc={score:.4f} bag_auc_secondary={bag_metrics["roc_auc"]:.4f} '
              f'lr={lr:g} time={elapsed:.1f}s', flush=True)
        if stale >= cfg.training.early_stopping_patience:
            print('Early stopping: modo corte solo/media por paciente sin mejora.', flush=True)
            break
    total = time.perf_counter() - start
    checkpoint = torch.load(weights / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    single_loss, bag_loss, metrics, bag_metrics, cuts, bags = evaluate_features(
        model, loaders.validation, device, cfg.training.threshold)
    singles = aggregate_by_patient(cuts.patient_id, cuts.target, cuts.probability)
    central = central_cut_predictions(cuts)
    cuts.to_csv(out / 'predictions_cut.csv', index=False)
    singles.to_csv(out / 'predictions_patient.csv', index=False)
    bags.to_csv(out / 'predictions_patient_bag_secondary.csv', index=False)
    central.to_csv(out / 'predictions_central_cut_secondary.csv', index=False)
    ecfg = EvaluationConfig(threshold=cfg.training.threshold, seed=cfg.training.seed)
    evaluation = build_evaluation_summary(cuts, singles, ecfg)
    (out / 'evaluation_summary.json').write_text(json.dumps(evaluation, indent=2))
    save_evaluation_figure(singles, evaluation, ecfg, out / 'evaluation_patient.png')
    save_training_dashboard(history, metrics, best_epoch, out / 'training_curves.png')
    debug = cfg.data.train_patients_per_class is not None or cfg.data.validation_patients_per_class is not None
    summary = dict(status='complete', experiment_id=cfg.experiment_id, scientific_result=not debug,
        debug_patient_subset=debug, test_evaluated=False, best_epoch=best_epoch,
        epochs_completed=len(history), total_seconds=total, parameter_count=trainable_parameter_count(model),
        parameter_breakdown=parameter_breakdown(model), environment=environment,
        patients=splits.patient_counts(), samples=splits.sample_counts(),
        loss_unit='patient', training_pool_stage=spec['pool_stage'],
        monitor='patient_roc_auc_single_cut_probabilities_mean', best_monitor_value=best,
        validation_loss=single_loss, bag_validation_loss_secondary=bag_loss,
        validation_metrics=metrics, bag_patient_metrics_secondary=bag_metrics,
        central_cut_patient_metrics_secondary=binary_metrics(central.target, central.probability, cfg.training.threshold),
        model_class='PatientFeatureNet_v1', web_checkpoint_loader_compatible=False,
        selection_note='patient_roc_auc usa media de probabilidades de cortes individuales; '
                       'no equivale a evaluar una única muestra por paciente',
        central_cut_selection='upper_median_slice_index_then_sample_id_without_labels_or_scores',
        note='Cabeza no lineal y lotes agrupados: no atribuir delta vs E19 solo a pooling. '
             'Modo bolsa secundaria, no sustituye evaluación del modo de un corte de la defensa.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', required=True, choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.device), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
