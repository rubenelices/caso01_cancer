"""La prueba larga cambia presupuesto/parada, no modelo ni datos."""
from dataclasses import replace

from src.experiment_config import load_experiment_config


def test_long_e13_preserves_model_data_and_optimizer():
    base = load_experiment_config("configs/experiments/E13_phase_differences.json")
    long = load_experiment_config("configs/experiments/E13_50epochs.json")
    assert long.model == base.model
    assert long.data == base.data
    assert long.training == replace(base.training, epochs=50, early_stopping_patience=51)
    assert long.experiment_id != base.experiment_id
    assert long.output_dir != base.output_dir
    assert long.checkpoint_dir != base.checkpoint_dir


def test_early_stopping_is_unreachable_even_without_improvement():
    config = load_experiment_config("configs/experiments/E13_50epochs.json")
    assert config.training.epochs == 50
    assert all(count < config.training.early_stopping_patience
               for count in range(config.training.epochs + 1))
    assert config.training.monitor == "patient_roc_auc"
