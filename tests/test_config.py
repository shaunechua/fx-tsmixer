from pathlib import Path

import pytest

from fx_forecasting.config import load_experiment_config


def test_load_experiment_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration root must be a mapping"):
        load_experiment_config(path)


def test_load_experiment_config_parses_committed_single_yaml() -> None:
    config = load_experiment_config(Path(__file__).parents[1] / "configs" / "experiment.yaml")

    assert config.seed == 42
    assert config.purge_business_days == 1
    assert config.arima.max_p == 2
    assert config.tsmixer.lookback == 32
    assert len(config.datasets) >= 1


def test_load_experiment_config_rejects_invalid_period_order(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "configs").mkdir(parents=True)
    config_path = project_root / "configs" / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "output_dir: outputs",
                "periods:",
                "  development_end: 2024-01-31",
                "  validation_start: 2024-01-01",
                "  validation_end: 2024-02-29",
                "  final_test_start: 2024-03-01",
                "  final_test_end: 2024-03-29",
                "purge_business_days: 1",
                "datasets:",
                "  fx_spot:",
                "    path: data/fx_spot.csv",
                "    date_column: observation_date",
                "    value_column: DEXJPUS",
                "    availability_lag: 0",
                "    availability_lag_unit: business_days",
                "    maximum_staleness_days: 0",
                'missing_values: ["", ".", "NA", "N/A"]',
                "models:",
                "  arima:",
                "    max_p: 1",
                "    max_q: 1",
                "    maximum_iterations: 10",
                "  tsmixer:",
                "    lookback: 32",
                "    hidden_size: 32",
                "    mixer_blocks: 2",
                "    dropout: 0.1",
                "    learning_rate: 0.001",
                "    weight_decay: 0.0",
                "    gradient_clip: 1.0",
                "    batch_size: 8",
                "    maximum_epochs: 2",
                "    early_stopping_patience: 1",
                "    early_stopping_min_delta: 0.0",
                "    feature_sets:",
                "      tsmixer_fx_only: [fx_return_1, fx_return_5, fx_volatility_20]",
                "signal_threshold: 0.0",
                "transaction_cost: 0.0001",
                "annualization_periods: 252",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="periods must satisfy"):
        load_experiment_config(config_path)
