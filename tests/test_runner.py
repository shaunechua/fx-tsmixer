from pathlib import Path

import polars as pl

from fx_forecasting.runner import run_final, run_validation
from tests.helpers import synthetic_project


def test_validation_runner_writes_expected_artifacts(tmp_path: Path) -> None:
    config_path = synthetic_project(tmp_path)

    outputs = run_validation(config_path)

    forecast_path = config_path.parent.parent / "outputs" / "forecasts" / "validation_forecasts.csv"
    forecasts = pl.read_csv(forecast_path, try_parse_dates=True)

    assert outputs["forecast_path"] == forecast_path
    assert outputs["metric_path"].is_file()
    assert outputs["strategy_metric_path"].is_file()
    output_root = config_path.parent.parent / "outputs"
    assert not (output_root / "tables" / "validation_frozen_tsmixer_features.csv").exists()
    assert not (output_root / "checkpoints").exists()
    assert not (output_root / "manifests").exists()
    assert not (output_root / "figures").exists()
    assert not (output_root / "tables" / "data_summary.csv").exists()
    assert not (output_root / "tables" / "feature_dictionary.csv").exists()
    assert not (output_root / "backtests" / "validation_equity.csv").exists()
    assert not (output_root / "backtests" / "validation_drawdowns.csv").exists()
    assert set(forecasts.get_column("model").to_list()) == {
        "random_walk",
        "historical_mean",
        "arima",
        "tsmixer",
        "tsmixer_with_rates",
    }


def test_validation_runner_is_unchanged_by_final_test_rows(tmp_path: Path) -> None:
    baseline_config = synthetic_project(tmp_path / "one", final_multiplier=1.0)
    altered_config = synthetic_project(tmp_path / "two", final_multiplier=3.0)

    run_validation(baseline_config)
    run_validation(altered_config)

    left = pl.read_csv(
        baseline_config.parent.parent / "outputs" / "forecasts" / "validation_forecasts.csv"
    )
    right = pl.read_csv(
        altered_config.parent.parent / "outputs" / "forecasts" / "validation_forecasts.csv"
    )

    assert left.equals(right)


def test_final_runner_writes_final_test_forecasts(tmp_path: Path) -> None:
    config_path = synthetic_project(tmp_path)

    outputs = run_final(config_path)

    assert outputs["forecast_path"].is_file()
