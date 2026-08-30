from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from polars.exceptions import PolarsError

from fx_forecasting.config import load_experiment_config
from fx_forecasting.data import build_model_frame, build_point_in_time_panel, load_all_datasets
from fx_forecasting.evaluation import (
    backtest_metrics,
    create_output_directories,
    filter_common_forecast_dates,
    forecast_metrics,
    generate_signals,
    run_backtests,
)
from fx_forecasting.models import generate_arima_forecasts, generate_baseline_forecasts
from fx_forecasting.tsmixer import generate_tsmixer_forecasts
from fx_forecasting.validation import (
    build_final_test_folds,
    build_validation_folds,
    development_validation_frame,
    final_test_frame,
    folds_frame,
)


def run_validation(config_path: Path) -> dict[str, Path]:
    config = load_experiment_config(config_path)
    outputs = create_output_directories(config.output_dir)
    model_frame = build_model_frame(build_point_in_time_panel(load_all_datasets(config)))
    tuning_frame = development_validation_frame(model_frame, config.periods)
    folds = build_validation_folds(tuning_frame, config.periods, config.purge_business_days)

    baseline_forecasts = generate_baseline_forecasts(tuning_frame, folds)
    arima_forecasts, arima_metadata = generate_arima_forecasts(tuning_frame, folds, config.arima)
    tsmixer_forecasts, tsmixer_metadata = generate_tsmixer_forecasts(
        tuning_frame, folds, config.tsmixer, seed=config.seed
    )

    forecasts = filter_common_forecast_dates(
        pl.concat([baseline_forecasts, arima_forecasts, tsmixer_forecasts]),
        strict=False,
    )
    signals = generate_signals(forecasts, config.signal_threshold)
    daily = run_backtests(signals, config.transaction_cost)
    metadata = pl.concat([arima_metadata, tsmixer_metadata], how="diagonal_relaxed").sort(
        ["model", "fold_id"]
    )

    forecast_path = outputs["forecasts"] / "validation_forecasts.csv"
    metric_path = outputs["tables"] / "validation_forecast_metrics.csv"
    strategy_metric_path = outputs["tables"] / "validation_strategy_metrics.csv"
    folds_frame(folds).write_csv(outputs["tables"] / "walk_forward_folds.csv")
    forecasts.write_csv(forecast_path)
    forecast_metrics(forecasts).write_csv(metric_path)
    metadata.write_csv(outputs["tables"] / "validation_model_metadata.csv")
    signals.write_csv(outputs["backtests"] / "validation_signals.csv")
    daily.write_csv(outputs["backtests"] / "validation_daily.csv")
    backtest_metrics(daily, config.annualization_periods).write_csv(strategy_metric_path)
    return {
        "forecast_path": forecast_path,
        "metric_path": metric_path,
        "strategy_metric_path": strategy_metric_path,
    }


def run_final(config_path: Path) -> dict[str, Path]:
    config = load_experiment_config(config_path)
    outputs = create_output_directories(config.output_dir)
    forecast_path = outputs["forecasts"] / "final_test_forecasts.csv"

    model_frame = build_model_frame(build_point_in_time_panel(load_all_datasets(config)))
    tuning_frame = development_validation_frame(model_frame, config.periods)
    final_only = final_test_frame(model_frame, config.periods)
    full_frame = pl.concat([tuning_frame, final_only]).sort("forecast_origin")
    folds = build_final_test_folds(full_frame, config.periods, config.purge_business_days)

    baseline_forecasts = generate_baseline_forecasts(full_frame, folds)
    arima_forecasts, arima_metadata = generate_arima_forecasts(full_frame, folds, config.arima)
    tsmixer_forecasts, tsmixer_metadata = generate_tsmixer_forecasts(
        full_frame, folds, config.tsmixer, seed=config.seed
    )
    if tsmixer_metadata.filter(pl.col("failure_reason").is_not_null()).height > 0:
        raise RuntimeError("TSMixer final-test evaluation produced failed folds.")

    forecasts = filter_common_forecast_dates(
        pl.concat([baseline_forecasts, arima_forecasts, tsmixer_forecasts]),
        strict=True,
    )
    signals = generate_signals(forecasts, config.signal_threshold)
    daily = run_backtests(signals, config.transaction_cost)
    metadata = pl.concat([arima_metadata, tsmixer_metadata], how="diagonal_relaxed").sort(
        ["model", "fold_id"]
    )

    metric_path = outputs["tables"] / "final_test_metrics.csv"
    strategy_metric_path = outputs["tables"] / "final_test_strategy_metrics.csv"
    forecasts.write_csv(forecast_path)
    forecast_metrics(forecasts).write_csv(metric_path)
    metadata.write_csv(outputs["tables"] / "final_test_model_metadata.csv")
    signals.write_csv(outputs["backtests"] / "final_test_signals.csv")
    daily.write_csv(outputs["backtests"] / "final_test_daily.csv")
    backtest_metrics(daily, config.annualization_periods).write_csv(strategy_metric_path)
    return {
        "forecast_path": forecast_path,
        "metric_path": metric_path,
        "strategy_metric_path": strategy_metric_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the USD/JPY experiment pipeline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("validation", "final"))
    arguments = parser.parse_args()
    try:
        outputs = (
            run_validation(arguments.config)
            if arguments.mode == "validation"
            else run_final(arguments.config)
        )
    except (FileNotFoundError, KeyError, PolarsError, RuntimeError, TypeError, ValueError) as error:
        sys.stderr.write(f"Experiment run failed: {error}\n")
        raise SystemExit(1) from error
    sys.stdout.write(f"Forecasts written to {outputs['forecast_path']}\n")
