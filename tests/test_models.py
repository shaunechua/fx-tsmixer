from datetime import date

import numpy as np
import polars as pl
import pytest

from fx_forecasting.config import ArimaConfig, TSMixerConfig
from fx_forecasting.models import (
    FORECAST_COLUMNS,
    generate_arima_forecasts,
    generate_baseline_forecasts,
)
from fx_forecasting.tsmixer import generate_tsmixer_forecasts
from fx_forecasting.validation import WalkForwardFold
from tests.helpers import business_dates


def _frame() -> pl.DataFrame:
    dates = business_dates(date(2023, 10, 2), 91)
    index = np.arange(90, dtype=float)
    returns = 0.001 * np.sin(index / 4.0)
    spot = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))
    return pl.DataFrame(
        {
            "forecast_origin": dates[:-1],
            "fx_spot": spot[:-1],
            "feature_a": index,
            "feature_b": np.cos(index),
            "target_date": dates[1:],
            "target_level": spot[1:],
            "target_return": returns,
            "target_carry": np.full(90, 0.0001),
            "fx_return_1": np.concatenate([[np.nan], returns[:-1]]),
            "fx_return_5": np.concatenate(
                [np.full(5, np.nan), np.cumsum(returns)[:-5] * 0 + returns[5:]]
            ),
            "fx_volatility_20": np.concatenate([np.full(20, np.nan), np.full(70, 0.01)]),
        },
        schema_overrides={"forecast_origin": pl.Date, "target_date": pl.Date},
    )


def _fold() -> WalkForwardFold:
    dates = business_dates(date(2023, 10, 2), 91)
    return WalkForwardFold(
        1,
        "refit-2024-02",
        dates[0],
        dates[59],
        dates[60],
        dates[60],
        dates[61],
        dates[89],
        60,
        1,
        29,
    )


def test_baselines_and_arima_emit_common_forecast_columns() -> None:
    frame = _frame().drop_nulls(["fx_return_1", "fx_return_5", "fx_volatility_20"])
    fold = WalkForwardFold(
        1,
        "refit-2024-02",
        frame.item(0, "forecast_origin"),
        frame.item(39, "forecast_origin"),
        frame.item(40, "forecast_origin"),
        frame.item(40, "forecast_origin"),
        frame.item(41, "forecast_origin"),
        frame.item(frame.height - 1, "forecast_origin"),
        40,
        1,
        frame.height - 41,
    )
    arima_config = ArimaConfig(1, 1, 20)

    baselines = generate_baseline_forecasts(frame, (fold,))
    arima, metadata = generate_arima_forecasts(frame, (fold,), arima_config)

    assert baselines.columns == list(FORECAST_COLUMNS)
    assert arima.columns == list(FORECAST_COLUMNS)
    assert metadata.item(0, "model") == "arima"


def test_arima_records_fallback_when_all_candidates_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame().drop_nulls(["fx_return_1", "fx_return_5", "fx_volatility_20"])
    fold = WalkForwardFold(
        1,
        "refit-2024-02",
        frame.item(0, "forecast_origin"),
        frame.item(39, "forecast_origin"),
        frame.item(40, "forecast_origin"),
        frame.item(40, "forecast_origin"),
        frame.item(41, "forecast_origin"),
        frame.item(frame.height - 1, "forecast_origin"),
        40,
        1,
        frame.height - 41,
    )
    config = ArimaConfig(1, 1, 20)

    def broken_candidate(*args: object, **kwargs: object) -> object:
        raise RuntimeError("broken")

    monkeypatch.setattr("fx_forecasting.models._fit_arima_candidate", broken_candidate)
    forecasts, metadata = generate_arima_forecasts(frame, (fold,), config)

    assert forecasts.is_empty()
    assert metadata.item(0, "forecast_rows") == 0
    assert metadata.item(0, "failure_reason") is not None


def test_tsmixer_forecasts_with_darts() -> None:
    frame = _frame().with_columns(
        pl.lit(0.01).alias("fx_volatility_20"),
        pl.col("feature_a").alias("fx_return_1"),
        pl.col("feature_b").alias("fx_return_5"),
    )
    fold = _fold()
    config = TSMixerConfig(
        lookback=32,
        hidden_size=32,
        mixer_blocks=2,
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.0001,
        gradient_clip=1.0,
        batch_size=16,
        maximum_epochs=2,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0001,
        feature_sets={"tsmixer_ab": ("feature_a", "feature_b"), "tsmixer": ()},
    )
    forecasts, metadata = generate_tsmixer_forecasts(frame, (fold,), config, seed=42)
    assert forecasts.columns == list(FORECAST_COLUMNS)
    assert forecasts.height == 2 * fold.validation_observations
    assert forecasts.get_column("predicted_return").is_finite().all()
    assert metadata.get_column("epochs_trained").to_list() == [config.maximum_epochs] * 2
    assert metadata.get_column("failure_reason").is_null().all()
    assert set(forecasts.get_column("model")) == {"tsmixer_ab", "tsmixer"}
