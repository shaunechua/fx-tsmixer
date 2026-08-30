import math
from datetime import date

import polars as pl
import pytest

from fx_forecasting.evaluation import backtest_metrics, generate_signals, run_backtests


def _forecasts(
    predicted_returns: list[float],
    realized_returns: list[float],
    model: str = "ridge",
    realized_carry: list[float] | None = None,
) -> pl.DataFrame:
    origins = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    targets = [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8)]
    levels = [100.0, 101.0, 102.0, 103.0]
    return pl.DataFrame(
        {
            "forecast_origin": origins,
            "target_date": targets,
            "horizon": [1] * 4,
            "model": [model] * 4,
            "target_kind": ["return"] * 4,
            "training_start": [date(2023, 1, 2)] * 4,
            "training_end": [date(2023, 12, 29)] * 4,
            "predicted_level": levels,
            "predicted_return": predicted_returns,
            "realized_level": levels,
            "realized_return": realized_returns,
            "realized_carry": realized_carry or [0.0] * 4,
            "refit_id": ["refit"] * 4,
        }
    )


def test_signal_generation_executes_same_day_and_earns_the_forecast_target_return() -> None:
    signals = generate_signals(_forecasts([0.1, -0.1, 0.0, 0.2], [0.01, -0.02, 0.03, 0.04]), 0.0)

    assert signals.get_column("target_position").to_list() == [1, -1, 0, 1]
    assert signals.get_column("effective_position_date").to_list() == [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
    ]
    assert signals.get_column("realized_return_date").to_list() == [
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]
    assert signals.get_column("realized_return").to_list() == [0.01, -0.02, 0.03, 0.04]


def test_backtest_matches_buy_and_hold_and_reversal_turnover() -> None:
    forecasts = pl.concat(
        [
            _forecasts([0.1, 0.1, 0.1, 0.1], [0.01, -0.02, 0.03, 0.04], "ridge"),
            _forecasts([0.1, 0.1, 0.1, 0.1], [0.01, -0.02, 0.03, 0.04], "arima"),
        ],
        how="vertical_relaxed",
    )
    signals = generate_signals(forecasts, 0.0)
    daily = run_backtests(signals, 0.01)
    ridge = daily.filter((pl.col("strategy") == "ridge") & (pl.col("scenario") == "zero_cost"))
    benchmark = daily.filter(
        (pl.col("strategy") == "buy_and_hold") & (pl.col("scenario") == "zero_cost")
    )

    assert (
        ridge.get_column("gross_return").to_list() == benchmark.get_column("gross_return").to_list()
    )
    assert ridge.get_column("turnover").to_list() == [1.0, 0.0, 0.0, 0.0]


def test_backtest_costs_do_not_improve_performance_and_metrics_include_trade_count() -> None:
    forecasts = pl.concat(
        [
            _forecasts([0.1, -0.1, -0.1, 0.1], [0.01, 0.02, -0.03, 0.04], "ridge"),
            _forecasts([0.1, -0.1, -0.1, 0.1], [0.01, 0.02, -0.03, 0.04], "arima"),
        ],
        how="vertical_relaxed",
    )
    signals = generate_signals(forecasts, 0.0)
    daily = run_backtests(signals, 0.01)
    metrics = backtest_metrics(daily, 252)
    ridge = daily.filter((pl.col("strategy") == "ridge") & (pl.col("scenario") == "base_cost"))
    ridge_metrics = metrics.filter(
        (pl.col("strategy") == "ridge") & (pl.col("scenario") == "base_cost")
    ).row(0, named=True)

    assert (ridge.get_column("net_return") <= ridge.get_column("gross_return")).all()
    assert ridge.get_column("turnover").to_list()[1] == 2.0
    assert ridge_metrics["trade_count"] >= 1


def test_backtest_rejects_misaligned_model_dates() -> None:
    forecasts = pl.concat(
        [
            _forecasts([0.1, 0.1, 0.1, 0.1], [0.01, 0.02, 0.03, 0.04], "ridge"),
            _forecasts([0.1, 0.1, 0.1, 0.1], [0.01, 0.02, 0.03, 0.04], "arima").head(3),
        ],
        how="vertical_relaxed",
    )
    signals = generate_signals(forecasts.sort(["model", "forecast_origin"]), 0.0)

    with pytest.raises(ValueError, match="identical realized return dates"):
        run_backtests(signals, 0.01)


def test_backtest_earns_carry_when_long_and_pays_it_when_short() -> None:
    carry = [0.001, 0.002, 0.003, 0.004]
    forecasts = pl.concat(
        [
            _forecasts([0.1, 0.1, 0.1, 0.1], [0.01, -0.02, 0.03, 0.04], "long", carry),
            _forecasts([-0.1, -0.1, -0.1, -0.1], [0.01, -0.02, 0.03, 0.04], "short", carry),
        ]
    )
    daily = run_backtests(generate_signals(forecasts, 0.0), 0.0).filter(
        pl.col("scenario") == "zero_cost"
    )

    def gross(strategy: str) -> list[float]:
        return daily.filter(pl.col("strategy") == strategy).get_column("gross_return").to_list()

    long_log = [0.01 + 0.001, -0.02 + 0.002, 0.03 + 0.003, 0.04 + 0.004]
    assert gross("long") == pytest.approx([math.expm1(value) for value in long_log])
    assert gross("short") == pytest.approx([math.expm1(-value) for value in long_log])
    assert gross("buy_and_hold") == gross("long")
    assert gross("cash") == [0.0, 0.0, 0.0, 0.0]
    final_equity = daily.filter(pl.col("strategy") == "long").get_column("gross_equity")[-1]
    assert final_equity == pytest.approx(math.exp(sum(long_log)))
