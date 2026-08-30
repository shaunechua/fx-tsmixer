from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

DAILY_COLUMNS: tuple[str, ...] = (
    "strategy",
    "scenario",
    "forecast_origin",
    "effective_position_date",
    "realized_return_date",
    "target_position",
    "effective_position",
    "realized_return",
    "realized_carry",
    "turnover",
    "transaction_cost",
    "gross_return",
    "net_return",
    "gross_equity",
    "net_equity",
    "gross_drawdown",
    "net_drawdown",
)


def create_output_directories(output_root: Path) -> dict[str, Path]:
    directories = {name: output_root / name for name in ("forecasts", "backtests", "tables")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def filter_common_forecast_dates(forecasts: pl.DataFrame, *, strict: bool) -> pl.DataFrame:
    if forecasts.filter(pl.col("training_end") >= pl.col("forecast_origin")).height > 0:
        raise ValueError("training_end must be earlier than forecast_origin.")
    if forecasts.filter(pl.col("target_date") <= pl.col("forecast_origin")).height > 0:
        raise ValueError("target_date must be later than forecast_origin.")
    models = sorted(set(forecasts.get_column("model").to_list()))
    origin_sets = [
        set(forecasts.filter(pl.col("model") == model).get_column("forecast_origin").to_list())
        for model in models
    ]
    common = set.intersection(*origin_sets) if origin_sets else set()
    filtered = forecasts.filter(pl.col("forecast_origin").is_in(sorted(common))).sort(
        ["model", "forecast_origin"]
    )
    if strict and any(len(origin_set) != len(common) for origin_set in origin_sets):
        raise ValueError("Final-test model forecasts must share identical forecast origins.")
    return filtered


def forecast_metrics(forecasts: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for (model,), frame in forecasts.group_by(["model"], maintain_order=True):
        predicted_level = frame.get_column("predicted_level").to_numpy()
        realized_level = frame.get_column("realized_level").to_numpy()
        predicted_return = frame.get_column("predicted_return").to_numpy()
        realized_return = frame.get_column("realized_return").to_numpy()
        level_errors = predicted_level - realized_level
        return_errors = predicted_return - realized_return
        rows.append(
            {
                "model": model,
                "observation_count": frame.height,
                "level_mae": float(np.mean(np.abs(level_errors))),
                "level_rmse": float(np.sqrt(np.mean(np.square(level_errors)))),
                "return_mae": float(np.mean(np.abs(return_errors))),
                "return_rmse": float(np.sqrt(np.mean(np.square(return_errors)))),
                "directional_accuracy": float(
                    np.mean(np.sign(predicted_return) == np.sign(realized_return))
                ),
            }
        )
    return pl.DataFrame(rows).sort("model")


def generate_signals(forecasts: pl.DataFrame, threshold: float) -> pl.DataFrame:
    # Same-day execution: the position is opened at the forecast origin's noon rate
    # and earns the forecast target's return and carry.
    position = (
        pl.when(pl.col("predicted_return") > threshold)
        .then(1)
        .when(pl.col("predicted_return") < -threshold)
        .then(-1)
        .otherwise(0)
        .cast(pl.Int64)
    )
    return forecasts.select(
        "forecast_origin",
        "target_date",
        pl.col("forecast_origin").alias("signal_date"),
        pl.col("forecast_origin").alias("effective_position_date"),
        pl.col("target_date").alias("realized_return_date"),
        "horizon",
        "model",
        "refit_id",
        position.alias("target_position"),
        "predicted_return",
        "realized_return",
        "realized_carry",
    ).sort(["model", "realized_return_date"])


def run_backtests(signals: pl.DataFrame, transaction_cost: float) -> pl.DataFrame:
    realized_dates = [
        tuple(frame.get_column("realized_return_date").to_list())
        for _, frame in signals.group_by(["model"], maintain_order=True)
    ]
    if len(set(realized_dates)) != 1:
        raise ValueError("All model strategies must use identical realized return dates.")
    if signals.get_column("realized_carry").null_count() > 0:
        raise ValueError("Carry is missing on some backtest dates; check us_rate and jp_rate.")
    benchmark_base = (
        signals.select("realized_return_date", "realized_return", "realized_carry")
        .unique()
        .sort("realized_return_date")
        .with_columns(
            pl.lit(None, dtype=pl.Date).alias("forecast_origin"),
            pl.lit(None, dtype=pl.Date).alias("effective_position_date"),
        )
    )
    strategies = [
        (str(model), frame) for (model,), frame in signals.group_by(["model"], maintain_order=True)
    ]
    for strategy, position in (("buy_and_hold", 1), ("cash", 0)):
        strategies.append(
            (strategy, benchmark_base.with_columns(pl.lit(position).alias("target_position")))
        )
    frames = [
        _run_strategy(frame, strategy=strategy, scenario=scenario, cost_per_unit=cost)
        for strategy, frame in strategies
        for scenario, cost in (("zero_cost", 0.0), ("base_cost", transaction_cost))
    ]
    return pl.concat(frames).sort(["strategy", "scenario", "realized_return_date"])


def backtest_metrics(daily: pl.DataFrame, annualization_periods: int) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for (strategy, scenario), frame in daily.group_by(
        ["strategy", "scenario"], maintain_order=True
    ):
        gross = frame.get_column("gross_return").to_numpy()
        net = frame.get_column("net_return").to_numpy()
        effective = frame.get_column("effective_position").to_numpy()
        rows.append(
            {
                "strategy": strategy,
                "scenario": scenario,
                "observation_count": frame.height,
                "gross_cumulative_return": float(frame.item(frame.height - 1, "gross_equity"))
                - 1.0,
                "net_cumulative_return": float(frame.item(frame.height - 1, "net_equity")) - 1.0,
                "gross_annualized_return": _annualized_return(gross, annualization_periods),
                "net_annualized_return": _annualized_return(net, annualization_periods),
                "gross_annualized_volatility": _annualized_volatility(gross, annualization_periods),
                "net_annualized_volatility": _annualized_volatility(net, annualization_periods),
                "gross_sharpe_ratio": _sharpe(gross, annualization_periods),
                "net_sharpe_ratio": _sharpe(net, annualization_periods),
                "gross_maximum_drawdown": float(
                    np.min(frame.get_column("gross_drawdown").to_numpy())
                ),
                "net_maximum_drawdown": float(np.min(frame.get_column("net_drawdown").to_numpy())),
                "gross_hit_rate_while_invested": _hit_rate(gross, effective),
                "net_hit_rate_while_invested": _hit_rate(net, effective),
                "total_turnover": float(frame.get_column("turnover").sum()),
                "trade_count": int((frame.get_column("turnover") > 0.0).sum()),
                "gross_final_equity": float(frame.item(frame.height - 1, "gross_equity")),
                "net_final_equity": float(frame.item(frame.height - 1, "net_equity")),
            }
        )
    return pl.DataFrame(rows).sort(["strategy", "scenario"])


def _run_strategy(
    signals: pl.DataFrame, *, strategy: str, scenario: str, cost_per_unit: float
) -> pl.DataFrame:
    ordered = signals.sort("realized_return_date")
    target = ordered.get_column("target_position").cast(pl.Float64).to_numpy()
    realized = ordered.get_column("realized_return").to_numpy()
    carry = ordered.get_column("realized_carry").to_numpy()
    prior_target = np.concatenate((np.array([0.0]), target[:-1]))
    turnover = np.abs(target - prior_target)
    # Log excess returns are converted to simple returns so equity can compound with cumprod.
    gross = np.expm1(target * (realized + carry))
    costs = turnover * cost_per_unit
    net = gross - costs
    gross_equity = np.cumprod(1.0 + gross)
    net_equity = np.cumprod(1.0 + net)
    gross_peaks = np.maximum.accumulate(np.concatenate((np.array([1.0]), gross_equity)))[1:]
    net_peaks = np.maximum.accumulate(np.concatenate((np.array([1.0]), net_equity)))[1:]
    return pl.DataFrame(
        {
            "strategy": [strategy] * ordered.height,
            "scenario": [scenario] * ordered.height,
            "forecast_origin": ordered["forecast_origin"],
            "effective_position_date": ordered["effective_position_date"],
            "realized_return_date": ordered["realized_return_date"],
            "target_position": target,
            "effective_position": target,
            "realized_return": realized,
            "realized_carry": carry,
            "turnover": turnover,
            "transaction_cost": costs,
            "gross_return": gross,
            "net_return": net,
            "gross_equity": gross_equity,
            "net_equity": net_equity,
            "gross_drawdown": gross_equity / gross_peaks - 1.0,
            "net_drawdown": net_equity / net_peaks - 1.0,
        },
        schema=DAILY_COLUMNS,
    )


def _annualized_return(returns: np.ndarray, periods: int) -> float:
    equity = float(np.prod(1.0 + returns))
    return math.pow(equity, periods / len(returns)) - 1.0


def _annualized_volatility(returns: np.ndarray, periods: int) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * math.sqrt(periods))


def _sharpe(returns: np.ndarray, periods: int) -> float:
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    if volatility == 0.0:
        return 0.0
    return float(np.mean(returns) / volatility * math.sqrt(periods))


def _hit_rate(returns: np.ndarray, effective: np.ndarray) -> float:
    invested = effective != 0.0
    if not invested.any():
        return 0.0
    return float(np.count_nonzero(returns[invested] > 0.0) / np.count_nonzero(invested))
