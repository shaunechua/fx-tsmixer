from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from fx_forecasting.config import DatasetConfig, ExperimentConfig


@dataclass(frozen=True)
class LoadedDataset:
    config: DatasetConfig
    frame: pl.DataFrame


def load_all_datasets(config: ExperimentConfig) -> dict[str, LoadedDataset]:
    return {
        dataset.name: load_csv_dataset(dataset, config.missing_values)
        for dataset in config.datasets
    }


def load_csv_dataset(config: DatasetConfig, missing_values: tuple[str, ...]) -> LoadedDataset:
    raw = pl.read_csv(
        config.path,
        infer_schema=False,
        null_values=list(missing_values),
        try_parse_dates=False,
    )
    frame = raw.select(
        pl.col(config.date_column).str.to_date(strict=True).alias("observation_date"),
        pl.col(config.value_column)
        .str.strip_chars()
        .cast(pl.Float64, strict=True)
        .alias(config.name),
    )
    if frame.get_column("observation_date").null_count() > 0:
        raise ValueError(f"Dataset {config.name} contains missing dates.")
    duplicate_count = frame.height - frame.get_column("observation_date").n_unique()
    if duplicate_count > 0:
        raise ValueError(
            f"Dataset {config.name} contains {duplicate_count} duplicate observation dates."
        )
    if not frame.get_column("observation_date").is_sorted():
        raise ValueError(f"Dataset {config.name} observation dates are not chronological.")
    return LoadedDataset(config=config, frame=frame)


def build_tradable_calendar(fx_spot: LoadedDataset) -> pl.DataFrame:
    calendar = fx_spot.frame.filter(pl.col("fx_spot").is_not_null()).select(
        pl.col("observation_date").alias("forecast_origin"),
        pl.col("fx_spot"),
    )
    if calendar.is_empty():
        raise ValueError("fx_spot has no valid observations for the tradable calendar.")
    return calendar


def align_feature_to_calendar(calendar: pl.DataFrame, feature: LoadedDataset) -> pl.DataFrame:
    calendar_dates = calendar.get_column("forecast_origin").to_list()
    value_column = feature.config.name
    source_column = f"{value_column}_source_observation_date"
    availability_column = f"{value_column}_availability_date"
    age_column = f"{value_column}_age_days"
    stale_column = f"{value_column}_is_stale"
    observations = feature.frame.filter(pl.col(value_column).is_not_null())
    source_dates = observations.get_column("observation_date").to_list()
    availability_dates = [
        _availability_date(
            source_date,
            lag=feature.config.availability_lag,
            lag_unit=feature.config.availability_lag_unit,
            calendar_dates=calendar_dates,
        )
        for source_date in source_dates
    ]
    available = (
        observations.with_columns(pl.Series(availability_column, availability_dates, dtype=pl.Date))
        .filter(pl.col(availability_column).is_not_null())
        .rename({"observation_date": source_column})
        .sort([availability_column, source_column])
    )
    aligned = calendar.sort("forecast_origin").join_asof(
        available,
        left_on="forecast_origin",
        right_on=availability_column,
        strategy="backward",
    )
    aligned = (
        aligned.with_columns(
            (pl.col("forecast_origin") - pl.col(source_column)).dt.total_days().alias(age_column)
        )
        .with_columns(
            pl.when(pl.col(source_column).is_null())
            .then(False)
            .otherwise(pl.col(age_column) > feature.config.maximum_staleness_days)
            .alias(stale_column)
        )
        .with_columns(
            pl.when(pl.col(stale_column))
            .then(None)
            .otherwise(pl.col(value_column))
            .alias(value_column)
        )
    )
    return aligned


def build_point_in_time_panel(datasets: dict[str, LoadedDataset]) -> pl.DataFrame:
    panel = build_tradable_calendar(datasets["fx_spot"])
    feature_names: list[str] = []
    for name, dataset in datasets.items():
        if name == "fx_spot":
            continue
        panel = align_feature_to_calendar(panel, dataset)
        feature_names.append(name)
    validate_point_in_time_alignment(panel, tuple(feature_names))
    return panel


def validate_point_in_time_alignment(frame: pl.DataFrame, feature_names: tuple[str, ...]) -> None:
    for feature_name in feature_names:
        source_column = f"{feature_name}_source_observation_date"
        availability_column = f"{feature_name}_availability_date"
        missing = {"forecast_origin", feature_name, source_column, availability_column}.difference(
            frame.columns
        )
        if missing:
            raise ValueError(f"Cannot validate {feature_name}; missing columns: {sorted(missing)}.")
        if (
            frame.filter(
                pl.col(feature_name).is_not_null()
                & (pl.col(source_column).is_null() | pl.col(availability_column).is_null())
            ).height
            > 0
        ):
            raise ValueError(f"Feature {feature_name} has values without point-in-time provenance.")
        if frame.filter(pl.col(source_column) > pl.col("forecast_origin")).height > 0:
            raise ValueError(
                f"Feature {feature_name} has a source observation after its forecast origin."
            )
        if frame.filter(pl.col(availability_column) > pl.col("forecast_origin")).height > 0:
            raise ValueError(f"Feature {feature_name} is used before its availability date.")


def build_model_frame(panel: pl.DataFrame) -> pl.DataFrame:
    frame = panel.with_columns(pl.col("fx_spot").log().alias("log_fx_spot"))
    frame = frame.with_columns(
        (pl.col("log_fx_spot") - pl.col("log_fx_spot").shift(1)).alias("fx_return_1"),
        (pl.col("log_fx_spot") - pl.col("log_fx_spot").shift(5)).alias("fx_return_5"),
    ).with_columns(
        pl.col("fx_return_1").rolling_std(window_size=20, min_samples=20).alias("fx_volatility_20")
    )
    frame = _add_market_features(frame)
    frame = _add_rate_features(frame)
    frame = _add_macro_features(frame)
    frame = frame.with_columns(
        pl.col("forecast_origin").shift(-1).alias("target_date"),
        pl.col("fx_spot").shift(-1).alias("target_level"),
        (pl.col("fx_spot").shift(-1).log() - pl.col("fx_spot").log()).alias("target_return"),
    )
    frame = frame.with_columns(
        (
            (pl.col("us_rate") - pl.col("jp_rate"))
            / 100.0
            * (pl.col("target_date") - pl.col("forecast_origin")).dt.total_days()
            / 365.0
        ).alias("target_carry")
    )
    return frame.drop_nulls(["fx_spot", "target_date", "target_level", "target_return"])


def _add_market_features(frame: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for dataset in ("nikkei", "wti"):
        if dataset in frame.columns:
            expressions.append(
                (pl.col(dataset).log() - pl.col(dataset).shift(1).log()).alias(
                    f"{dataset}_return_1"
                )
            )
    if "vix" in frame.columns:
        expressions.append((pl.col("vix") - pl.col("vix").shift(1)).alias("vix_change_1"))
    return frame if not expressions else frame.with_columns(*expressions)


def _add_rate_features(frame: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for rate in ("us_rate", "jp_rate"):
        if rate in frame.columns:
            expressions.append((pl.col(rate) - pl.col(rate).shift(1)).alias(f"{rate}_change_1"))
    if "us_curve_10y2y" in frame.columns:
        expressions.append(
            (pl.col("us_curve_10y2y") - pl.col("us_curve_10y2y").shift(1)).alias(
                "us_curve_10y2y_change_1"
            )
        )
    frame = frame if not expressions else frame.with_columns(*expressions)
    if {"us_rate", "jp_rate"}.issubset(frame.columns):
        frame = frame.with_columns(
            (pl.col("us_rate") - pl.col("jp_rate")).alias("rate_differential")
        )
        frame = frame.with_columns(
            (pl.col("rate_differential") - pl.col("rate_differential").shift(1)).alias(
                "rate_differential_change_1"
            )
        )
    return frame


def _add_macro_features(frame: pl.DataFrame) -> pl.DataFrame:
    for dataset_name, feature_name in (
        ("us_cpi", "us_cpi_growth"),
        ("us_indpro", "us_indpro_growth"),
    ):
        source_column = f"{dataset_name}_source_observation_date"
        if {dataset_name, source_column}.issubset(frame.columns):
            frame = frame.with_columns(
                _release_safe_log_growth(dataset_name, source_column).alias(feature_name)
            )
    return frame


def _release_safe_log_growth(value_column: str, source_column: str) -> pl.Expr:
    is_new_release = pl.col(source_column).is_not_null() & (
        pl.col(source_column).shift(1).is_null()
        | (pl.col(source_column) != pl.col(source_column).shift(1))
    )
    release_growth = pl.when(is_new_release).then(
        pl.col(value_column).log() - pl.col(value_column).shift(1).log()
    )
    return (
        pl.when(pl.col(value_column).is_not_null())
        .then(release_growth.forward_fill())
        .otherwise(None)
    )


def _availability_date(
    source_date: date,
    *,
    lag: int,
    lag_unit: str,
    calendar_dates: list[date],
) -> date | None:
    if lag_unit == "calendar_days":
        return source_date + timedelta(days=lag)
    index = (
        bisect_left(calendar_dates, source_date)
        if lag == 0
        else bisect_right(calendar_dates, source_date) + lag - 1
    )
    if index >= len(calendar_dates):
        return None
    return calendar_dates[index]
