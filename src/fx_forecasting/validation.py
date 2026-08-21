from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import polars as pl

from fx_forecasting.config import PeriodsConfig


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    refit_id: str
    training_start: date
    training_end: date
    purge_start: date
    purge_end: date
    validation_start: date
    validation_end: date
    training_observations: int
    purge_observations: int
    validation_observations: int


def development_validation_frame(model_frame: pl.DataFrame, periods: PeriodsConfig) -> pl.DataFrame:
    return model_frame.filter(
        (
            (pl.col("forecast_origin") <= periods.development_end)
            | pl.col("forecast_origin").is_between(
                periods.validation_start,
                periods.validation_end,
                closed="both",
            )
        )
        & (pl.col("target_date") < periods.final_test_start)
    )


def final_test_frame(model_frame: pl.DataFrame, periods: PeriodsConfig) -> pl.DataFrame:
    return model_frame.filter(
        pl.col("forecast_origin").is_between(
            periods.final_test_start,
            periods.final_test_end,
            closed="both",
        )
        & (pl.col("target_date") <= periods.final_test_end)
    )


def build_validation_folds(
    tuning_frame: pl.DataFrame,
    periods: PeriodsConfig,
    purge_observations: int,
) -> tuple[WalkForwardFold, ...]:
    if tuning_frame.filter(pl.col("forecast_origin") > periods.validation_end).height > 0:
        raise ValueError("Validation fold generation must not receive final-test rows.")
    return _build_monthly_folds(
        tuning_frame,
        start=periods.validation_start,
        end=periods.validation_end,
        purge_observations=purge_observations,
    )


def build_final_test_folds(
    full_frame: pl.DataFrame,
    periods: PeriodsConfig,
    purge_observations: int,
) -> tuple[WalkForwardFold, ...]:
    if full_frame.filter(pl.col("forecast_origin") > periods.final_test_end).height > 0:
        raise ValueError("Final-test fold generation must not receive rows beyond final_test_end.")
    return _build_monthly_folds(
        full_frame,
        start=periods.final_test_start,
        end=periods.final_test_end,
        purge_observations=purge_observations,
    )


def fold_frames(frame: pl.DataFrame, fold: WalkForwardFold) -> tuple[pl.DataFrame, pl.DataFrame]:
    training = frame.filter(pl.col("forecast_origin") <= fold.training_end)
    validation = frame.filter(
        pl.col("forecast_origin").is_between(
            fold.validation_start,
            fold.validation_end,
            closed="both",
        )
    )
    return training, validation


def folds_frame(folds: tuple[WalkForwardFold, ...]) -> pl.DataFrame:
    return pl.DataFrame([asdict(fold) for fold in folds])


def _build_monthly_folds(
    frame: pl.DataFrame,
    *,
    start: date,
    end: date,
    purge_observations: int,
) -> tuple[WalkForwardFold, ...]:
    origins = frame.get_column("forecast_origin").to_list()
    validation_indices = [index for index, origin in enumerate(origins) if start <= origin <= end]
    if not validation_indices:
        raise ValueError("No model-frame rows fall in the requested validation range.")
    blocks: list[list[int]] = []
    current_month: tuple[int, int] | None = None
    for index in validation_indices:
        month = (origins[index].year, origins[index].month)
        if month != current_month:
            blocks.append([index])
            current_month = month
        else:
            blocks[-1].append(index)
    return tuple(
        _build_fold(frame, origins, block, purge_observations, fold_id)
        for fold_id, block in enumerate(blocks, start=1)
    )


def _build_fold(
    frame: pl.DataFrame,
    origins: list[date],
    block: list[int],
    purge_observations: int,
    fold_id: int,
) -> WalkForwardFold:
    validation_start_index = block[0]
    purge_start_index = validation_start_index - purge_observations
    training_end_index = purge_start_index - 1
    if training_end_index < 0:
        raise ValueError("Insufficient training rows for the configured purge gap.")
    validation_start = origins[validation_start_index]
    training = frame.slice(0, training_end_index + 1)
    if training.filter(pl.col("target_date") >= validation_start).height > 0:
        raise ValueError(
            f"Fold {fold_id} training target dates overlap its validation forecast origins."
        )
    validation_month = origins[validation_start_index]
    return WalkForwardFold(
        fold_id=fold_id,
        refit_id=f"refit-{validation_month.year}-{validation_month.month:02d}",
        training_start=origins[0],
        training_end=origins[training_end_index],
        purge_start=origins[purge_start_index],
        purge_end=origins[validation_start_index - 1],
        validation_start=validation_start,
        validation_end=origins[block[-1]],
        training_observations=training_end_index + 1,
        purge_observations=purge_observations,
        validation_observations=len(block),
    )
