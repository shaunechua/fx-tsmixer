from datetime import date

import polars as pl
import pytest

from fx_forecasting.config import PeriodsConfig
from fx_forecasting.validation import (
    build_final_test_folds,
    build_validation_folds,
    development_validation_frame,
    final_test_frame,
    fold_frames,
)
from tests.helpers import business_dates


def _model_frame() -> pl.DataFrame:
    dates = business_dates(date(2021, 11, 1), 120)
    return pl.DataFrame(
        {
            "forecast_origin": dates[:-1],
            "target_date": dates[1:],
            "fx_spot": [100.0 + index for index in range(len(dates) - 1)],
            "target_level": [101.0 + index for index in range(len(dates) - 1)],
            "target_return": [0.01] * (len(dates) - 1),
        },
        schema_overrides={"forecast_origin": pl.Date, "target_date": pl.Date},
    )


def _periods() -> PeriodsConfig:
    return PeriodsConfig(
        development_end=date(2021, 12, 31),
        validation_start=date(2022, 1, 3),
        validation_end=date(2022, 2, 28),
        final_test_start=date(2022, 3, 1),
        final_test_end=date(2022, 3, 31),
    )


def test_build_validation_folds_is_chronological_and_purged() -> None:
    tuning = development_validation_frame(_model_frame(), _periods())

    folds = build_validation_folds(tuning, _periods(), 1)

    assert [fold.refit_id for fold in folds] == ["refit-2022-01", "refit-2022-02"]
    assert all(
        fold.training_end < fold.purge_start <= fold.purge_end < fold.validation_start
        for fold in folds
    )
    training, validation = fold_frames(tuning, folds[0])
    assert training.filter(pl.col("target_date") >= folds[0].validation_start).is_empty()
    assert validation.height == folds[0].validation_observations


def test_final_test_frame_and_folds_stay_out_of_validation() -> None:
    model_frame = _model_frame()
    tuning = development_validation_frame(model_frame, _periods())
    final_only = final_test_frame(model_frame, _periods())
    full = pl.concat([tuning, final_only]).sort("forecast_origin")

    folds = build_final_test_folds(full, _periods(), 1)

    assert tuning.filter(pl.col("forecast_origin") >= _periods().final_test_start).is_empty()
    assert final_only.filter(pl.col("forecast_origin") < _periods().final_test_start).is_empty()
    assert all(fold.validation_start >= _periods().final_test_start for fold in folds)


def test_validation_folds_reject_final_test_rows() -> None:
    with pytest.raises(ValueError, match="must not receive final-test rows"):
        build_validation_folds(_model_frame(), _periods(), 1)
