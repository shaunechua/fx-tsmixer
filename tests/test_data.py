from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fx_forecasting.config import DatasetConfig
from fx_forecasting.data import (
    LoadedDataset,
    align_feature_to_calendar,
    build_model_frame,
    build_tradable_calendar,
    load_csv_dataset,
    validate_point_in_time_alignment,
)


def _dataset_config(path: Path, *, date_column: str, value_column: str) -> DatasetConfig:
    return DatasetConfig(
        name="sample",
        path=path,
        date_column=date_column,
        value_column=value_column,
        availability_lag=0,
        availability_lag_unit="business_days",
        maximum_staleness_days=0,
    )


def _loaded_dataset(
    name: str,
    dates: list[date],
    values: list[float | None],
    *,
    lag: int,
    maximum_staleness_days: int,
) -> LoadedDataset:
    return LoadedDataset(
        config=DatasetConfig(
            name=name,
            path=Path(f"{name}.csv"),
            date_column="observation_date",
            value_column=name,
            availability_lag=lag,
            availability_lag_unit="business_days",
            maximum_staleness_days=maximum_staleness_days,
        ),
        frame=pl.DataFrame({"observation_date": dates, name: values}),
    )


def test_load_csv_dataset_uses_configured_columns_and_preserves_raw_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    raw = "date,selected,ignored\n2024-01-02,1.5,9\n2024-01-03,2.5,10\n"
    path.write_text(raw, encoding="utf-8")

    loaded = load_csv_dataset(
        _dataset_config(path, date_column="date", value_column="selected"), ("", ".", "NA", "N/A")
    )

    assert loaded.frame.columns == ["observation_date", "sample"]
    assert loaded.frame.get_column("sample").to_list() == [1.5, 2.5]
    assert path.read_text(encoding="utf-8") == raw


def test_load_csv_dataset_rejects_duplicate_dates_and_invalid_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("date,value\n2024-01-02,1\n2024-01-02,2\n", encoding="utf-8")
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("date,value\n2024-01-02,not-a-number\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate observation dates"):
        load_csv_dataset(
            _dataset_config(duplicate, date_column="date", value_column="value"), ("",)
        )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="not-a-number"):
        load_csv_dataset(_dataset_config(invalid, date_column="date", value_column="value"), ("",))


def test_alignment_applies_one_day_lag_without_backward_fill_or_stale_carry() -> None:
    fx_spot = _loaded_dataset(
        "fx_spot",
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)],
        [150.0, 151.0, 152.0],
        lag=0,
        maximum_staleness_days=0,
    )
    feature = _loaded_dataset("vix", [date(2024, 1, 2)], [14.0], lag=1, maximum_staleness_days=2)

    aligned = align_feature_to_calendar(build_tradable_calendar(fx_spot), feature)

    assert aligned.get_column("vix").to_list() == [None, 14.0, None]
    assert aligned.get_column("vix_availability_date").to_list() == [
        None,
        date(2024, 1, 3),
        date(2024, 1, 3),
    ]
    assert aligned.get_column("vix_source_observation_date").to_list() == [
        None,
        date(2024, 1, 2),
        date(2024, 1, 2),
    ]


def test_build_model_frame_sets_one_day_target_and_never_forward_fills_future_spot() -> None:
    panel = pl.DataFrame(
        {
            "forecast_origin": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            "fx_spot": [100.0, 101.0, None],
            "us_rate": [5.0, 5.0, 5.0],
            "jp_rate": [0.0, 0.0, 0.0],
        },
        schema={
            "forecast_origin": pl.Date,
            "fx_spot": pl.Float64,
            "us_rate": pl.Float64,
            "jp_rate": pl.Float64,
        },
    )

    model_frame = build_model_frame(panel)

    assert model_frame.height == 1
    assert model_frame.item(0, "target_date") == date(2024, 1, 3)
    assert np.isclose(model_frame.item(0, "target_return"), np.log(101.0 / 100.0))
    assert np.isclose(model_frame.item(0, "target_carry"), 0.05 / 365.0)


def test_point_in_time_validator_detects_future_feature_use() -> None:
    leaked = pl.DataFrame(
        {
            "forecast_origin": [date(2024, 1, 3)],
            "vix": [14.0],
            "vix_source_observation_date": [date(2024, 1, 3)],
            "vix_availability_date": [date(2024, 1, 4)],
        }
    )

    with pytest.raises(ValueError, match="used before its availability date"):
        validate_point_in_time_alignment(leaked, ("vix",))
