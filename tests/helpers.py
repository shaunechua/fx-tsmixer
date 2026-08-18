from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np


def business_dates(start: date, count: int) -> list[date]:
    dates: list[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def write_series_csv(path: Path, header: str, dates: list[date], values: list[float]) -> None:
    lines = [f"observation_date,{header}"]
    lines.extend(
        f"{observation_date.isoformat()},{value:.10f}"
        for observation_date, value in zip(dates, values, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fx_spot_csv(path: Path, dates: list[date], *, final_multiplier: float) -> None:
    index = np.arange(len(dates) - 1, dtype=float)
    returns = 0.0004 + 0.0005 * np.sin(index / 6.0)
    levels = [140.0]
    for value in returns:
        levels.append(levels[-1] * math.exp(float(value)))
    adjusted = levels.copy()
    for row_index, observation_date in enumerate(dates):
        if observation_date >= date(2024, 4, 1):
            adjusted[row_index] *= final_multiplier
    write_series_csv(path, "DEXJPUS", dates, [float(value) for value in adjusted])


def synthetic_project(tmp_path: Path, *, final_multiplier: float = 1.0) -> Path:
    project_root = tmp_path / f"project-{final_multiplier}"
    (project_root / "configs").mkdir(parents=True)
    (project_root / "data").mkdir(parents=True)
    dates = business_dates(date(2023, 9, 1), 185)
    write_fx_spot_csv(
        project_root / "data" / "fx_spot.csv", dates, final_multiplier=final_multiplier
    )
    us_rate = [5.0 + 0.02 * math.sin(index / 5.0) for index in range(len(dates))]
    write_series_csv(project_root / "data" / "us_rate.csv", "DFF", dates, us_rate)
    jp_rate = [0.1] * len(dates)
    write_series_csv(project_root / "data" / "jp_rate.csv", "IRSTCI01JPM156N", dates, jp_rate)
    (project_root / "configs" / "experiment.yaml").write_text(
        "\n".join(
            [
                "seed: 42",
                "output_dir: outputs",
                "periods:",
                "  development_end: 2024-01-31",
                "  validation_start: 2024-02-01",
                "  validation_end: 2024-03-29",
                "  final_test_start: 2024-04-01",
                "  final_test_end: 2024-05-31",
                "purge_business_days: 1",
                "datasets:",
                "  fx_spot:",
                "    path: data/fx_spot.csv",
                "    date_column: observation_date",
                "    value_column: DEXJPUS",
                "    availability_lag: 0",
                "    availability_lag_unit: business_days",
                "    maximum_staleness_days: 0",
                "  us_rate:",
                "    path: data/us_rate.csv",
                "    date_column: observation_date",
                "    value_column: DFF",
                "    availability_lag: 1",
                "    availability_lag_unit: business_days",
                "    maximum_staleness_days: 7",
                "  jp_rate:",
                "    path: data/jp_rate.csv",
                "    date_column: observation_date",
                "    value_column: IRSTCI01JPM156N",
                "    availability_lag: 1",
                "    availability_lag_unit: business_days",
                "    maximum_staleness_days: 7",
                "missing_values:",
                '  - ""',
                '  - "."',
                "  - NA",
                "  - N/A",
                "models:",
                "  arima:",
                "    max_p: 1",
                "    max_q: 1",
                "    maximum_iterations: 20",
                "  tsmixer:",
                "    lookback: 32",
                "    hidden_size: 32",
                "    mixer_blocks: 2",
                "    dropout: 0.1",
                "    learning_rate: 0.001",
                "    weight_decay: 0.0001",
                "    gradient_clip: 1.0",
                "    batch_size: 16",
                "    maximum_epochs: 2",
                "    early_stopping_patience: 1",
                "    early_stopping_min_delta: 0.0001",
                "    feature_sets:",
                "      tsmixer: []",
                "      tsmixer_with_rates:",
                "        [fx_return_1, fx_return_5, fx_volatility_20, us_rate_change_1]",
                "signal_threshold: 0.0",
                "transaction_cost: 0.0001",
                "annualization_periods: 252",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return project_root / "configs" / "experiment.yaml"
