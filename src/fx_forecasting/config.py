from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    date_column: str
    value_column: str
    availability_lag: int
    availability_lag_unit: str
    maximum_staleness_days: int


@dataclass(frozen=True)
class PeriodsConfig:
    development_end: date
    validation_start: date
    validation_end: date
    final_test_start: date
    final_test_end: date


@dataclass(frozen=True)
class ArimaConfig:
    max_p: int
    max_q: int
    maximum_iterations: int


@dataclass(frozen=True)
class TSMixerConfig:
    lookback: int
    hidden_size: int
    mixer_blocks: int
    dropout: float
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    batch_size: int
    maximum_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    feature_sets: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    output_dir: Path
    periods: PeriodsConfig
    purge_business_days: int
    datasets: tuple[DatasetConfig, ...]
    missing_values: tuple[str, ...]
    arima: ArimaConfig
    tsmixer: TSMixerConfig
    signal_threshold: float
    transaction_cost: float
    annualization_periods: int


def load_experiment_config(path: Path) -> ExperimentConfig:
    config_path = path.resolve(strict=True)
    project_root = config_path.parent.parent
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    periods = PeriodsConfig(**raw["periods"])
    if not (
        periods.development_end < periods.validation_start <= periods.validation_end
        and periods.validation_end < periods.final_test_start <= periods.final_test_end
    ):
        raise ValueError(
            "periods must satisfy development_end < validation_start <= validation_end "
            "< final_test_start <= final_test_end."
        )

    datasets = tuple(
        DatasetConfig(
            name=name,
            path=project_root / values["path"],
            date_column=values["date_column"],
            value_column=values["value_column"],
            availability_lag=values["availability_lag"],
            availability_lag_unit=values["availability_lag_unit"],
            maximum_staleness_days=values["maximum_staleness_days"],
        )
        for name, values in raw["datasets"].items()
    )
    if not {"fx_spot", "us_rate", "jp_rate"}.issubset(dataset.name for dataset in datasets):
        raise ValueError("datasets must include fx_spot, us_rate, and jp_rate.")
    for dataset in datasets:
        if dataset.availability_lag_unit not in {"business_days", "calendar_days"}:
            raise ValueError(
                f"datasets.{dataset.name}.availability_lag_unit must be "
                "business_days or calendar_days."
            )
        if dataset.availability_lag < 0 or dataset.maximum_staleness_days < 0:
            raise ValueError("Dataset lags and staleness limits must be non-negative.")

    tsmixer_raw = raw["models"]["tsmixer"]
    tsmixer = TSMixerConfig(
        **{
            **tsmixer_raw,
            "feature_sets": {
                name: tuple(features) for name, features in tsmixer_raw["feature_sets"].items()
            },
        }
    )

    config = ExperimentConfig(
        seed=raw["seed"],
        output_dir=project_root / raw["output_dir"],
        periods=periods,
        purge_business_days=raw["purge_business_days"],
        datasets=datasets,
        missing_values=tuple(raw["missing_values"]),
        arima=ArimaConfig(**raw["models"]["arima"]),
        tsmixer=tsmixer,
        signal_threshold=float(raw["signal_threshold"]),
        transaction_cost=float(raw["transaction_cost"]),
        annualization_periods=raw["annualization_periods"],
    )
    if config.purge_business_days <= 0:
        raise ValueError("purge_business_days must be positive.")
    if config.signal_threshold < 0.0 or config.transaction_cost < 0.0:
        raise ValueError("Signal threshold and transaction cost must be non-negative.")
    return config
