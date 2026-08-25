from __future__ import annotations

import math
import tempfile
import time
from typing import cast

import numpy as np
import polars as pl
import torch
from darts import TimeSeries
from darts.models import TSMixerModel
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from torch import nn

from fx_forecasting.config import TSMixerConfig
from fx_forecasting.models import (
    MODEL_EXECUTION_ERRORS,
    FloatArray,
    forecast_frame,
)
from fx_forecasting.validation import WalkForwardFold, fold_frames


def generate_tsmixer_forecasts(
    frame: pl.DataFrame,
    folds: tuple[WalkForwardFold, ...],
    config: TSMixerConfig,
    *,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    forecast_frames: list[pl.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    for model_name, feature_names in config.feature_sets.items():
        completed_folds = 0
        for fold in folds:
            started = time.perf_counter()
            training, validation = fold_frames(frame, fold)
            history = frame.filter(pl.col("forecast_origin") <= fold.validation_end)
            try:
                predictions, epochs_trained, best_loss = _fit_tsmixer_model(
                    history,
                    training.height,
                    fold.purge_observations,
                    config,
                    feature_names,
                    seed=seed,
                )
                forecast_frames.append(forecast_frame(validation, fold, model_name, predictions))
                completed_folds += 1
                metadata_rows.append(
                    {
                        "model": model_name,
                        "fold_id": fold.fold_id,
                        "refit_id": fold.refit_id,
                        "training_end": fold.training_end,
                        "epochs_trained": epochs_trained,
                        "best_validation_loss": best_loss,
                        "runtime_seconds": time.perf_counter() - started,
                        "failure_reason": None,
                    }
                )
            except MODEL_EXECUTION_ERRORS as error:
                metadata_rows.append(
                    {
                        "model": model_name,
                        "fold_id": fold.fold_id,
                        "refit_id": fold.refit_id,
                        "training_end": fold.training_end,
                        "epochs_trained": 0,
                        "best_validation_loss": None,
                        "runtime_seconds": time.perf_counter() - started,
                        "failure_reason": str(error),
                    }
                )
        if completed_folds == 0:
            raise ValueError(f"{model_name} did not complete any fold successfully.")
    return (
        pl.concat(forecast_frames).sort(["model", "forecast_origin"]),
        pl.DataFrame(metadata_rows).sort(["model", "fold_id"]),
    )


def _fit_tsmixer_model(
    history: pl.DataFrame,
    training_rows: int,
    purge_rows: int,
    config: TSMixerConfig,
    feature_names: tuple[str, ...],
    *,
    seed: int,
) -> tuple[FloatArray, int, float]:
    if training_rows < config.lookback + 4:
        raise ValueError("Insufficient windows for chronological inner validation.")
    prediction_start = training_rows + purge_rows
    split_index = int(training_rows * 0.8)
    if split_index <= config.lookback or split_index >= training_rows:
        raise ValueError("Insufficient windows for chronological inner validation.")

    targets = history.get_column("target_return").to_numpy().astype(np.float64)
    target_scaler = StandardScaler().fit(targets[:split_index].reshape(-1, 1))
    scaled_targets = target_scaler.transform(targets.reshape(-1, 1)).astype(np.float32)
    validation_start = split_index - config.lookback
    train_target = TimeSeries.from_values(scaled_targets[:split_index])
    valid_target = TimeSeries.from_values(scaled_targets[validation_start:training_rows])
    target_histories = [
        TimeSeries.from_values(scaled_targets[:index])
        for index in range(prediction_start, history.height)
    ]

    train_covariates = valid_covariates = feature_histories = None
    if feature_names:
        features = history.select(list(feature_names)).to_numpy().astype(np.float64)
        feature_medians = np.nanmedian(features[:split_index], axis=0)
        if not np.isfinite(feature_medians).all():
            raise ValueError("Inner training data cannot impute an entirely missing feature.")
        imputed_features = np.where(np.isnan(features), feature_medians, features)
        feature_scaler = StandardScaler().fit(imputed_features[:split_index])
        scaled_features = feature_scaler.transform(imputed_features).astype(np.float32)
        train_covariates = TimeSeries.from_values(scaled_features[:split_index])
        valid_covariates = TimeSeries.from_values(scaled_features[validation_start:training_rows])
        feature_histories = [
            TimeSeries.from_values(scaled_features[: index + 1])
            for index in range(prediction_start, history.height)
        ]

    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        mode="min",
    )

    with tempfile.TemporaryDirectory(prefix="fx-tsmixer-") as work_dir:
        model = TSMixerModel(
            input_chunk_length=config.lookback,
            output_chunk_length=1,
            hidden_size=config.hidden_size,
            ff_size=config.hidden_size,
            num_blocks=config.mixer_blocks,
            activation="GELU",
            dropout=config.dropout,
            normalize_before=True,
            use_static_covariates=False,
            loss_fn=nn.L1Loss(),
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
            batch_size=min(config.batch_size, split_index - config.lookback),
            n_epochs=config.maximum_epochs,
            random_state=seed,
            model_name="tsmixer",
            work_dir=work_dir,
            save_checkpoints=True,
            pl_trainer_kwargs={
                "accelerator": "cpu",
                "callbacks": [early_stopper],
                "deterministic": True,
                "enable_model_summary": False,
                "enable_progress_bar": False,
                "gradient_clip_val": config.gradient_clip,
                "logger": False,
                "num_sanity_val_steps": 0,
            },
        )
        model.fit(
            series=train_target,
            future_covariates=train_covariates,
            val_series=valid_target,
            val_future_covariates=valid_covariates,
            dataloader_kwargs={"pin_memory": False},
            verbose=False,
            load_best=True,
        )
        epochs_trained = model.trainer.current_epoch
        forecasts = cast(
            list[TimeSeries],
            model.predict(
                1,
                series=target_histories,
                future_covariates=feature_histories,
                dataloader_kwargs={"pin_memory": False},
                verbose=False,
                show_warnings=False,
            ),
        )
        scaled_predictions = np.asarray(
            [forecast.values(copy=False).item() for forecast in forecasts],
            dtype=np.float64,
        )
        best_loss = float(early_stopper.best_score)

    if not math.isfinite(best_loss) or not np.isfinite(scaled_predictions).all():
        raise ValueError("TSMixer training did not produce finite results.")
    predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).ravel()
    return np.asarray(predictions, dtype=np.float64), epochs_trained, best_loss
