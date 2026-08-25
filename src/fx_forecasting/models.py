from __future__ import annotations

import json
import math
import warnings
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from statsmodels.tsa.arima.model import ARIMA

from fx_forecasting.config import ArimaConfig
from fx_forecasting.validation import WalkForwardFold, fold_frames

FloatArray = NDArray[np.float64]
MODEL_RANDOM_WALK = "random_walk"
MODEL_HISTORICAL_MEAN = "historical_mean"
MODEL_ARIMA = "arima"
MODEL_EXECUTION_ERRORS = (
    FloatingPointError,
    OverflowError,
    RuntimeError,
    ValueError,
    ZeroDivisionError,
    np.linalg.LinAlgError,
)
FORECAST_SCHEMA = {
    "forecast_origin": pl.Date,
    "target_date": pl.Date,
    "horizon": pl.Int64,
    "model": pl.String,
    "target_kind": pl.String,
    "training_start": pl.Date,
    "training_end": pl.Date,
    "predicted_level": pl.Float64,
    "predicted_return": pl.Float64,
    "realized_level": pl.Float64,
    "realized_return": pl.Float64,
    "realized_carry": pl.Float64,
    "refit_id": pl.String,
}
FORECAST_COLUMNS = tuple(FORECAST_SCHEMA)


def generate_baseline_forecasts(
    frame: pl.DataFrame,
    folds: tuple[WalkForwardFold, ...],
) -> pl.DataFrame:
    forecast_frames: list[pl.DataFrame] = []
    for fold in folds:
        training, validation = fold_frames(frame, fold)
        mean_return = training.get_column("target_return").mean()
        forecast_frames.append(
            forecast_frame(
                validation,
                fold,
                MODEL_RANDOM_WALK,
                np.zeros(validation.height, dtype=np.float64),
            )
        )
        forecast_frames.append(
            forecast_frame(
                validation,
                fold,
                MODEL_HISTORICAL_MEAN,
                np.full(validation.height, mean_return, dtype=np.float64),
            )
        )
    return pl.concat(forecast_frames).sort(["model", "forecast_origin"])


def generate_arima_forecasts(
    frame: pl.DataFrame,
    folds: tuple[WalkForwardFold, ...],
    config: ArimaConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    forecast_frames: list[pl.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    for fold in folds:
        training, validation = fold_frames(frame, fold)
        training_returns = _float_array(training, "target_return")
        purge_returns = _float_array(
            frame.filter(
                pl.col("forecast_origin").is_between(
                    fold.purge_start,
                    fold.purge_end,
                    closed="both",
                )
            ),
            "target_return",
        )
        candidates: list[dict[str, object]] = []
        for p in range(config.max_p + 1):
            for q in range(config.max_q + 1):
                try:
                    candidates.append(
                        _fit_arima_candidate(training_returns, p, q, config.maximum_iterations)
                    )
                except MODEL_EXECUTION_ERRORS as error:
                    candidates.append(
                        {
                            "order": (p, 0, q),
                            "aic": None,
                            "aicc": None,
                            "converged": False,
                            "warning_messages": [],
                            "failure_reason": str(error),
                        }
                    )
        selected = _select_arima_candidate(candidates)
        if selected is None:
            metadata_rows.append(
                {
                    "model": MODEL_ARIMA,
                    "fold_id": fold.fold_id,
                    "refit_id": fold.refit_id,
                    "selected_order_json": json.dumps(None),
                    "selected_aicc": None,
                    "converged": False,
                    "forecast_rows": 0,
                    "failure_reason": "No ARIMA(p,0,q) candidate produced a finite AICc.",
                    "candidate_details_json": json.dumps(candidates),
                }
            )
            continue
        order = cast(tuple[int, int, int], selected["order"])
        failure_reason: str | None = None
        predicted: FloatArray | None = None
        try:
            fitted = _fit_arima(training_returns, order, config.maximum_iterations)
            for value in purge_returns:
                fitted = fitted.append([float(value)], refit=False)
            predictions: list[float] = []
            realized_returns = _float_array(validation, "target_return")
            for row_index, realized_return in enumerate(realized_returns):
                step = np.asarray(fitted.forecast(steps=1), dtype=np.float64)
                if step.size != 1 or not np.isfinite(step[0]):
                    raise ValueError("ARIMA forecast produced a non-finite value.")
                predictions.append(float(step[0]))
                if row_index < validation.height - 1:
                    fitted = fitted.append([float(realized_return)], refit=False)
            predicted = np.asarray(predictions, dtype=np.float64)
            forecast_frames.append(forecast_frame(validation, fold, MODEL_ARIMA, predicted))
        except MODEL_EXECUTION_ERRORS as error:
            failure_reason = str(error)
        metadata_rows.append(
            {
                "model": MODEL_ARIMA,
                "fold_id": fold.fold_id,
                "refit_id": fold.refit_id,
                "selected_order_json": json.dumps(order),
                "selected_aicc": selected["aicc"],
                "converged": selected["converged"],
                "forecast_rows": 0 if predicted is None else validation.height,
                "failure_reason": failure_reason,
                "candidate_details_json": json.dumps(candidates),
            }
        )
    return _concat_forecasts(forecast_frames), pl.DataFrame(metadata_rows).sort("fold_id")


def _fit_arima_candidate(
    training_returns: FloatArray,
    p: int,
    q: int,
    maximum_iterations: int,
) -> dict[str, object]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = _fit_arima(training_returns, (p, 0, q), maximum_iterations)
    aic_value = float(fitted.aic)
    aic = aic_value if math.isfinite(aic_value) else None
    parameter_count = len(fitted.params)
    return {
        "order": (p, 0, q),
        "aic": aic,
        "aicc": None if aic is None else _aicc(aic, len(training_returns), parameter_count),
        "converged": _fit_converged(fitted),
        "warning_messages": [str(item.message) for item in caught],
        "failure_reason": None,
    }


def _fit_arima(
    training_returns: FloatArray,
    order: tuple[int, int, int],
    maximum_iterations: int,
) -> Any:
    model = ARIMA(
        training_returns,
        order=order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(method_kwargs={"maxiter": maximum_iterations})


def _select_arima_candidate(candidates: list[dict[str, object]]) -> dict[str, object] | None:
    usable = [candidate for candidate in candidates if candidate.get("aicc") is not None]
    if not usable:
        return None
    converged = [candidate for candidate in usable if candidate.get("converged") is True]
    pool = converged if converged else usable
    return min(
        pool,
        key=lambda candidate: (
            cast(float, candidate["aicc"]),
            cast(tuple[int, int, int], candidate["order"]),
        ),
    )


def forecast_frame(
    validation: pl.DataFrame,
    fold: WalkForwardFold,
    model_name: str,
    predicted_returns: FloatArray,
) -> pl.DataFrame:
    if predicted_returns.shape[0] != validation.height:
        raise ValueError("Predicted returns must align one-for-one with validation rows.")
    origin_spot = _float_array(validation, "fx_spot")
    predicted_levels = origin_spot * np.exp(predicted_returns)
    return pl.DataFrame(
        {
            "forecast_origin": validation["forecast_origin"],
            "target_date": validation["target_date"],
            "horizon": [1] * validation.height,
            "model": [model_name] * validation.height,
            "target_kind": ["return"] * validation.height,
            "training_start": [fold.training_start] * validation.height,
            "training_end": [fold.training_end] * validation.height,
            "predicted_level": predicted_levels,
            "predicted_return": predicted_returns,
            "realized_level": validation["target_level"],
            "realized_return": validation["target_return"],
            "realized_carry": validation["target_carry"],
            "refit_id": [fold.refit_id] * validation.height,
        },
        schema=FORECAST_SCHEMA,
    )


def _concat_forecasts(forecast_frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not forecast_frames:
        return pl.DataFrame(schema=FORECAST_SCHEMA)
    return pl.concat(forecast_frames).sort("forecast_origin")


def _float_array(frame: pl.DataFrame, column: str) -> FloatArray:
    values = frame.get_column(column).to_numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Column {column} must contain only finite numeric values.")
    return values


def _aicc(aic: float, sample_size: int, parameter_count: int) -> float | None:
    denominator = sample_size - parameter_count - 1
    if denominator <= 0:
        return None
    return aic + (2 * parameter_count * (parameter_count + 1)) / denominator


def _fit_converged(result: Any) -> bool:
    mle_retvals = getattr(result, "mle_retvals", None)
    if isinstance(mle_retvals, dict):
        converged = mle_retvals.get("converged")
        if isinstance(converged, bool):
            return converged
    return True
