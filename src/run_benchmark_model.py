"""Run statistical and naive precipitation benchmarks on the GLSTM dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import json
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from Data.temporal_dataset import (
    TimeSeriesSplits,
    WindowedSplits,
    build_station_feature_dataset,
    chronological_split,
    create_windowed_splits,
    load_station_catalog,
)
from Evaluation.experiment_outputs import save_dataset_contract, save_prediction_outputs
from Training.experiment_runner import print_status, resolve_catalog_path, select_stations
from output_layout import logs_directory


# Data selection: keep these values equal to run_experiment.py for a direct comparison.
START_DATE = "2000-01-01"
END_DATE = "2025-12-31"
STATE = "RS"
CATALOG_PATH = ROOT / "Datasets" / "dados_inmet" / "Catalogo*.csv"
MAX_STATIONS = None  # Set an integer for a quick smoke run.

INCLUDE_PRECIPITATION = True
INCLUDE_TEMPERATURE = True
INCLUDE_SPECIFIC_HUMIDITY = True
INCLUDE_WIND = False
INCLUDE_VERTICAL_VELOCITY = True

WINDOW_SIZE = 15
FORECAST_HORIZON = 5
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2

# One or more of: auto_arima, persistence_station, seasonal_persistence,
# station_mean, station_median, zero.
BENCHMARK_MODELS = ["auto_arima"]
SEASONAL_PERSISTENCE_LAG_DAYS = 365
CLIP_NEGATIVE_PREDICTIONS = True

# pmdarima.auto_arima options. Models are selected independently per station.
AUTO_ARIMA_SEASONAL = False
AUTO_ARIMA_M = 1
AUTO_ARIMA_START_P = 0
AUTO_ARIMA_START_Q = 0
AUTO_ARIMA_MAX_P = 5
AUTO_ARIMA_MAX_Q = 5
AUTO_ARIMA_START_P_SEASONAL = 0
AUTO_ARIMA_START_Q_SEASONAL = 0
AUTO_ARIMA_MAX_P_SEASONAL = 2
AUTO_ARIMA_MAX_Q_SEASONAL = 2
AUTO_ARIMA_MAX_ORDER = 10
AUTO_ARIMA_INFORMATION_CRITERION = "aic"
AUTO_ARIMA_STEPWISE = True
AUTO_ARIMA_N_JOBS = 1  # pmdarima uses this only when STEPWISE=False.
AUTO_ARIMA_MAXITER = 50
AUTO_ARIMA_TRACE = False
AUTO_ARIMA_ERROR_POLICY = "raise"  # "raise" or "station_mean".

OUTPUT_ROOT = ROOT / "Experiments" / "run_benchmark_model"
RUN_NAME = None
PLOT_STATION_NAME = "PORTO ALEGRE JARDIM BOTANICO"
SHOW_CONSOLE_INFO = True
PROGRESS_TIME_CHUNK_DAYS = 30
USE_DAILY_CACHE = True


@dataclass(frozen=True)
class BenchmarkConfig:
    """Complete, serializable configuration for one benchmark pipeline."""

    start_date: str = START_DATE
    end_date: str = END_DATE
    state: str = STATE
    catalog_path: Path | str = CATALOG_PATH
    max_stations: int | None = MAX_STATIONS
    include_precipitation: bool = INCLUDE_PRECIPITATION
    include_temperature: bool = INCLUDE_TEMPERATURE
    include_specific_humidity: bool = INCLUDE_SPECIFIC_HUMIDITY
    include_wind: bool = INCLUDE_WIND
    include_vertical_velocity: bool = INCLUDE_VERTICAL_VELOCITY
    window_size: int = WINDOW_SIZE
    forecast_horizon: int = FORECAST_HORIZON
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    benchmark_models: tuple[str, ...] = tuple(BENCHMARK_MODELS)
    seasonal_persistence_lag_days: int = SEASONAL_PERSISTENCE_LAG_DAYS
    clip_negative_predictions: bool = CLIP_NEGATIVE_PREDICTIONS
    auto_arima_seasonal: bool = AUTO_ARIMA_SEASONAL
    auto_arima_m: int = AUTO_ARIMA_M
    auto_arima_start_p: int = AUTO_ARIMA_START_P
    auto_arima_start_q: int = AUTO_ARIMA_START_Q
    auto_arima_max_p: int = AUTO_ARIMA_MAX_P
    auto_arima_max_q: int = AUTO_ARIMA_MAX_Q
    auto_arima_start_P: int = AUTO_ARIMA_START_P_SEASONAL
    auto_arima_start_Q: int = AUTO_ARIMA_START_Q_SEASONAL
    auto_arima_max_P: int = AUTO_ARIMA_MAX_P_SEASONAL
    auto_arima_max_Q: int = AUTO_ARIMA_MAX_Q_SEASONAL
    auto_arima_max_order: int = AUTO_ARIMA_MAX_ORDER
    auto_arima_information_criterion: str = AUTO_ARIMA_INFORMATION_CRITERION
    auto_arima_stepwise: bool = AUTO_ARIMA_STEPWISE
    auto_arima_n_jobs: int = AUTO_ARIMA_N_JOBS
    auto_arima_maxiter: int = AUTO_ARIMA_MAXITER
    auto_arima_trace: bool = AUTO_ARIMA_TRACE
    auto_arima_error_policy: str = AUTO_ARIMA_ERROR_POLICY
    plot_station_name: str | None = PLOT_STATION_NAME
    show_console_info: bool = SHOW_CONSOLE_INFO
    progress_time_chunk_days: int = PROGRESS_TIME_CHUNK_DAYS
    use_daily_cache: bool = USE_DAILY_CACHE

    # save_dataset_contract expects these two attributes. Benchmarks use raw rain.
    normalize_features: bool = False
    normalize_target: bool = False


_MODEL_ALIASES = {
    "auto_arima": "auto_arima",
    "autoarima": "auto_arima",
    "arima": "auto_arima",
    "persistence": "persistence_station",
    "persistence_station": "persistence_station",
    "station_persistence": "persistence_station",
    "seasonal_persistence": "seasonal_persistence",
    "persistence_seasonal": "seasonal_persistence",
    "station_mean": "station_mean",
    "mean": "station_mean",
    "station_median": "station_median",
    "median": "station_median",
    "zero": "zero",
    "no_rain": "zero",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, indent=2, sort_keys=True)


def _unique_run_directory(output_root: Path, run_name: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = run_name or f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for suffix in range(1, 10_000):
        name = base_name if suffix == 1 else f"{base_name}_{suffix:02d}"
        candidate = output_root / name
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique run below {output_root}.")


def _normalize_model_names(model_names: str | Iterable[str]) -> tuple[str, ...]:
    raw_names = [model_names] if isinstance(model_names, str) else list(model_names)
    if not raw_names:
        raise ValueError("At least one benchmark model must be selected.")

    resolved: list[str] = []
    for raw_name in raw_names:
        token = re.sub(r"[\s-]+", "_", str(raw_name).strip().lower())
        try:
            model_name = _MODEL_ALIASES[token]
        except KeyError as exc:
            available = ", ".join(sorted(set(_MODEL_ALIASES.values())))
            raise ValueError(f"Unknown benchmark model {raw_name!r}. Available: {available}.") from exc
        if model_name not in resolved:
            resolved.append(model_name)
    return tuple(resolved)


def _validate_config(config: BenchmarkConfig, selected_models: Iterable[str]) -> None:
    selected_models = set(selected_models)
    if not config.include_precipitation:
        raise ValueError("INCLUDE_PRECIPITATION must be True because precipitation is the target.")
    if config.window_size < 1 or config.forecast_horizon < 1:
        raise ValueError("WINDOW_SIZE and FORECAST_HORIZON must be positive.")
    if (
        "seasonal_persistence" in selected_models
        and config.seasonal_persistence_lag_days < config.forecast_horizon
    ):
        raise ValueError(
            "SEASONAL_PERSISTENCE_LAG_DAYS must be at least FORECAST_HORIZON "
            "to prevent future-target leakage."
        )
    if (
        "auto_arima" in selected_models
        and config.auto_arima_error_policy not in {"raise", "station_mean"}
    ):
        raise ValueError("AUTO_ARIMA_ERROR_POLICY must be 'raise' or 'station_mean'.")
    if (
        "auto_arima" in selected_models
        and config.auto_arima_seasonal
        and config.auto_arima_m < 2
    ):
        raise ValueError("AUTO_ARIMA_M must be at least 2 when seasonal AutoARIMA is enabled.")


def persistence_station_predictions(windowed: WindowedSplits) -> np.ndarray:
    """Repeat each station's last observed rainfall over every forecast lead."""
    if "tp" not in {str(value) for value in windowed.test_X.feature.values}:
        raise ValueError("The precipitation feature 'tp' is required for persistence.")
    last_observed = np.asarray(
        windowed.test_X.sel(feature="tp").isel(lag=-1).values,
        dtype=float,
    )
    horizon = windowed.test_y.sizes["lead_day"]
    return np.repeat(last_observed[:, None, :], horizon, axis=1)


def station_statistic_predictions(
    splits: TimeSeriesSplits,
    windowed: WindowedSplits,
    statistic: str,
) -> np.ndarray:
    """Forecast a train-only station mean/median, or zero rainfall."""
    train = np.asarray(splits.train_y.values, dtype=float)
    if statistic == "mean":
        station_values = np.mean(train, axis=0)
    elif statistic == "median":
        station_values = np.median(train, axis=0)
    elif statistic == "zero":
        station_values = np.zeros(train.shape[1], dtype=float)
    else:
        raise ValueError("statistic must be 'mean', 'median', or 'zero'.")
    return np.broadcast_to(station_values, windowed.test_y.shape).copy()


def seasonal_persistence_predictions(
    raw_y: xr.DataArray,
    test_y: xr.DataArray,
    lag_days: int,
) -> np.ndarray:
    """Use rainfall from the same station and date ``lag_days`` earlier."""
    target_times = np.asarray(test_y.coords["target_time"].values, dtype="datetime64[ns]")
    input_end = np.asarray(test_y.coords["input_end_time"].values, dtype="datetime64[ns]")
    source_times = target_times - np.timedelta64(int(lag_days), "D")
    if np.any(source_times > input_end[:, None]):
        raise ValueError("Seasonal persistence would access observations after the forecast origin.")

    raw_index = pd.DatetimeIndex(pd.to_datetime(raw_y.time.values))
    positions = raw_index.get_indexer(pd.to_datetime(source_times.reshape(-1)))
    if np.any(positions < 0):
        first_missing = pd.Timestamp(source_times.reshape(-1)[np.flatnonzero(positions < 0)[0]])
        raise ValueError(
            f"No observation exists for seasonal-persistence source date {first_missing.date()}. "
            "Increase START_DATE or reduce SEASONAL_PERSISTENCE_LAG_DAYS."
        )
    values = np.asarray(raw_y.values, dtype=float)[positions]
    return values.reshape(test_y.shape)


def _require_auto_arima():
    try:
        from pmdarima import auto_arima
    except ImportError as exc:
        raise ImportError(
            "The auto_arima benchmark requires pmdarima. "
            "Install the project requirements with: pip install -r requirements.txt"
        ) from exc
    return auto_arima


def auto_arima_predictions(
    splits: TimeSeriesSplits,
    windowed: WindowedSplits,
    config: BenchmarkConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit one train-only AutoARIMA per station and run rolling-origin forecasts."""
    auto_arima = _require_auto_arima()
    train = np.asarray(splits.train_y.values, dtype=float)
    validation = np.asarray(splits.val_y.values, dtype=float)
    test = np.asarray(splits.test_y.values, dtype=float)
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(validation)) or not np.all(np.isfinite(test)):
        raise ValueError("AutoARIMA requires finite precipitation values in all chronological splits.")

    n_samples, horizon, n_stations = windowed.test_y.shape
    expected_samples = test.shape[0] - config.window_size - horizon + 1
    if n_samples != expected_samples:
        raise ValueError("Test-window shape is inconsistent with the chronological test split.")

    predictions = np.empty((n_samples, horizon, n_stations), dtype=float)
    details: list[dict[str, Any]] = []
    station_names = [str(value) for value in splits.train_y.station.values]

    for station_index, station_name in enumerate(station_names):
        print_status(
            f"AutoARIMA station {station_index + 1}/{n_stations}: {station_name}",
            config.show_console_info,
        )
        train_series = train[:, station_index]
        try:
            model = auto_arima(
                train_series,
                seasonal=config.auto_arima_seasonal,
                m=config.auto_arima_m,
                start_p=config.auto_arima_start_p,
                start_q=config.auto_arima_start_q,
                max_p=config.auto_arima_max_p,
                max_q=config.auto_arima_max_q,
                start_P=config.auto_arima_start_P,
                start_Q=config.auto_arima_start_Q,
                max_P=config.auto_arima_max_P,
                max_Q=config.auto_arima_max_Q,
                max_order=config.auto_arima_max_order,
                information_criterion=config.auto_arima_information_criterion,
                stepwise=config.auto_arima_stepwise,
                n_jobs=config.auto_arima_n_jobs,
                maxiter=config.auto_arima_maxiter,
                trace=config.auto_arima_trace,
                error_action="raise",
                suppress_warnings=True,
            )
            result = model.arima_res_
            initial_observations = np.concatenate(
                (validation[:, station_index], test[: config.window_size, station_index])
            )
            if initial_observations.size:
                result = result.extend(initial_observations)

            for sample_index in range(n_samples):
                predictions[sample_index, :, station_index] = np.asarray(
                    result.forecast(steps=horizon), dtype=float
                )
                # The next origin may use the target that has become observed. No refit occurs.
                if sample_index + 1 < n_samples:
                    observed_index = config.window_size + sample_index
                    result = result.extend([test[observed_index, station_index]])

            details.append(
                {
                    "station": station_name,
                    "status": "fitted",
                    "order": list(model.order),
                    "seasonal_order": list(model.seasonal_order),
                    "aic": float(model.aic()),
                    "n_train": int(train_series.size),
                }
            )
        except Exception as exc:
            if config.auto_arima_error_policy == "raise":
                raise RuntimeError(f"AutoARIMA failed for station {station_name!r}.") from exc
            fallback = float(np.mean(train_series))
            predictions[:, :, station_index] = fallback
            details.append(
                {
                    "station": station_name,
                    "status": "fallback_station_mean",
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback_value": fallback,
                    "n_train": int(train_series.size),
                }
            )

    return predictions, details


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Return GLSTM-compatible global and per-lead regression metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 3:
        raise ValueError("actual and predicted must share [sample, lead_day, station] shape.")
    finite = np.isfinite(actual) & np.isfinite(predicted)
    if not np.all(finite):
        raise ValueError("Metrics require finite actual and predicted values.")

    residual = actual - predicted
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    mape = float(np.mean(np.abs(residual) / (np.abs(actual) + 1e-3)) * 100.0)

    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return float("nan") if np.isclose(ss_tot, 0.0) else float(1.0 - ss_res / ss_tot)

    return {
        "loss": mse,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "mape": mape,
        "r2": r2_score(actual, predicted),
        "r2_by_step": [
            r2_score(actual[:, lead, :], predicted[:, lead, :])
            for lead in range(actual.shape[1])
        ],
        "bias": float(np.mean(predicted - actual)),
    }


def _predict_model(
    model_name: str,
    raw_y: xr.DataArray,
    splits: TimeSeriesSplits,
    windowed: WindowedSplits,
    config: BenchmarkConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model_name == "auto_arima":
        predictions, station_details = auto_arima_predictions(splits, windowed, config)
        return predictions, {"stations": station_details}
    if model_name == "persistence_station":
        return persistence_station_predictions(windowed), {}
    if model_name == "seasonal_persistence":
        return seasonal_persistence_predictions(
            raw_y, windowed.test_y, config.seasonal_persistence_lag_days
        ), {"lag_days": config.seasonal_persistence_lag_days}
    if model_name == "station_mean":
        return station_statistic_predictions(splits, windowed, "mean"), {}
    if model_name == "station_median":
        return station_statistic_predictions(splits, windowed, "median"), {}
    if model_name == "zero":
        return station_statistic_predictions(splits, windowed, "zero"), {}
    raise AssertionError(f"Unhandled benchmark model: {model_name}")


def run_benchmark_models(
    config: BenchmarkConfig | None = None,
    *,
    model_names: str | Iterable[str] | None = None,
    output_root: Path | str | None = None,
    run_name: str | None = None,
) -> Path:
    """Run selected benchmarks once over the exact dated GLSTM data pipeline."""
    config = BenchmarkConfig() if config is None else config
    selected_models = _normalize_model_names(
        config.benchmark_models if model_names is None else model_names
    )
    _validate_config(config, selected_models)
    run_dir = _unique_run_directory(
        Path(OUTPUT_ROOT if output_root is None else output_root),
        RUN_NAME if run_name is None else run_name,
    )

    print_status(
        f"Loading benchmark dataset ({config.start_date} -> {config.end_date})",
        config.show_console_info,
    )
    stations = select_stations(
        load_station_catalog(resolve_catalog_path(config.catalog_path), state=config.state),
        config.max_stations,
    )
    raw_X, raw_y = build_station_feature_dataset(
        stations=stations,
        start_date=config.start_date,
        end_date=config.end_date,
        include_precipitation=config.include_precipitation,
        include_temperature=config.include_temperature,
        include_specific_humidity=config.include_specific_humidity,
        include_wind=config.include_wind,
        include_vertical_velocity=config.include_vertical_velocity,
        show_progress=config.show_console_info,
        progress_time_chunk_days=config.progress_time_chunk_days,
        prefer_daily_cache=config.use_daily_cache,
    )
    splits = chronological_split(
        raw_X, raw_y, train_ratio=config.train_ratio, val_ratio=config.val_ratio
    )
    windowed = create_windowed_splits(
        splits, window_size=config.window_size, horizon=config.forecast_horizon
    )
    actual = np.asarray(windowed.test_y.values, dtype=float)

    parent_config = asdict(config)
    parent_config["benchmark_models"] = list(selected_models)
    logs_dir = logs_directory(run_dir, create=True)
    _write_json(logs_dir / "config.json", parent_config)
    save_dataset_contract(run_dir, raw_X, raw_y, windowed, windowed, config)

    summaries: list[dict[str, Any]] = []
    for model_name in selected_models:
        print_status(f"Running benchmark: {model_name}", config.show_console_info)
        model_dir = run_dir / model_name
        model_dir.mkdir()
        model_logs_dir = logs_directory(model_dir, create=True)
        predictions, model_details = _predict_model(
            model_name, raw_y, splits, windowed, config
        )
        if config.clip_negative_predictions:
            predictions = np.maximum(predictions, 0.0)

        metrics = regression_metrics(actual, predictions)
        model_config = dict(parent_config)
        model_config["benchmark_model"] = model_name
        _write_json(model_logs_dir / "config.json", model_config)
        _write_json(model_logs_dir / "test_metrics.json", metrics)
        if model_details:
            _write_json(model_logs_dir / "model_details.json", model_details)
        save_prediction_outputs(
            model_dir,
            actual,
            predictions,
            windowed.test_y,
            target_scaler=None,
            plot_station_name=config.plot_station_name,
        )
        summaries.append({"model": model_name, **metrics, "output": str(model_dir)})

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(logs_dir / "benchmark_summary.csv", index=False)
    _write_json(logs_dir / "benchmark_summary.json", summaries)
    print_status(f"Benchmark complete: {run_dir}", config.show_console_info)
    return run_dir


if __name__ == "__main__":
    run_benchmark_models()
