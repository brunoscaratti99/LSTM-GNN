from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is listed in requirements.
    tqdm = None

from Data.dataset_paths import open_daily_meteorological_dataset, open_meteorological_dataset
from Data.feature_extraction import (
    _infer_coord_name,
    _wrap_lon,
    daily_specific_humidity_features,
    daily_temp_features,
    daily_vertical_velocity,
    daily_wind_uv_features,
    era5_daily_precip_all,
    station_dictionary,
)


SUPPORTED_SCALERS = ("standard", "minmax")
DEFAULT_PROGRESS_TIME_CHUNK_DAYS = 30


def target_standard_deviation_to_physical_scale(
    values,
    target_scaler=None,
) -> np.ndarray:
    """Convert target-scale standard deviations to physical target units.

    Unlike ordinary inverse transformation, a standard deviation must only be
    rescaled; affine offsets must not be added.  The maintained training path
    uses a single target feature, so fitted scalers must expose exactly one
    scale value.  Persisted inference scalers are also accepted through their
    public ``kind`` and ``scale`` attributes.
    """
    deviations = np.asarray(values, dtype=np.float64)
    if target_scaler is None:
        return deviations

    if isinstance(target_scaler, StandardScaler):
        scaler_kind = "standard"
        raw_scale = getattr(target_scaler, "scale_", None)
    elif isinstance(target_scaler, MinMaxScaler):
        scaler_kind = "minmax"
        raw_scale = getattr(target_scaler, "scale_", None)
    else:
        scaler_kind = str(getattr(target_scaler, "kind", "")).lower()
        raw_scale = getattr(target_scaler, "scale", None)

    if scaler_kind not in SUPPORTED_SCALERS or raw_scale is None:
        raise TypeError(
            "target_scaler must be None, a fitted StandardScaler/MinMaxScaler, "
            "or an object exposing kind={'standard','minmax'} and scale."
        )

    scale = np.asarray(raw_scale, dtype=np.float64).reshape(-1)
    if scale.size != 1 or not np.isfinite(scale[0]) or scale[0] == 0.0:
        raise ValueError("The target scaler must contain exactly one finite, non-zero scale value.")

    scale_value = abs(float(scale[0]))
    if scaler_kind == "standard":
        return deviations * scale_value
    return deviations / scale_value


class _DatasetLoadProgressBar:
    """Per-dataset progress bar used while xarray values are materialized."""

    def __init__(self, label: str | None, total: int, enabled: bool) -> None:
        self.label = label or "xarray dataset"
        self.total = total
        self.enabled = bool(enabled and total > 0 and tqdm is not None)
        self._bar = None

    def __enter__(self):
        if self.enabled:
            self._bar = tqdm(
                total=self.total,
                desc=f"Loading {self.label}",
                unit="time-step",
                dynamic_ncols=True,
                leave=True,
            )
        return self

    def set_variable(self, variable: str) -> None:
        if self._bar is not None:
            self._bar.set_postfix_str(str(variable), refresh=True)

    def update(self, amount: int) -> None:
        if self._bar is not None:
            self._bar.update(amount)

    def __exit__(self, exc_type, exc, traceback):
        if self._bar is not None:
            self._bar.close()


def _time_chunk_slices(n_time: int, chunk_size: int) -> list[slice]:
    chunk_size = max(1, int(chunk_size))
    return [slice(start, min(start + chunk_size, n_time)) for start in range(0, n_time, chunk_size)]


def _load_time_chunks(
    data: xr.DataArray,
    progress: _DatasetLoadProgressBar,
    time_chunk_size: int,
) -> np.ndarray:
    chunks = []
    for time_slice in _time_chunk_slices(data.sizes["time"], time_chunk_size):
        chunks.append(np.asarray(data.isel(time=time_slice).values))
        progress.update(time_slice.stop - time_slice.start)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.float32)


@dataclass(frozen=True)
class TimeSeriesSplits:
    """Chronological xarray splits made before window creation."""

    train_X: xr.DataArray
    train_y: xr.DataArray
    val_X: xr.DataArray
    val_y: xr.DataArray
    test_X: xr.DataArray
    test_y: xr.DataArray


@dataclass(frozen=True)
class WindowedSplits:
    """Windowed xarray data plus target dates for each lead day."""

    train_X: xr.DataArray
    train_y: xr.DataArray
    val_X: xr.DataArray
    val_y: xr.DataArray
    test_X: xr.DataArray
    test_y: xr.DataArray


@dataclass(frozen=True)
class TorchWindowSplits:
    """Final tensors used by the neural network."""

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor


@dataclass(frozen=True)
class ScalingState:
    """Training-only scalers used to transform xarray windows."""

    feature_scaler: StandardScaler | MinMaxScaler | None
    target_scaler: StandardScaler | MinMaxScaler | None


def load_station_catalog(catalog_path: str | Path, state: str = "RS") -> dict[str, list[str]]:
    """Load the INMET station catalog and return the station coordinate mapping."""
    catalog = pd.read_csv(catalog_path, sep=";", encoding="latin1")
    return station_dictionary(catalog, UF=state)


def _feature_label(variable: str, extra_dims: Sequence[str], extra_index: tuple[object, ...]) -> str:
    if not extra_dims:
        return str(variable)
    suffix = "_".join(f"{dim}_{value}" for dim, value in zip(extra_dims, extra_index))
    return f"{variable}_{suffix}"


def grid_dataset_to_station_features(
    dataset: xr.Dataset,
    stations: Mapping[str, Sequence[float | str]],
    variables: Sequence[str] | None = None,
    progress_label: str | None = None,
    show_progress: bool = False,
    progress_time_chunk_days: int = DEFAULT_PROGRESS_TIME_CHUNK_DAYS,
) -> xr.DataArray:
    """
    Sample a gridded dataset at station coordinates and keep dates in xarray.

    Any non-time/station dimensions, such as pressure level, are flattened into
    the feature axis with explicit feature names.
    """
    if "time" not in dataset.dims and "time" not in dataset.coords:
        raise ValueError("Dataset must have a 'time' dimension or coordinate.")

    lat_name = _infer_coord_name(dataset, ["latitude", "lat"])
    lon_name = _infer_coord_name(dataset, ["longitude", "lon"])
    station_names = list(stations.keys())
    station_lats = [float(stations[name][0]) for name in station_names]
    station_lons = [_wrap_lon(float(stations[name][1]), dataset[lon_name]) for name in station_names]

    lat_da = xr.DataArray(station_lats, dims="station", coords={"station": station_names})
    lon_da = xr.DataArray(station_lons, dims="station", coords={"station": station_names})
    variables = list(variables) if variables is not None else list(dataset.data_vars)

    feature_arrays: list[np.ndarray] = []
    feature_names: list[str] = []
    time_values = pd.to_datetime(dataset.time.values)
    progress_total = len(time_values) * len(variables)

    with _DatasetLoadProgressBar(progress_label, progress_total, show_progress) as progress:
        for variable in variables:
            progress.set_variable(str(variable))
            selected = dataset[variable].sel({lat_name: lat_da, lon_name: lon_da}, method="nearest")
            selected = selected.transpose("time", "station", ...)
            extra_dims = [dim for dim in selected.dims if dim not in {"time", "station"}]

            if not extra_dims:
                feature_arrays.append(
                    _load_time_chunks(selected, progress, progress_time_chunk_days)[..., None]
                )
                feature_names.append(str(variable))
                continue

            stacked = selected.stack(feature=extra_dims).transpose("time", "station", "feature")
            feature_arrays.append(_load_time_chunks(stacked, progress, progress_time_chunk_days))
            for item in stacked.feature.values:
                item_tuple = item if isinstance(item, tuple) else (item,)
                feature_names.append(_feature_label(str(variable), extra_dims, item_tuple))

    if not feature_arrays:
        raise ValueError("No variables were selected for station feature extraction.")

    values = np.concatenate(feature_arrays, axis=-1).astype(np.float32)
    return xr.DataArray(
        values,
        dims=("time", "station", "feature"),
        coords={
            "time": time_values,
            "station": station_names,
            "feature": feature_names,
        },
        name="station_features",
    )


def align_feature_arrays(feature_arrays: Sequence[xr.DataArray]) -> xr.DataArray:
    """Align station feature arrays on their common time/station coordinates."""
    if not feature_arrays:
        raise ValueError("At least one feature array is required.")
    aligned = xr.align(*feature_arrays, join="inner", exclude={"feature"})
    return xr.concat(aligned, dim="feature").transpose("time", "station", "feature")


def _daily_or_raw_aggregate(
    dataset_token: str,
    start_date: str,
    end_date: str,
    aggregate_fn,
    prefer_daily_cache: bool,
    show_status: bool = False,
) -> xr.Dataset:
    if prefer_daily_cache:
        try:
            daily = open_daily_meteorological_dataset(
                dataset_token,
                start_date=start_date,
                end_date=end_date,
            )
            if show_status:
                print(f"Using precomputed daily cache: {dataset_token}")
            return daily
        except FileNotFoundError:
            if show_status:
                print(f"Daily cache not found for {dataset_token}; aggregating raw dataset")

    raw = open_meteorological_dataset(
        dataset_token,
        start_date=start_date,
        end_date=end_date,
    )
    return aggregate_fn(raw).sel(time=slice(start_date, end_date))


def build_station_feature_dataset(
    stations: Mapping[str, Sequence[float | str]],
    start_date: str,
    end_date: str,
    include_precipitation: bool = True,
    include_temperature: bool = True,
    include_specific_humidity: bool = True,
    include_wind: bool = True,
    include_vertical_velocity: bool = True,
    precipitation_token: str = "precipitation",
    temperature_token: str = "temp",
    specific_humidity_token: str = "sh",
    wind_token: str = "wind",
    vertical_velocity_token: str = "vv",
    show_progress: bool = False,
    progress_time_chunk_days: int = DEFAULT_PROGRESS_TIME_CHUNK_DAYS,
    prefer_daily_cache: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Build station-level xarray features and precipitation targets.

    The returned feature matrix keeps dimensions `[time, station, feature]`;
    the target keeps `[time, station]`. Both retain real timestamps.
    """
    feature_arrays: list[xr.DataArray] = []
    target_da: xr.DataArray | None = None

    if include_precipitation:
        precip_daily = _daily_or_raw_aggregate(
            precipitation_token,
            start_date,
            end_date,
            era5_daily_precip_all,
            prefer_daily_cache,
            show_status=show_progress,
        )
        precip_features = grid_dataset_to_station_features(
            precip_daily,
            stations,
            variables=["tp"],
            progress_label="precipitation",
            show_progress=show_progress,
            progress_time_chunk_days=progress_time_chunk_days,
        )
        feature_arrays.append(precip_features)
        target_da = precip_features.sel(feature="tp").drop_vars("feature", errors="ignore")

    if include_temperature:
        temp_daily = _daily_or_raw_aggregate(
            temperature_token,
            start_date,
            end_date,
            daily_temp_features,
            prefer_daily_cache,
            show_status=show_progress,
        )
        feature_arrays.append(
            grid_dataset_to_station_features(
                temp_daily,
                stations,
                progress_label="temperature",
                show_progress=show_progress,
                progress_time_chunk_days=progress_time_chunk_days,
            )
        )

    if include_specific_humidity:
        sh_daily = _daily_or_raw_aggregate(
            specific_humidity_token,
            start_date,
            end_date,
            daily_specific_humidity_features,
            prefer_daily_cache,
            show_status=show_progress,
        )
        feature_arrays.append(
            grid_dataset_to_station_features(
                sh_daily,
                stations,
                progress_label="specific humidity",
                show_progress=show_progress,
                progress_time_chunk_days=progress_time_chunk_days,
            )
        )

    if include_wind:
        wind_daily = _daily_or_raw_aggregate(
            wind_token,
            start_date,
            end_date,
            daily_wind_uv_features,
            prefer_daily_cache,
            show_status=show_progress,
        )
        feature_arrays.append(
            grid_dataset_to_station_features(
                wind_daily,
                stations,
                progress_label="wind",
                show_progress=show_progress,
                progress_time_chunk_days=progress_time_chunk_days,
            )
        )

    if include_vertical_velocity:
        vv_daily = _daily_or_raw_aggregate(
            vertical_velocity_token,
            start_date,
            end_date,
            lambda raw: daily_vertical_velocity(raw, "w", load_into_memory=False),
            prefer_daily_cache,
            show_status=show_progress,
        )
        feature_arrays.append(
            grid_dataset_to_station_features(
                vv_daily,
                stations,
                progress_label="vertical velocity",
                show_progress=show_progress,
                progress_time_chunk_days=progress_time_chunk_days,
            )
        )

    if target_da is None:
        raise ValueError("Precipitation must be included to build the target array.")

    X = align_feature_arrays(feature_arrays)
    X, target_da = xr.align(X, target_da, join="inner")
    return X.astype(np.float32), target_da.astype(np.float32)


def chronological_split(
    X: xr.DataArray,
    y: xr.DataArray,
    train_ratio: float,
    val_ratio: float,
) -> TimeSeriesSplits:
    """Split the dated time series before creating any overlapping windows."""
    if "time" not in X.dims or "time" not in y.dims:
        raise ValueError("X and y must keep a 'time' dimension.")
    X, y = xr.align(X, y, join="inner")

    if X.sizes["time"] != y.sizes["time"]:
        raise ValueError("X and y must have the same time length after alignment.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Use positive TRAIN_RATIO/VAL_RATIO with sum smaller than 1.")

    n_time = X.sizes["time"]
    train_end = int(np.floor(n_time * train_ratio))
    val_end = train_end + int(np.floor(n_time * val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n_time:
        raise ValueError("Split ratios leave at least one empty time block.")

    return TimeSeriesSplits(
        train_X=X.isel(time=slice(0, train_end)),
        train_y=y.isel(time=slice(0, train_end)),
        val_X=X.isel(time=slice(train_end, val_end)),
        val_y=y.isel(time=slice(train_end, val_end)),
        test_X=X.isel(time=slice(val_end, None)),
        test_y=y.isel(time=slice(val_end, None)),
    )


def _window_split(
    X: xr.DataArray,
    y: xr.DataArray,
    window_size: int,
    horizon: int,
    split_name: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    if window_size < 1 or horizon < 1:
        raise ValueError("window_size and horizon must be positive.")
    if X.sizes["time"] < window_size + horizon:
        raise ValueError(
            f"{split_name} split is too short for window_size={window_size} "
            f"and horizon={horizon}."
        )

    X_values = []
    y_values = []
    sample_start = []
    input_end = []
    target_dates = []
    time_values = pd.to_datetime(X.time.values)

    for target_idx in range(window_size, X.sizes["time"] - horizon + 1):
        X_values.append(X.isel(time=slice(target_idx - window_size, target_idx)).values)
        y_values.append(y.isel(time=slice(target_idx, target_idx + horizon)).values)
        sample_start.append(time_values[target_idx - window_size])
        input_end.append(time_values[target_idx - 1])
        target_dates.append(time_values[target_idx : target_idx + horizon])

    sample_coord = np.arange(len(X_values), dtype=np.int64)
    lead_coord = np.arange(1, horizon + 1, dtype=np.int64)
    lag_coord = np.arange(-window_size + 1, 1, dtype=np.int64)

    X_da = xr.DataArray(
        np.asarray(X_values, dtype=np.float32),
        dims=("sample", "lag", "station", "feature"),
        coords={
            "sample": sample_coord,
            "lag": lag_coord,
            "station": X.station.values,
            "feature": X.feature.values,
            "sample_start_time": ("sample", np.asarray(sample_start, dtype="datetime64[ns]")),
            "input_end_time": ("sample", np.asarray(input_end, dtype="datetime64[ns]")),
            "split": split_name,
        },
        name="X_windows",
    )
    y_da = xr.DataArray(
        np.asarray(y_values, dtype=np.float32),
        dims=("sample", "lead_day", "station"),
        coords={
            "sample": sample_coord,
            "lead_day": lead_coord,
            "station": y.station.values,
            "target_time": (
                ("sample", "lead_day"),
                np.asarray(target_dates, dtype="datetime64[ns]"),
            ),
            "input_end_time": ("sample", np.asarray(input_end, dtype="datetime64[ns]")),
            "split": split_name,
        },
        name="y_windows",
    )
    return X_da, y_da


def create_windowed_splits(
    splits: TimeSeriesSplits,
    window_size: int,
    horizon: int,
) -> WindowedSplits:
    """Create dated windows independently inside each chronological split."""
    train_X, train_y = _window_split(splits.train_X, splits.train_y, window_size, horizon, "train")
    val_X, val_y = _window_split(splits.val_X, splits.val_y, window_size, horizon, "validation")
    test_X, test_y = _window_split(splits.test_X, splits.test_y, window_size, horizon, "test")
    return WindowedSplits(train_X, train_y, val_X, val_y, test_X, test_y)


def _create_scaler(scaler_type: str) -> StandardScaler | MinMaxScaler:
    scaler_type = scaler_type.lower()
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler_type={scaler_type!r}. Use {SUPPORTED_SCALERS}.")


def _transform_dataarray_by_last_dim(
    data: xr.DataArray,
    scaler: StandardScaler | MinMaxScaler,
) -> xr.DataArray:
    values = np.asarray(data.values, dtype=np.float32)
    flat = values.reshape(-1, values.shape[-1])
    transformed = scaler.transform(flat).reshape(values.shape).astype(np.float32)
    return xr.DataArray(transformed, dims=data.dims, coords=data.coords, name=data.name)


def scale_windowed_splits(
    splits: WindowedSplits,
    normalize_features: bool = True,
    feature_scaler_type: str = "standard",
    normalize_target: bool = False,
    target_scaler_type: str = "standard",
) -> tuple[WindowedSplits, ScalingState]:
    """Fit scalers on train windows only and transform held-out windows."""
    feature_scaler = None
    target_scaler = None
    train_X, val_X, test_X = splits.train_X, splits.val_X, splits.test_X
    train_y, val_y, test_y = splits.train_y, splits.val_y, splits.test_y

    if normalize_features:
        feature_scaler = _create_scaler(feature_scaler_type)
        train_values = np.asarray(train_X.values, dtype=np.float32)
        feature_scaler.fit(train_values.reshape(-1, train_values.shape[-1]))
        train_X = _transform_dataarray_by_last_dim(train_X, feature_scaler)
        val_X = _transform_dataarray_by_last_dim(val_X, feature_scaler)
        test_X = _transform_dataarray_by_last_dim(test_X, feature_scaler)

    if normalize_target:
        target_scaler = _create_scaler(target_scaler_type)
        train_values = np.asarray(train_y.values, dtype=np.float32)
        target_scaler.fit(train_values.reshape(-1, 1))
        train_y = xr.DataArray(
            target_scaler.transform(train_y.values.reshape(-1, 1)).reshape(train_y.shape).astype(np.float32),
            dims=train_y.dims,
            coords=train_y.coords,
            name=train_y.name,
        )
        val_y = xr.DataArray(
            target_scaler.transform(val_y.values.reshape(-1, 1)).reshape(val_y.shape).astype(np.float32),
            dims=val_y.dims,
            coords=val_y.coords,
            name=val_y.name,
        )
        test_y = xr.DataArray(
            target_scaler.transform(test_y.values.reshape(-1, 1)).reshape(test_y.shape).astype(np.float32),
            dims=test_y.dims,
            coords=test_y.coords,
            name=test_y.name,
        )

    return (
        WindowedSplits(train_X, train_y, val_X, val_y, test_X, test_y),
        ScalingState(feature_scaler=feature_scaler, target_scaler=target_scaler),
    )


def to_torch_window_splits(splits: WindowedSplits, dtype: torch.dtype = torch.float32) -> TorchWindowSplits:
    """Convert dated xarray windows to torch tensors at the model boundary."""
    return TorchWindowSplits(
        X_train=torch.as_tensor(splits.train_X.values, dtype=dtype),
        y_train=torch.as_tensor(splits.train_y.values, dtype=dtype),
        X_val=torch.as_tensor(splits.val_X.values, dtype=dtype),
        y_val=torch.as_tensor(splits.val_y.values, dtype=dtype),
        X_test=torch.as_tensor(splits.test_X.values, dtype=dtype),
        y_test=torch.as_tensor(splits.test_y.values, dtype=dtype),
    )
