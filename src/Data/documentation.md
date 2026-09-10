# Data Package Documentation

`src/Data` owns dataset discovery, ERA5/INMET feature extraction, preprocessing, temporal splitting, dated window creation, scaling, and DataLoader construction. The package is intentionally the first stop for changes that affect meteorological inputs.

## Package-Level Conventions

- Raw meteorological inputs are NetCDF files (`*.nc`) under `Datasets/raw/` or the legacy `Datasets/nc_files/`.
- Processed datasets are Zarr stores (`*.zarr`) under `Datasets/processed/`.
- Daily aggregated caches live under `Datasets/processed/daily/`.
- Station inputs are INMET catalog/CSV files under `Datasets/dados_inmet/` or the legacy `Datasets/Dados INMET/`.
- Dated arrays should remain as `xarray.Dataset` or `xarray.DataArray` objects until the final Torch boundary.
- The main model-ready feature shape is `[time, station, feature]` before windowing and `[sample, lag, station, feature]` after windowing.
- Targets are `[time, station]` before windowing and `[sample, lead_day, station]` after windowing.

## Libraries Used

- `pathlib` and `os` for filesystem resolution.
- `xarray`, `netCDF4`, `zarr`, and `dask` for gridded meteorological datasets and lazy reads.
- `numpy` and `pandas` for numeric arrays, station catalogs, date handling, and reshaping.
- `torch` and `torch.utils.data` for tensors, TensorDatasets, and DataLoaders.
- `sklearn.preprocessing` for `StandardScaler` and `MinMaxScaler`.
- `dateutil.relativedelta` for annual/monthly interval slicing.
- `tqdm` is used indirectly by `temporal_dataset.py` for progress display when available.

## `dataset_paths.py`

Central dataset path and open/convert API. New data-reading code should use this file instead of hard-coded `xr.open_dataset(...)` paths.

Functions:

- `ensure_dataset_directories()`: creates the canonical dataset folders and returns their paths. Inputs: none. Output: dictionary with `datasets`, `raw`, `processed`, `daily`, and `legacy_nc` paths.
- `_unique_existing_or_declared(paths)`: internal deduplication helper for path lists. It resolves existing paths and keeps declared absolute paths for folders that do not exist yet.
- `_raw_search_dirs(raw_dir=None, include_legacy=True)`: builds the ordered raw NetCDF search path list, optionally including `Datasets/nc_files`.
- `_processed_search_dirs(processed_dir=None)`: builds the ordered processed Zarr search path list.
- `_daily_search_dirs(daily_dir=None)`: builds the ordered daily-cache search path list.
- `_path_candidates(path_like, default_dirs)`: expands a path, relative repository path, and default-directory candidate names.
- `_find_by_token(search_dirs, token, suffix)`: finds the first file/store whose lowercase name contains `token` and ends with `suffix`.
- `find_raw_dataset(dataset, raw_dir=None, include_legacy=True)`: resolves a raw NetCDF by path, file name, or token. Returns a `Path` or `None`.
- `find_processed_dataset(dataset, processed_dir=None)`: resolves a processed Zarr store by explicit path, raw file name, or token. Returns a `Path` or `None`.
- `zarr_path_for_raw(raw_path, processed_dir=None)`: maps a raw NetCDF path to its canonical processed `.zarr` destination.
- `daily_zarr_path(dataset, daily_dir=None)`: maps a dataset token to `Datasets/processed/daily/<token>_daily.zarr`.
- `find_daily_dataset(dataset, daily_dir=None)`: resolves a daily Zarr cache by explicit path or token.
- `_open_zarr_dataset(path, chunks=None)`: internal `xr.open_zarr` wrapper, preserving optional chunk arguments.
- `_select_time_range(dataset, start_date=None, end_date=None)`: lazily applies `time` slicing when the dataset has a time coordinate.
- `open_meteorological_dataset(...)`: main loader. Inputs may be a token, file name, or path. It prefers processed Zarr when requested, falls back to raw NetCDF, and optionally slices by date before materialization.
- `open_daily_meteorological_dataset(dataset, daily_dir=None, chunks=None, start_date=None, end_date=None)`: opens a precomputed daily Zarr cache and applies optional lazy date slicing.
- `convert_raw_nc_to_zarr(...)`: converts one raw NetCDF dataset into a processed Zarr store. Inputs include overwrite, chunking, and consolidation options. Output: destination `Path`.
- `convert_all_raw_nc_to_zarr(...)`: converts all top-level raw NetCDF files found in the raw search directories.

## `feature_extraction.py`

ERA5 and station-level feature helpers. This file still contains legacy tensor helpers, but new code should prefer the xarray-preserving functions used by `temporal_dataset.py`.

Functions:

- `smart_load_dataset(path, variable, prefer_processed=True)`: compatibility loader that delegates to `open_meteorological_dataset`. Inputs: directory/path hint and dataset token such as `precipitation` or `temp`. Output: `xarray.Dataset`.
- `haversine_km(lat1, lon1, lat2, lon2)`: computes great-circle distance in kilometers between two coordinates.
- `change_comma(frame)`: converts decimal-comma station coordinate columns to decimal-dot strings/numbers for downstream numeric use.
- `total_precipitation(data, lat, lon, time)`: extracts total precipitation for one date and nearest grid point.
- `tensor_data(t1, t2, era, stations_RS)`: legacy station tensor builder over a date interval.
- `get_data(dataset, lat, lon, t1, t2)`: extracts nearest-grid data between two date indices or labels.
- `_wrap_lon(lon, ds_lons)`: internal longitude convention adapter for datasets using `0..360` instead of `-180..180`.
- `uv_to_dir_speed(u, v, convention="meteorological")`: converts wind components to direction and speed.
- `era5_uv_to_tensor(...)`: opens wind NetCDF/Zarr data, aggregates to daily wind features, samples stations, and returns a tensor or xarray object.
- `daily_vertical_velocity(ds, var_name, percentiles=(10,), time_chunk_days=30, load_into_memory=True)`: aggregates vertical velocity to daily statistics, including percentile features.
- `get_vv(t1, t2, ds, stations)`: samples daily vertical velocity features at station coordinates.
- `daily_temp_features(ds, day_shift_hours=0)`: builds daily temperature/dew-point features such as max, mean, and min.
- `daily_specific_humidity_features(ds, var_name="q", day_shift_hours=0)`: builds daily specific-humidity max/mean/min features.
- `get_temp(t1, t2, ds_daily, stations)`: samples daily temperature features into a `[T, N, F]` tensor.
- `get_specific_humidity(t1, t2, ds_daily, stations)`: samples daily specific humidity into a `[T, N, 3]` tensor.
- `era5_specific_humidity_tensor(...)`: opens humidity data, builds daily features, and returns a station tensor.
- `daily_wind_uv_features(...)`: builds daily wind U/V statistics by pressure level.
- `_infer_coord_name(ds, candidates)`: finds the first matching coordinate/dimension name from a list.
- `get_wind_uv(...)`: samples daily wind features into a `[T, N, 12]` tensor by station.
- `forecast_steps_to_daily_precip(...)`: converts ERA5 forecast-step precipitation into daily precipitation totals.
- `era5_daily_precip(data, lat, lon)`: daily precipitation extraction for one nearest grid point.
- `day_index(dataset, start_date, index)`: maps an integer offset from a start date into a dataset time index.
- `station_dictionary(catalogo, UF="RS")`: converts an INMET catalog DataFrame into `{station_name: [lat, lon]}` filtered by state.
- `station_era(era, inmet, lat, lon)`: maps one station to the nearest ERA5 grid point.
- `tensor_data_old(t1, t2, era, stations_RS)`: older precipitation tensor construction helper retained for compatibility.
- `era5_daily_precip_all(data)`: converts an entire ERA5 precipitation dataset to daily precipitation over the grid.
- `daily_precip_dataset_to_tensor(tp_daily, stations, var_name="tp")`: samples a daily precipitation grid into `[days, station]`.
- `tensor_data_precip(data, t1, t2, stations)`: legacy precipitation tensor helper for a date interval.

## `file_utils.py`

Small filesystem and formatting helpers used by older experiments.

Functions:

- `change_comma(frame)`: decimal-comma normalization helper for tabular data.
- `format_path(path)`: escapes or normalizes path strings to avoid path separator conflicts.
- `create_next_experiment_folder(base_path)`: creates the next numbered experiment folder under a base path.

## `prepare_data.py`

Sliding-window and DataLoader helpers. This module supports both legacy tensor workflows and the newer source-level runner.

Functions:

- `create_sliding_windows(X, y, window_size, horizon=1, multi_step=False)`: converts time series into overlapping windows. Inputs are array-like tensors where the first axis is time. Output: `(Xs, ys)` windows.
- `_empty_window_split(X, y, window_size, horizon)`: creates correctly shaped empty splits when a time block cannot produce windows.
- `_create_windows_for_target_interval(X, y, window_size, horizon, start_idx, end_idx, use_context=True)`: internal helper that creates windows whose targets live inside a requested time interval.
- `temporal_train_val_test_split(...)`: split-before-windowing helper that avoids leakage across chronological blocks.
- `train_split(...)`: compatibility split helper that can either use legacy slicing or the leakage-safe temporal split when `window_size` is available.
- `create_batchs(...)`: wraps train/validation/test tensors in `TensorDataset` and `DataLoader`. Inputs follow `[B, T, N, F]` and `[B, H, N]`; `shuffle_train` controls only the training loader.
- `slice_intervalos_anuais(dataset, start_date, end_date, months, days)`: slices annual intervals from an xarray dataset using date offsets.

## `preprocessing.py`

Feature normalization and numerical-safety helpers.

Functions:

- `normalize_tp(tp, eps=1e-6)`: precipitation normalization helper.
- `add_consecutive_dry_days_feature(X, precip_col=0, dry_threshold=0.0)`: appends a dry-spell count feature per node.
- `normalize_features(X_local, scaler=StandardScaler)`: fits a scaler on flattened features and returns normalized data plus the fitted scaler.
- `assert_finite(name, tensor)`: raises when a tensor contains NaN or infinite values.
- `fit_log1p_zscore_stats(X_train_local, y_train_local, target_col=0, eps=1e-6)`: computes train-only log1p/z-score statistics for precipitation-like values.
- `apply_log1p_zscore(t, mean, std)`: applies the log1p/z-score transform.
- `inverse_log1p_zscore(t, mean, std)`: maps transformed values back to the original scale.

## `process_datasets.py`

Command-line interface for materializing processed Zarr caches from raw NetCDF datasets.

Functions:

- `_parse_args(argv=None)`: parses CLI arguments such as `--all`, selected datasets, `--overwrite`, and chunk settings.
- `_require_zarr()`: checks whether the optional Zarr dependency is installed before conversion.
- `main(argv=None)`: executes one or many NetCDF-to-Zarr conversions through `dataset_paths.py`.

## `temporal_dataset.py`

The maintained dated pipeline. This is the center of gravity for new experiment data preparation.

Classes:

- `_DatasetLoadProgressBar`: internal progress wrapper used while xarray chunks are materialized. It tracks dataset label, variable name, and time chunks.
- `TimeSeriesSplits`: dataclass containing `train_X`, `train_y`, `val_X`, `val_y`, `test_X`, and `test_y` as dated xarray arrays before windowing.
- `WindowedSplits`: dataclass containing dated xarray train/validation/test windows and targets after windowing.
- `TorchWindowSplits`: dataclass containing final Torch tensors for model training/evaluation.
- `ScalingState`: dataclass containing fitted train-only feature and target scalers, or `None` when normalization is disabled.

Functions:

- `_time_chunk_slices(n_time, chunk_size)`: creates time-axis slices for controlled materialization.
- `_load_time_chunks(data, progress, time_chunk_size)`: loads xarray data in time chunks and updates the progress bar.
- `load_station_catalog(catalog_path, state="RS")`: reads the INMET catalog and returns a station coordinate dictionary.
- `_feature_label(variable, extra_dims, extra_index)`: builds stable feature names when non-time/station dimensions are flattened.
- `grid_dataset_to_station_features(...)`: samples gridded ERA5 variables at station coordinates and returns `[time, station, feature]`.
- `align_feature_arrays(feature_arrays)`: aligns multiple feature arrays on common time and station coordinates, excluding the feature axis from alignment.
- `_daily_or_raw_aggregate(...)`: uses a daily cache when available, otherwise opens raw data and applies an aggregation function.
- `build_station_feature_dataset(...)`: builds the full station feature matrix and precipitation target array for the requested date range and feature switches.
- `chronological_split(X, y, train_ratio, val_ratio)`: splits the dated time axis before creating windows.
- `_window_split(X, y, window_size, horizon, split_name)`: creates windows inside one split and adds sample/input/target date coordinates.
- `create_windowed_splits(splits, window_size, horizon)`: creates train/validation/test windows independently.
- `_create_scaler(scaler_type)`: resolves `"standard"` or `"minmax"` to a scikit-learn scaler.
- `target_standard_deviation_to_physical_scale(values, target_scaler=None)`: converts target-scale standard deviations back to physical units without applying the scaler offset (`* scale_` for standard scaling and `/ scale_` for min-max scaling).
- `_transform_dataarray_by_last_dim(data, scaler)`: transforms the last dimension of an xarray array while preserving dims and coords.
- `scale_windowed_splits(...)`: fits scalers on training windows only and transforms held-out windows.
- `to_torch_window_splits(splits, dtype=torch.float32)`: converts dated xarray windows to model-ready Torch tensors.
