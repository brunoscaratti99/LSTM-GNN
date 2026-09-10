from pathlib import Path

import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "Datasets"
RAW_DATASETS_DIR = DATASETS_DIR / "raw"
PROCESSED_DATASETS_DIR = DATASETS_DIR / "processed"
DAILY_DATASETS_DIR = PROCESSED_DATASETS_DIR / "daily"
LEGACY_NC_DATASETS_DIR = DATASETS_DIR / "nc_files"


def ensure_dataset_directories():
    """Create the canonical dataset folders if they do not exist."""
    RAW_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "datasets": DATASETS_DIR,
        "raw": RAW_DATASETS_DIR,
        "processed": PROCESSED_DATASETS_DIR,
        "daily": DAILY_DATASETS_DIR,
        "legacy_nc": LEGACY_NC_DATASETS_DIR,
    }


def _unique_existing_or_declared(paths):
    seen = set()
    unique_paths = []
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        key = str(path.resolve()) if path.exists() else str(path.absolute())
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def _raw_search_dirs(raw_dir=None, include_legacy=True):
    dirs = [raw_dir, RAW_DATASETS_DIR]
    if include_legacy:
        dirs.append(LEGACY_NC_DATASETS_DIR)
    return _unique_existing_or_declared(dirs)


def _processed_search_dirs(processed_dir=None):
    return _unique_existing_or_declared([processed_dir, PROCESSED_DATASETS_DIR])


def _daily_search_dirs(daily_dir=None):
    return _unique_existing_or_declared([daily_dir, DAILY_DATASETS_DIR])


def _path_candidates(path_like, default_dirs):
    path = Path(path_like)
    candidates = [path]

    if not path.is_absolute():
        candidates.append(PROJECT_ROOT / path)

    for base_dir in default_dirs:
        candidates.append(Path(base_dir) / path.name)

    return _unique_existing_or_declared(candidates)


def _find_by_token(search_dirs, token, suffix):
    token = str(token).lower()
    suffix = suffix.lower()

    for base_dir in search_dirs:
        base_dir = Path(base_dir)
        if not base_dir.exists():
            continue

        matches = sorted(
            path
            for path in base_dir.iterdir()
            if path.name.lower().endswith(suffix) and token in path.name.lower()
        )
        if matches:
            return matches[0]

    return None


def find_raw_dataset(dataset, raw_dir=None, include_legacy=True):
    """
    Resolve a raw meteorological NetCDF file.

    `dataset` can be a file path, a file name, or a token such as
    "precipitation", "temp", "sh", "wind", or "vv".
    """
    ensure_dataset_directories()
    search_dirs = _raw_search_dirs(raw_dir=raw_dir, include_legacy=include_legacy)
    dataset_path = Path(str(dataset))

    if dataset_path.suffix.lower() == ".nc":
        for candidate in _path_candidates(dataset_path, search_dirs):
            if candidate.exists():
                return candidate
        return None

    return _find_by_token(search_dirs, dataset, ".nc")


def find_processed_dataset(dataset, processed_dir=None):
    """
    Resolve a processed Zarr dataset by path, raw file name, or variable token.
    """
    ensure_dataset_directories()
    search_dirs = _processed_search_dirs(processed_dir=processed_dir)
    dataset_path = Path(str(dataset))

    if dataset_path.suffix.lower() == ".zarr":
        for candidate in _path_candidates(dataset_path, search_dirs):
            if candidate.exists():
                return candidate
        return None

    if dataset_path.suffix.lower() == ".nc":
        zarr_name = dataset_path.with_suffix(".zarr").name
        for candidate in _path_candidates(zarr_name, search_dirs):
            if candidate.exists():
                return candidate
        return None

    return _find_by_token(search_dirs, dataset, ".zarr")


def zarr_path_for_raw(raw_path, processed_dir=None):
    processed_base = Path(processed_dir) if processed_dir is not None else PROCESSED_DATASETS_DIR
    return processed_base / f"{Path(raw_path).stem}.zarr"


def daily_zarr_path(dataset, daily_dir=None):
    """Return the canonical daily-aggregation Zarr path for a dataset token."""
    daily_base = Path(daily_dir) if daily_dir is not None else DAILY_DATASETS_DIR
    token = str(dataset).lower().replace(".zarr", "").replace(".nc", "")
    token = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in token)
    return daily_base / f"{token}_daily.zarr"


def find_daily_dataset(dataset, daily_dir=None):
    """Resolve a precomputed daily Zarr dataset by token or explicit path."""
    ensure_dataset_directories()
    search_dirs = _daily_search_dirs(daily_dir=daily_dir)
    dataset_path = Path(str(dataset))

    if dataset_path.suffix.lower() == ".zarr":
        for candidate in _path_candidates(dataset_path, search_dirs):
            if candidate.exists():
                return candidate
        return None

    canonical = daily_zarr_path(dataset, daily_dir=daily_dir)
    if canonical.exists():
        return canonical

    return _find_by_token(search_dirs, f"{dataset}_daily", ".zarr")


def _open_zarr_dataset(path, chunks=None):
    kwargs = {}
    if chunks is not None:
        kwargs["chunks"] = chunks
    return xr.open_zarr(path, **kwargs)


def _select_time_range(dataset, start_date=None, end_date=None):
    """Keep xarray data lazy while narrowing reads to the requested time range."""
    if start_date is None and end_date is None:
        return dataset
    if "time" not in dataset.dims and "time" not in dataset.coords:
        return dataset
    return dataset.sel(time=slice(start_date, end_date))


def open_meteorological_dataset(
    dataset,
    raw_dir=None,
    processed_dir=None,
    prefer_processed=True,
    include_legacy=True,
    engine="netcdf4",
    chunks=None,
    start_date=None,
    end_date=None,
    **open_dataset_kwargs,
):
    """
    Open a meteorological dataset using the repository data convention.

    The loader prefers `Datasets/processed/*.zarr` when available and falls
    back to `Datasets/raw/*.nc`. During migration it can also find legacy
    files in `Datasets/nc_files`. When `start_date` and/or `end_date` are
    provided, the returned xarray object is sliced before any caller-side
    materialization, so later `.values`/`.load()` calls only touch that range.
    """
    ensure_dataset_directories()

    if prefer_processed:
        processed_path = find_processed_dataset(dataset, processed_dir=processed_dir)
        if processed_path is not None:
            return _select_time_range(
                _open_zarr_dataset(processed_path, chunks=chunks),
                start_date=start_date,
                end_date=end_date,
            )

    raw_path = find_raw_dataset(dataset, raw_dir=raw_dir, include_legacy=include_legacy)
    if raw_path is not None:
        kwargs = dict(open_dataset_kwargs)
        if engine is not None:
            kwargs["engine"] = engine
        if chunks is not None:
            kwargs["chunks"] = chunks
        return _select_time_range(
            xr.open_dataset(raw_path, **kwargs),
            start_date=start_date,
            end_date=end_date,
        )

    if not prefer_processed:
        processed_path = find_processed_dataset(dataset, processed_dir=processed_dir)
        if processed_path is not None:
            return _select_time_range(
                _open_zarr_dataset(processed_path, chunks=chunks),
                start_date=start_date,
                end_date=end_date,
            )

    searched = {
        "raw": [str(path) for path in _raw_search_dirs(raw_dir, include_legacy=include_legacy)],
        "processed": [str(path) for path in _processed_search_dirs(processed_dir)],
    }
    raise FileNotFoundError(f"Dataset {dataset!r} not found. Searched: {searched}")


def open_daily_meteorological_dataset(
    dataset,
    daily_dir=None,
    chunks=None,
    start_date=None,
    end_date=None,
):
    """
    Open a precomputed daily meteorological Zarr dataset.

    These caches are generated by `src/precompute_daily_datasets.py` and store
    the expensive hourly/forecast-step to daily aggregation result.
    """
    daily_path = find_daily_dataset(dataset, daily_dir=daily_dir)
    if daily_path is None:
        searched = [str(path) for path in _daily_search_dirs(daily_dir)]
        raise FileNotFoundError(f"Daily dataset {dataset!r} not found. Searched: {searched}")
    return _select_time_range(
        _open_zarr_dataset(daily_path, chunks=chunks),
        start_date=start_date,
        end_date=end_date,
    )


def convert_raw_nc_to_zarr(
    dataset,
    raw_dir=None,
    processed_dir=None,
    overwrite=False,
    include_legacy=True,
    engine="netcdf4",
    chunks=None,
    consolidated=True,
    **to_zarr_kwargs,
):
    """
    Convert one raw NetCDF dataset to the processed Zarr cache.

    Returns the destination `.zarr` path. Existing caches are reused unless
    `overwrite=True`.
    """
    ensure_dataset_directories()
    raw_path = find_raw_dataset(dataset, raw_dir=raw_dir, include_legacy=include_legacy)
    if raw_path is None:
        raise FileNotFoundError(f"Raw NetCDF dataset {dataset!r} not found.")

    output_path = zarr_path_for_raw(raw_path, processed_dir=processed_dir)
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_kwargs = {"engine": engine}
    if chunks is not None:
        dataset_kwargs["chunks"] = chunks

    with xr.open_dataset(raw_path, **dataset_kwargs) as ds:
        ds.to_zarr(
            output_path,
            mode="w" if overwrite else "w-",
            consolidated=consolidated,
            **to_zarr_kwargs,
        )

    return output_path


def convert_all_raw_nc_to_zarr(
    raw_dir=None,
    processed_dir=None,
    overwrite=False,
    include_legacy=False,
    **kwargs,
):
    """Convert all top-level raw `.nc` files into `.zarr` stores."""
    ensure_dataset_directories()
    search_dirs = _raw_search_dirs(raw_dir=raw_dir, include_legacy=include_legacy)
    converted = []

    for base_dir in search_dirs:
        base_dir = Path(base_dir)
        if not base_dir.exists():
            continue
        for nc_path in sorted(base_dir.glob("*.nc")):
            converted.append(
                convert_raw_nc_to_zarr(
                    nc_path,
                    processed_dir=processed_dir,
                    overwrite=overwrite,
                    include_legacy=False,
                    **kwargs,
                )
            )

    return converted
