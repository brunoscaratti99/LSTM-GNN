"""Precompute daily meteorological aggregation caches.

Run from the repository root, for example:

    python src/precompute_daily_datasets.py
    python src/precompute_daily_datasets.py precipitation temp sh --overwrite

The outputs are saved under `Datasets/processed/daily/*.zarr` and are consumed
automatically by `run_experiment.py` when `USE_DAILY_CACHE=True`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Callable

import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from Data.dataset_paths import daily_zarr_path, open_meteorological_dataset
from Data.feature_extraction import (
    daily_specific_humidity_features,
    daily_temp_features,
    daily_vertical_velocity,
    daily_wind_uv_features,
    era5_daily_precip_all,
)

try:
    from dask.diagnostics import ProgressBar
except ImportError:  # pragma: no cover - dask is listed in requirements.
    ProgressBar = None


DEFAULT_DATASETS = ("precipitation", "temp", "sh", "wind", "vv")
DATASET_ALIASES = {
    "precip": "precipitation",
    "rain": "precipitation",
    "temperature": "temp",
    "specific_humidity": "sh",
    "humidity": "sh",
    "vertical_velocity": "vv",
}
AGGREGATORS: dict[str, Callable[[xr.Dataset], xr.Dataset]] = {
    "precipitation": era5_daily_precip_all,
    "temp": daily_temp_features,
    "sh": daily_specific_humidity_features,
    "wind": daily_wind_uv_features,
    "vv": lambda ds: daily_vertical_velocity(ds, "w", load_into_memory=False),
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Precompute daily aggregated meteorological Zarr datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=(
            "Dataset tokens to aggregate. Defaults to all: "
            f"{', '.join(DEFAULT_DATASETS)}."
        ),
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional first date to include before aggregation, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional last date to include before aggregation, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing daily Zarr caches.",
    )
    parser.add_argument(
        "--chunks-time",
        type=int,
        default=None,
        help="Optional chunk size for the source time dimension.",
    )
    parser.add_argument(
        "--daily-chunks-time",
        type=int,
        default=365,
        help="Chunk size for the output daily time dimension.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable dask progress output while writing Zarr stores.",
    )
    return parser.parse_args(argv)


def _normalize_dataset_token(dataset: str) -> str:
    token = str(dataset).strip().lower()
    token = DATASET_ALIASES.get(token, token)
    if token not in AGGREGATORS:
        valid = ", ".join(sorted(AGGREGATORS))
        raise ValueError(f"Unsupported dataset token {dataset!r}. Use one of: {valid}.")
    return token


def _slice_time(dataset: xr.Dataset, start_date: str | None, end_date: str | None) -> xr.Dataset:
    if start_date is None and end_date is None:
        return dataset
    if "time" not in dataset.dims and "time" not in dataset.coords:
        return dataset
    return dataset.sel(time=slice(start_date, end_date))


def _write_zarr(dataset: xr.Dataset, output_path: Path, overwrite: bool, show_progress: bool) -> Path:
    if output_path.exists():
        if not overwrite:
            print(f"[skip] {output_path} already exists")
            return output_path
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write = lambda: dataset.to_zarr(output_path, mode="w", consolidated=True)
    if show_progress and ProgressBar is not None:
        with ProgressBar():
            write()
    else:
        write()
    return output_path


def precompute_daily_dataset(
    dataset: str,
    start_date: str | None = None,
    end_date: str | None = None,
    overwrite: bool = False,
    chunks_time: int | None = None,
    daily_chunks_time: int | None = 365,
    show_progress: bool = True,
) -> Path:
    token = _normalize_dataset_token(dataset)
    output_path = daily_zarr_path(token)
    if output_path.exists() and not overwrite:
        print(f"[skip] {token}: {output_path}")
        return output_path

    chunks = {"time": chunks_time} if chunks_time is not None else None
    print(f"[open] {token}")
    raw = open_meteorological_dataset(
        token,
        chunks=chunks,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"[aggregate daily] {token}")
    daily = AGGREGATORS[token](raw)
    daily = _slice_time(daily, start_date, end_date)
    daily.attrs.update(
        {
            "precomputed_daily_cache": "true",
            "source_dataset_token": token,
            "source_start_date": "" if start_date is None else str(start_date),
            "source_end_date": "" if end_date is None else str(end_date),
        }
    )
    if daily_chunks_time is not None and "time" in daily.dims:
        daily = daily.chunk({"time": daily_chunks_time})

    print(f"[write] {output_path}")
    return _write_zarr(daily, output_path, overwrite=overwrite, show_progress=show_progress)


def main(argv=None) -> None:
    args = _parse_args(argv)
    datasets = args.datasets or list(DEFAULT_DATASETS)
    outputs = [
        precompute_daily_dataset(
            dataset,
            start_date=args.start_date,
            end_date=args.end_date,
            overwrite=args.overwrite,
            chunks_time=args.chunks_time,
            daily_chunks_time=args.daily_chunks_time,
            show_progress=not args.no_progress,
        )
        for dataset in datasets
    ]
    print("\nDaily caches ready:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
