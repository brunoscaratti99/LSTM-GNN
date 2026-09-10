"""Create exploratory plots for ERA5 meteorological variables.

Edit ``dataset``, ``features``, ``plots``, and ``temporal_resolution`` below
to select the analysis. Results are saved in ``Dataset_Plots`` at the project
root.

ERA5 data are large spatial grids. The script samples the ERA5 grid point
nearest to ``station_id`` and creates all plots from that station. Precipitation
is converted from metres to millimetres and summed at the selected resolution;
``t2m`` and ``d2m`` are converted from Kelvin to Celsius and averaged.
Quantile thresholds are only available for daily precipitation.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib

# Allow execution in environments without a graphical interface.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Selection parameters
# ---------------------------------------------------------------------------
dataset = "era5"
features = ["precipitation"]
plots = ["histogram", "timeseries", "scatter", "boxplot"]
station_id = "PORTO ALEGRE JARDIM BOTANICO"  # Name used by PLOT_STATION_NAME in run_experiment.py.
temporal_resolution = "day"  # Options: "hour", "day", "month", or "year".
quantiles = [0.3,0.5,0.7,0.9,0.99]  # Daily precipitation quantile thresholds.


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Dataset_Plots"
SRC_DIR = ROOT / "src"
STATION_STATE = "RS"
CATALOG_PATH = ROOT / "Datasets" / "dados_inmet" / "Catalogo*.csv"
MAX_SCATTER_POINTS = 10_000
TEMPORAL_FREQUENCIES = {
    "hour": "1h",
    "day": "1D",
    "month": "MS",
    "year": "YS",
}
TEMPORAL_LABELS = {
    "hour": "Hourly",
    "day": "Daily",
    "month": "Monthly",
    "year": "Yearly",
}
TEMPORAL_UNITS = {
    "hour": "hours",
    "day": "days",
    "month": "months",
    "year": "years",
}

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Data.dataset_paths import open_meteorological_dataset  # noqa: E402
from Data.temporal_dataset import load_station_catalog  # noqa: E402
from Training.experiment_runner import resolve_catalog_path  # noqa: E402


FEATURE_SPECS = {
    "precipitation": {
        "dataset_token": "precipitation",
        "variables": ("tp", "precipitation"),
        "label": "Precipitation",
        "color": "#1976D2",
    },
    "t2m": {
        "dataset_token": "temp",
        "variables": ("t2m",),
        "label": "2 m temperature",
        "color": "#E64A19",
    },
    "d2m": {
        "dataset_token": "temp",
        "variables": ("d2m",),
        "label": "2 m dewpoint temperature",
        "color": "#6A1B9A",
    },
}
SUPPORTED_PLOTS = {"histogram", "timeseries", "scatter", "boxplot"}


def _validate_choices() -> tuple[list[str], list[str], str, list[float]]:
    """Validate the editable configuration at the top of the file."""
    if dataset.casefold() != "era5":
        raise ValueError("Only the 'era5' dataset is configured in this script.")

    selected_features = [str(feature).casefold() for feature in features]
    unknown_features = sorted(set(selected_features) - FEATURE_SPECS.keys())
    if unknown_features:
        raise ValueError(
            "Unsupported features: "
            f"{', '.join(unknown_features)}. Available: {', '.join(FEATURE_SPECS)}."
        )
    if not selected_features:
        raise ValueError("Select at least one feature.")

    selected_plots = [str(plot).casefold() for plot in plots]
    unknown_plots = sorted(set(selected_plots) - SUPPORTED_PLOTS)
    if unknown_plots:
        raise ValueError(
            "Unsupported plot types: "
            f"{', '.join(unknown_plots)}. Available: {', '.join(sorted(SUPPORTED_PLOTS))}."
        )
    if not selected_plots:
        raise ValueError("Select at least one plot type.")

    selected_resolution = str(temporal_resolution).casefold()
    if selected_resolution not in TEMPORAL_FREQUENCIES:
        raise ValueError(
            "Unsupported temporal_resolution: "
            f"{temporal_resolution!r}. Available: {', '.join(TEMPORAL_FREQUENCIES)}."
        )

    try:
        selected_quantiles = [float(quantile) for quantile in quantiles]
    except (TypeError, ValueError) as error:
        raise ValueError("'quantiles' must contain only numeric values between 0 and 1.") from error
    invalid_quantiles = [quantile for quantile in selected_quantiles if not 0 <= quantile <= 1]
    if invalid_quantiles:
        raise ValueError(
            "Every value in 'quantiles' must be between 0 and 1. "
            f"Invalid values: {invalid_quantiles}."
        )

    return (
        list(dict.fromkeys(selected_features)),
        list(dict.fromkeys(selected_plots)),
        selected_resolution,
        list(dict.fromkeys(selected_quantiles)),
    )


def _feature_unit(feature: str, resolution: str) -> str:
    """Return the display unit for a feature at the selected resolution."""
    if feature == "precipitation":
        return f"mm/{resolution}"
    return "°C"


def _station_lookup_key(value: object) -> str:
    """Normalize station names so catalog punctuation is not significant."""
    return "".join(character for character in str(value).upper() if character.isalnum())


def _resolve_station() -> tuple[str, float, float]:
    """Resolve ``station_id`` through the same INMET catalog flow as the runner."""
    stations = load_station_catalog(
        resolve_catalog_path(CATALOG_PATH),
        state=STATION_STATE,
    )
    if not stations:
        raise ValueError(f"No stations were found for state {STATION_STATE!r}.")

    requested_key = _station_lookup_key(station_id)
    matches = [
        (name, coordinates)
        for name, coordinates in stations.items()
        if _station_lookup_key(name) == requested_key
    ]
    if len(matches) != 1:
        available = ", ".join(list(stations)[:10])
        raise ValueError(
            f"station_id={station_id!r} was not found in the {STATION_STATE} INMET catalog. "
            f"Example station names: {available}."
        )

    station_name, coordinates = matches[0]
    return station_name, float(coordinates[0]), float(coordinates[1])


def _coordinate_name(data: xr.DataArray, candidates: tuple[str, ...]) -> str:
    """Find the latitude or longitude coordinate name used by an ERA5 array."""
    for candidate in candidates:
        if candidate in data.coords or candidate in data.dims:
            return candidate
    raise ValueError(f"No coordinate was found among: {', '.join(candidates)}.")


def _wrap_station_longitude(longitude: float, data_longitudes: xr.DataArray) -> float:
    """Adapt an INMET longitude to the ERA5 longitude convention."""
    min_longitude = float(data_longitudes.min())
    max_longitude = float(data_longitudes.max())
    if max_longitude > 180 and longitude < 0:
        return longitude + 360
    if min_longitude < 0 and longitude > 180:
        return longitude - 360
    return longitude


def _select_station_grid_point(
    data: xr.DataArray,
    station_latitude: float,
    station_longitude: float,
) -> xr.DataArray:
    """Select the ERA5 grid point nearest to the configured INMET station."""
    latitude_name = _coordinate_name(data, ("latitude", "lat"))
    longitude_name = _coordinate_name(data, ("longitude", "lon"))
    longitude = _wrap_station_longitude(station_longitude, data[longitude_name])
    return data.sel(
        {latitude_name: station_latitude, longitude_name: longitude},
        method="nearest",
    )


def _resolve_variable(data: xr.Dataset, feature: str) -> xr.DataArray:
    """Return the ERA5 variable associated with the selected feature."""
    spec = FEATURE_SPECS[feature]
    for variable in spec["variables"]:
        if variable in data.data_vars:
            return data[variable]
    raise KeyError(
        f"No variable for '{feature}' was found. "
        f"Expected: {spec['variables']}; available: {list(data.data_vars)}."
    )


def _precipitation_valid_time_series(data: xr.DataArray) -> xr.DataArray:
    """Return precipitation indexed by ERA5 valid time in millimetres.

    ERA5 total precipitation commonly has ``time`` and ``step`` dimensions.
    ``valid_time`` maps each forecast step to the physical timestamp. Duplicate
    valid times are resolved by retaining the last forecast value, matching the
    precipitation processing convention used elsewhere in this project.
    """
    if "step" not in data.dims:
        return data
    if "valid_time" not in data.coords:
        raise ValueError(
            "ERA5 precipitation with a 'step' dimension must provide a 'valid_time' coordinate."
        )

    flattened = data.stack(sample=("time", "step"))
    valid_times = flattened["valid_time"].data
    flattened = flattened.reset_index("sample", drop=True)
    flattened = flattened.assign_coords(time=("sample", valid_times))
    flattened = flattened.swap_dims({"sample": "time"}).sortby("time")
    return flattened.drop_vars("valid_time").groupby("time").last()


def _temporal_station_series(
    data: xr.DataArray, feature: str, resolution: str
) -> xr.DataArray:
    """Aggregate an ERA5 station grid point into a temporal series.

    Precipitation forecast steps are first indexed by their ``valid_time``;
    this creates an hourly series before summing it at the selected temporal
    resolution. Temperatures use their native timestamps and are averaged at
    the selected resolution.
    """
    if "time" not in data.dims:
        raise ValueError(f"The feature '{feature}' does not have a 'time' dimension.")

    frequency = TEMPORAL_FREQUENCIES[resolution]
    if feature == "precipitation":
        series = data * 1_000.0  # ERA5 precipitation: metres -> millimetres.
        series = _precipitation_valid_time_series(series)
        return series.resample(time=frequency).sum(skipna=True).rename(feature)

    series = data - 273.15  # ERA5 t2m and d2m: Kelvin -> Celsius.
    return series.resample(time=frequency).mean(skipna=True).rename(feature)


def _load_temporal_series(
    selected_features: list[str],
    resolution: str,
    station_latitude: float,
    station_longitude: float,
) -> dict[str, xr.DataArray]:
    """Open required datasets and materialize compact temporal series."""
    opened_datasets: dict[str, xr.Dataset] = {}
    temporal_series: dict[str, xr.DataArray] = {}

    try:
        for feature in selected_features:
            spec = FEATURE_SPECS[feature]
            token = spec["dataset_token"]
            if token not in opened_datasets:
                opened_datasets[token] = open_meteorological_dataset(
                    token,
                    prefer_processed=True,
                    chunks={"time": 365},
                )

            station_data = _select_station_grid_point(
                _resolve_variable(opened_datasets[token], feature),
                station_latitude,
                station_longitude,
            )
            series = _temporal_station_series(station_data, feature, resolution)
            # The source grid is read in chunks; only the station series is
            # retained in memory.
            temporal_series[feature] = series.compute()
    finally:
        for data in opened_datasets.values():
            data.close()

    return temporal_series


def _finite_values(series: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite timestamps and values from a materialized time series."""
    values = np.asarray(series.values, dtype=float)
    times = np.asarray(series["time"].values)
    valid = np.isfinite(values)
    return times[valid], values[valid]


def precipitation_quantile_thresholds(
    temporal_series: dict[str, xr.DataArray],
    resolution: str,
    selected_quantiles: list[float],
) -> dict[float, float]:
    """Calculate daily precipitation quantile thresholds in mm/day."""
    if resolution != "day" or "precipitation" not in temporal_series or not selected_quantiles:
        return {}

    _, precipitation_values = _finite_values(temporal_series["precipitation"])
    if not len(precipitation_values):
        raise ValueError("The daily precipitation series has no finite values.")
    return {
        quantile: float(np.quantile(precipitation_values, quantile))
        for quantile in selected_quantiles
    }


def _add_precipitation_thresholds(
    axis: plt.Axes,
    thresholds: dict[float, float],
    direction: str,
) -> None:
    """Add daily-precipitation quantile threshold lines to an axis."""
    color_map = plt.get_cmap("viridis")
    for index, (quantile, threshold) in enumerate(thresholds.items()):
        percent = quantile * 100
        line_kwargs = {
            "color": color_map(0.25 + 0.6 * index / max(len(thresholds) - 1, 1)),
            "linestyle": "--",
            "linewidth": 1.4,
            "label": f"Q{percent:g}: {threshold:.2f} mm/day",
        }
        if direction == "x":
            axis.axvline(threshold, **line_kwargs)
        elif direction == "y":
            axis.axhline(threshold, **line_kwargs)
        else:
            raise ValueError("'direction' must be either 'x' or 'y'.")


def _save_figure(figure: plt.Figure, filename: str) -> Path:
    """Save a figure with consistent dimensions and close its resources."""
    output_path = OUTPUT_DIR / filename
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _station_file_token(station_name: str) -> str:
    """Return a portable station identifier for plot file names."""
    token = "".join(character.lower() if character.isalnum() else "_" for character in station_name)
    return token.strip("_") or "station"


def plot_histograms(
    temporal_series: dict[str, xr.DataArray],
    resolution: str,
    precipitation_thresholds: dict[float, float],
    station_name: str,
) -> list[Path]:
    """Create one temporal-distribution histogram per selected feature."""
    saved_paths = []
    for feature, series in temporal_series.items():
        _, values = _finite_values(series)
        spec = FEATURE_SPECS[feature]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(values, bins="auto", color=spec["color"], alpha=0.82, edgecolor="white")
        axis.set_title(f"Histogram - {spec['label']} (ERA5)\nStation: {station_name}")
        axis.set_xlabel(_feature_unit(feature, resolution))
        axis.set_ylabel(f"Frequency ({TEMPORAL_UNITS[resolution]})")
        axis.grid(axis="y", alpha=0.25)
        if feature == "precipitation" and precipitation_thresholds:
            _add_precipitation_thresholds(axis, precipitation_thresholds, direction="x")
            axis.legend(title="Daily threshold")
        saved_paths.append(
            _save_figure(
                figure,
                f"era5_{_station_file_token(station_name)}_{feature}_{resolution}_histogram.png",
            )
        )
    return saved_paths


def plot_timeseries(
    temporal_series: dict[str, xr.DataArray],
    resolution: str,
    precipitation_thresholds: dict[float, float],
    station_name: str,
) -> list[Path]:
    """Create a temporal-series panel with one axis per selected feature."""
    figure, axes = plt.subplots(
        nrows=len(temporal_series),
        ncols=1,
        figsize=(12, 3.6 * len(temporal_series)),
        sharex=True,
        squeeze=False,
    )

    for axis, (feature, series) in zip(axes[:, 0], temporal_series.items()):
        spec = FEATURE_SPECS[feature]
        times, values = _finite_values(series)
        axis.plot(times, values, color=spec["color"], linewidth=0.85)
        axis.set_ylabel(_feature_unit(feature, resolution))
        axis.set_title(spec["label"])
        axis.grid(alpha=0.25)
        if feature == "precipitation" and precipitation_thresholds:
            _add_precipitation_thresholds(axis, precipitation_thresholds, direction="y")
            axis.legend(title="Daily threshold")

    axes[-1, 0].set_xlabel("Date")
    figure.suptitle(
        f"{TEMPORAL_LABELS[resolution]} time series - ERA5\nStation: {station_name}",
        y=1.02,
    )
    return [
        _save_figure(
            figure,
            f"era5_{_station_file_token(station_name)}_{resolution}_timeseries.png",
        )
    ]


def plot_scatters(
    temporal_series: dict[str, xr.DataArray],
    resolution: str,
    precipitation_thresholds: dict[float, float],
    station_name: str,
) -> list[Path]:
    """Create scatter plots for every pair of selected features."""
    saved_paths = []
    for x_feature, y_feature in combinations(temporal_series, 2):
        x_series, y_series = xr.align(
            temporal_series[x_feature], temporal_series[y_feature], join="inner"
        )
        values = np.column_stack(
            (np.asarray(x_series.values, dtype=float), np.asarray(y_series.values, dtype=float))
        )
        values = values[np.isfinite(values).all(axis=1)]
        if not len(values):
            raise ValueError(f"No coincident observations exist for {x_feature} and {y_feature}.")
        if len(values) > MAX_SCATTER_POINTS:
            indices = np.linspace(0, len(values) - 1, MAX_SCATTER_POINTS, dtype=int)
            values = values[indices]

        x_spec = FEATURE_SPECS[x_feature]
        y_spec = FEATURE_SPECS[y_feature]
        figure, axis = plt.subplots(figsize=(7, 5.5))
        axis.scatter(
            values[:, 0],
            values[:, 1],
            s=10,
            alpha=0.35,
            color=y_spec["color"],
            edgecolors="none",
        )
        axis.set_title(
            f"Scatter plot - {x_spec['label']} vs. {y_spec['label']} (ERA5)\n"
            f"Station: {station_name}"
        )
        axis.set_xlabel(f"{x_spec['label']} ({_feature_unit(x_feature, resolution)})")
        axis.set_ylabel(f"{y_spec['label']} ({_feature_unit(y_feature, resolution)})")
        axis.grid(alpha=0.25)
        if precipitation_thresholds and x_feature == "precipitation":
            _add_precipitation_thresholds(axis, precipitation_thresholds, direction="x")
            axis.legend(title="Daily threshold")
        elif precipitation_thresholds and y_feature == "precipitation":
            _add_precipitation_thresholds(axis, precipitation_thresholds, direction="y")
            axis.legend(title="Daily threshold")
        saved_paths.append(
            _save_figure(
                figure,
                f"era5_{_station_file_token(station_name)}_"
                f"{x_feature}_vs_{y_feature}_{resolution}_scatter.png",
            )
        )
    return saved_paths


def plot_boxplots(
    temporal_series: dict[str, xr.DataArray],
    resolution: str,
    precipitation_thresholds: dict[float, float],
    station_name: str,
) -> list[Path]:
    """Create seasonal box plots for the selected features."""
    saved_paths = []
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for feature, series in temporal_series.items():
        spec = FEATURE_SPECS[feature]
        figure, axis = plt.subplots(figsize=(11, 5))

        if resolution == "year":
            _, values = _finite_values(series)
            axis.boxplot([values], tick_labels=["All years"], showfliers=False)
            axis.set_title(f"Yearly box plot - {spec['label']} (ERA5)\nStation: {station_name}")
            axis.set_xlabel("Aggregation")
        else:
            values_by_month = []
            for month in range(1, 13):
                monthly_values = np.asarray(
                    series.where(series.time.dt.month == month, drop=True).values, dtype=float
                )
                values_by_month.append(monthly_values[np.isfinite(monthly_values)])
            axis.boxplot(values_by_month, tick_labels=month_labels, showfliers=False)
            axis.set_title(
                f"Calendar-month box plot - {spec['label']} (ERA5)\nStation: {station_name}"
            )
            axis.set_xlabel("Calendar month")

        axis.set_ylabel(_feature_unit(feature, resolution))
        axis.grid(axis="y", alpha=0.25)
        if feature == "precipitation" and precipitation_thresholds:
            _add_precipitation_thresholds(axis, precipitation_thresholds, direction="y")
            axis.legend(title="Daily threshold")
        saved_paths.append(
            _save_figure(
                figure,
                f"era5_{_station_file_token(station_name)}_{feature}_{resolution}_boxplot.png",
            )
        )
    return saved_paths


def main() -> list[Path]:
    """Generate the selected plots and return their output paths."""
    selected_features, selected_plots, selected_resolution, selected_quantiles = _validate_choices()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    station_name, station_latitude, station_longitude = _resolve_station()
    temporal_series = _load_temporal_series(
        selected_features,
        selected_resolution,
        station_latitude,
        station_longitude,
    )
    precipitation_thresholds = precipitation_quantile_thresholds(
        temporal_series, selected_resolution, selected_quantiles
    )

    saved_paths: list[Path] = []
    if "histogram" in selected_plots:
        saved_paths.extend(
            plot_histograms(
                temporal_series,
                selected_resolution,
                precipitation_thresholds,
                station_name,
            )
        )
    if "timeseries" in selected_plots:
        saved_paths.extend(
            plot_timeseries(
                temporal_series,
                selected_resolution,
                precipitation_thresholds,
                station_name,
            )
        )
    if "scatter" in selected_plots:
        if len(temporal_series) < 2:
            print("Scatter plot skipped: select at least two features.")
        else:
            saved_paths.extend(
                plot_scatters(
                    temporal_series,
                    selected_resolution,
                    precipitation_thresholds,
                    station_name,
                )
            )
    if "boxplot" in selected_plots:
        saved_paths.extend(
            plot_boxplots(
                temporal_series,
                selected_resolution,
                precipitation_thresholds,
                station_name,
            )
        )

    if precipitation_thresholds:
        print("Daily precipitation quantile thresholds:")
        for quantile, threshold in precipitation_thresholds.items():
            print(f"- Q{quantile * 100:g}: {threshold:.2f} mm/day")
    elif selected_quantiles and "precipitation" in temporal_series and selected_resolution != "day":
        print("Quantile thresholds were skipped because they are only defined for daily precipitation.")

    print(
        f"Station: {station_name} "
        f"(latitude={station_latitude:.4f}, longitude={station_longitude:.4f})"
    )
    print(f"Saved {len(saved_paths)} plot(s) to: {OUTPUT_DIR}")
    for path in saved_paths:
        print(f"- {path.name}")
    return saved_paths


if __name__ == "__main__":
    main()
