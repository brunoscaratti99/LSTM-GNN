"""Export Rio Grande do Sul precipitation maps as Beamer animation frames.

The public entry point reads the dated test predictions already saved by a
trained run. One forecast origin (``sample``) becomes one animation: each lead
day is rendered to matching ``Predict``, ``Real``, and ``Residual`` folders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gzip
import json
from pathlib import Path
import shutil
import urllib.request
from uuid import uuid4

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


RS_IBGE_GEOJSON_URL = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/estados/43"
    "?formato=application/vnd.geo+json"
)

PRECIPITATION_CMAP = LinearSegmentedColormap.from_list(
    "rs_precipitation",
    ["#ffffff", "#d9efff", "#82c5ff", "#2185d0", "#004a99", "#002b67"],
)

# Keep a narrow white band around zero so negligible errors do not look like
# either over- or underprediction.
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "rs_predicted_minus_real_residual",
    [
        (0.00, "#2166ac"),
        (0.43, "#dbe9f6"),
        (0.48, "#ffffff"),
        (0.52, "#ffffff"),
        (0.57, "#f8dddd"),
        (1.00, "#b2182b"),
    ],
)


def load_rs_state_geojson(
    *,
    url: str = RS_IBGE_GEOJSON_URL,
    cache_path: Path | str | None = None,
) -> dict[str, object]:
    """Load the RS boundary from IBGE, optionally reusing a local JSON cache."""
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        with open(cache, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a GeoJSON object in {cache}.")
        return payload

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, application/geo+json",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The IBGE response is not a GeoJSON object.")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as file:
            json.dump(payload, file)
    return payload


def _iter_geometries(payload: Mapping[str, object]):
    payload_type = payload.get("type")
    if payload_type == "FeatureCollection":
        features = payload.get("features", [])
        if not isinstance(features, list):
            raise ValueError("GeoJSON FeatureCollection has an invalid features field.")
        for feature in features:
            if isinstance(feature, Mapping):
                yield from _iter_geometries(feature)
        return
    if payload_type == "Feature":
        geometry = payload.get("geometry")
        if isinstance(geometry, Mapping):
            yield from _iter_geometries(geometry)
        return
    if payload_type == "GeometryCollection":
        geometries = payload.get("geometries", [])
        if not isinstance(geometries, list):
            raise ValueError("GeoJSON GeometryCollection has an invalid geometries field.")
        for geometry in geometries:
            if isinstance(geometry, Mapping):
                yield from _iter_geometries(geometry)
        return
    if payload_type in {"Polygon", "MultiPolygon"}:
        yield payload


def _geojson_polygons(payload: Mapping[str, object]) -> list[list[np.ndarray]]:
    polygons: list[list[np.ndarray]] = []
    for geometry in _iter_geometries(payload):
        geometry_type = geometry["type"]
        coordinates = geometry.get("coordinates")
        raw_polygons = coordinates if geometry_type == "MultiPolygon" else [coordinates]
        if not isinstance(raw_polygons, list):
            continue
        for raw_polygon in raw_polygons:
            if not isinstance(raw_polygon, list) or not raw_polygon:
                continue
            rings = []
            for raw_ring in raw_polygon:
                ring = np.asarray(raw_ring, dtype=float)
                if ring.ndim != 2 or ring.shape[0] < 3 or ring.shape[1] < 2:
                    raise ValueError("The RS GeoJSON contains an invalid polygon ring.")
                if not np.all(np.isfinite(ring[:, :2])):
                    raise ValueError("The RS GeoJSON contains non-finite coordinates.")
                rings.append(ring[:, :2])
            if rings:
                polygons.append(rings)
    if not polygons:
        raise ValueError("The supplied GeoJSON contains no Polygon or MultiPolygon geometry.")
    return polygons


def _resolve_predictions_csv(run_dir: Path, predictions_csv: Path | str | None) -> Path:
    if predictions_csv is not None:
        path = Path(predictions_csv).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Prediction CSV not found: {path}")
        return path
    candidates = (
        run_dir / "test_predictions_by_lead_day.csv",
        run_dir / "inference_predictions_by_lead_day.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "The run has no test_predictions_by_lead_day.csv or "
        "inference_predictions_by_lead_day.csv."
    )


def _prediction_columns(path: Path) -> tuple[str, str, list[str]]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    predicted = "predicted_mm" if "predicted_mm" in columns else "predicted"
    actual = "actual_mm" if "actual_mm" in columns else "actual"
    required = ["sample", "lead_day", "target_time", "station", actual, predicted]
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {', '.join(missing)}")
    return actual, predicted, required


def _last_sample(path: Path) -> int:
    latest: int | None = None
    for chunk in pd.read_csv(path, usecols=["sample"], chunksize=250_000):
        values = pd.to_numeric(chunk["sample"], errors="coerce").dropna()
        if not values.empty:
            chunk_latest = int(values.max())
            latest = chunk_latest if latest is None else max(latest, chunk_latest)
    if latest is None:
        raise ValueError("Prediction CSV contains no sample identifiers.")
    return latest


def _read_prediction_sample(path: Path, sample: int) -> tuple[pd.DataFrame, str, str, int]:
    actual_column, predicted_column, columns = _prediction_columns(path)
    resolved_sample = _last_sample(path) if sample == -1 else int(sample)
    if resolved_sample < 0:
        raise ValueError("sample must be a non-negative CSV sample id, or -1 for the last sample.")

    selected = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000):
        sample_values = pd.to_numeric(chunk["sample"], errors="coerce")
        matching = chunk.loc[sample_values == resolved_sample]
        if not matching.empty:
            selected.append(matching)
    if not selected:
        raise ValueError(f"Sample {resolved_sample} does not exist in {path}.")
    frame = pd.concat(selected, ignore_index=True)
    frame["lead_day"] = pd.to_numeric(frame["lead_day"], errors="raise").astype(int)
    if (frame["lead_day"] < 1).any():
        raise ValueError("lead_day values must start at 1.")
    if frame.duplicated(["lead_day", "station"]).any():
        raise ValueError("The selected sample has duplicate lead-day/station rows.")
    return frame, actual_column, predicted_column, resolved_sample


def _read_prediction_period(
    path: Path,
    *,
    start_date: str,
    end_date: str,
    lead_day: int,
) -> tuple[pd.DataFrame, str, str, pd.Timestamp, pd.Timestamp]:
    actual_column, predicted_column, columns = _prediction_columns(path)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if pd.isna(start) or pd.isna(end):
        raise ValueError("start_date and end_date must be valid dates.")
    if start > end:
        raise ValueError("start_date must not be after end_date.")
    resolved_lead = int(lead_day)
    if resolved_lead < 1:
        raise ValueError("lead_day must be a positive integer starting at 1.")

    selected = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000):
        lead_values = pd.to_numeric(chunk["lead_day"], errors="coerce")
        target_times = pd.to_datetime(chunk["target_time"], errors="coerce").dt.normalize()
        matching = chunk.loc[
            (lead_values == resolved_lead) & target_times.between(start, end, inclusive="both")
        ].copy()
        if not matching.empty:
            selected.append(matching)
    if not selected:
        raise ValueError(
            f"No D+{resolved_lead} predictions exist between "
            f"{start.date().isoformat()} and {end.date().isoformat()} in {path}."
        )
    frame = pd.concat(selected, ignore_index=True)
    frame["sample"] = pd.to_numeric(frame["sample"], errors="raise").astype(int)
    frame["lead_day"] = pd.to_numeric(frame["lead_day"], errors="raise").astype(int)
    frame["target_time"] = pd.to_datetime(frame["target_time"], errors="raise").dt.normalize()
    return frame, actual_column, predicted_column, start, end


def _source_run_candidates(run_dir: Path) -> list[Path]:
    candidates = [run_dir]
    config_path = run_dir / "inference_config.json"
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        source_run = config.get("source_run") if isinstance(config, Mapping) else None
        if source_run:
            candidates.append(Path(str(source_run)).expanduser().resolve())
    if run_dir.parent.name == "inference":
        candidates.append(run_dir.parent.parent)
    return list(dict.fromkeys(candidates))


def _load_station_coordinates(run_dir: Path) -> dict[str, tuple[float, float]]:
    searched = []
    for candidate in _source_run_candidates(run_dir):
        state_path = candidate / "inference_state.json"
        searched.append(str(state_path))
        if not state_path.is_file():
            continue
        with open(state_path, "r", encoding="utf-8") as file:
            state = json.load(file)
        raw_coordinates = state.get("station_coordinates") if isinstance(state, Mapping) else None
        if not isinstance(raw_coordinates, Mapping):
            raise ValueError(f"{state_path} has no station_coordinates mapping.")
        coordinates = {}
        for station, raw in raw_coordinates.items():
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 2:
                raise ValueError(f"Invalid coordinates for station {station!r} in {state_path}.")
            latitude, longitude = float(raw[0]), float(raw[1])
            if not np.isfinite(latitude) or not np.isfinite(longitude):
                raise ValueError(f"Non-finite coordinates for station {station!r}.")
            coordinates[str(station)] = (latitude, longitude)
        return coordinates
    raise FileNotFoundError(
        "Could not find inference_state.json with station coordinates. Searched: "
        + ", ".join(searched)
    )


def _validated_lead_frames(
    prediction_rows: pd.DataFrame,
    station_coordinates: Mapping[str, tuple[float, float]],
    actual_column: str,
    predicted_column: str,
) -> list[dict[str, object]]:
    expected_stations = set(station_coordinates)
    frames = []
    for lead_day, rows in prediction_rows.groupby("lead_day", sort=True):
        row_stations = set(rows["station"].astype(str))
        missing = sorted(expected_stations - row_stations)
        unexpected = sorted(row_stations - expected_stations)
        if missing or unexpected:
            raise ValueError(
                f"Station mismatch at D+{lead_day}: missing={missing}, unexpected={unexpected}."
            )
        ordered_stations = list(station_coordinates)
        ordered = rows.assign(station=rows["station"].astype(str)).set_index("station").loc[ordered_stations]
        predicted = pd.to_numeric(ordered[predicted_column], errors="coerce").to_numpy(dtype=float)
        actual = pd.to_numeric(ordered[actual_column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(predicted)):
            raise ValueError(f"Predicted precipitation contains NaN/inf at D+{lead_day}.")
        if not np.all(np.isfinite(actual)):
            raise ValueError(
                "Real frames require historical targets, but actual precipitation contains "
                f"NaN/inf at D+{lead_day}. Use a backtest/test prediction CSV."
            )
        target_times = ordered["target_time"].dropna().astype(str).unique()
        if len(target_times) != 1:
            raise ValueError(f"Expected one target_time for all stations at D+{lead_day}.")
        frames.append(
            {
                "lead_day": int(lead_day),
                "target_time": str(target_times[0]),
                "predicted": predicted,
                "actual": actual,
            }
        )
    if not frames:
        raise ValueError("The selected sample has no lead-day rows.")
    lead_days = [int(frame["lead_day"]) for frame in frames]
    expected_leads = list(range(1, max(lead_days) + 1))
    if lead_days != expected_leads:
        raise ValueError(f"Lead days must be consecutive from 1; got {lead_days}.")
    return frames


def _validated_period_frames(
    prediction_rows: pd.DataFrame,
    station_coordinates: Mapping[str, tuple[float, float]],
    actual_column: str,
    predicted_column: str,
    lead_day: int,
) -> list[dict[str, object]]:
    expected_stations = set(station_coordinates)
    ordered_stations = list(station_coordinates)
    frames = []
    for sample, rows in prediction_rows.groupby("sample", sort=True):
        row_stations = set(rows["station"].astype(str))
        missing = sorted(expected_stations - row_stations)
        unexpected = sorted(row_stations - expected_stations)
        if missing or unexpected:
            raise ValueError(
                f"Station mismatch for sample {sample} at D+{lead_day}: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if rows.duplicated(["station"]).any():
            raise ValueError(f"Sample {sample} has duplicate station rows at D+{lead_day}.")
        ordered = rows.assign(station=rows["station"].astype(str)).set_index("station").loc[ordered_stations]
        predicted = pd.to_numeric(ordered[predicted_column], errors="coerce").to_numpy(dtype=float)
        actual = pd.to_numeric(ordered[actual_column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(predicted)):
            raise ValueError(f"Predicted precipitation contains NaN/inf for sample {sample}.")
        if not np.all(np.isfinite(actual)):
            raise ValueError(
                "Real frames require historical targets, but actual precipitation contains "
                f"NaN/inf for sample {sample}. Use a backtest/test prediction CSV."
            )
        target_times = pd.DatetimeIndex(ordered["target_time"].dropna().unique())
        if len(target_times) != 1:
            raise ValueError(f"Expected one target_time for all stations in sample {sample}.")
        frames.append(
            {
                "sample": int(sample),
                "lead_day": int(lead_day),
                "target_time": target_times[0],
                "predicted": predicted,
                "actual": actual,
            }
        )
    if not frames:
        raise ValueError("The selected period has no prediction frames.")
    frames.sort(key=lambda frame: (pd.Timestamp(frame["target_time"]), int(frame["sample"])))
    target_times = [pd.Timestamp(frame["target_time"]) for frame in frames]
    duplicated_dates = pd.DatetimeIndex(target_times).duplicated(keep=False)
    if duplicated_dates.any():
        duplicates = sorted({time.date().isoformat() for time, duplicate in zip(target_times, duplicated_dates) if duplicate})
        raise ValueError(
            "The selected lead day has more than one sample for target date(s): "
            + ", ".join(duplicates)
        )
    return frames


def _draw_state_boundary(
    ax,
    polygons: list[list[np.ndarray]],
    *,
    facecolor: str = "#f7f9fb",
) -> tuple[float, float, float, float]:
    all_points = []
    for rings in polygons:
        exterior = rings[0]
        ax.fill(
            exterior[:, 0],
            exterior[:, 1],
            facecolor=facecolor,
            edgecolor="#4b5563",
            linewidth=0.9,
            zorder=1,
        )
        for hole in rings[1:]:
            ax.fill(hole[:, 0], hole[:, 1], facecolor="white", edgecolor="#4b5563", linewidth=0.5, zorder=2)
        all_points.append(exterior)
    points = np.concatenate(all_points, axis=0)
    return (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
    )


def _save_map_frame(
    path: Path,
    *,
    polygons: list[list[np.ndarray]],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: LinearSegmentedColormap,
    frame_label: str,
    date_label: str,
    show_colorbar: bool,
    colorbar_label: str | None,
    clip_negative_values: bool,
    node_size: float,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    figure, axis = plt.subplots(figsize=figsize, facecolor="white")
    axis.set_facecolor("white")
    lon_min, lon_max, lat_min, lat_max = _draw_state_boundary(
        axis,
        polygons,
        facecolor="white",
    )
    normalization = Normalize(vmin=vmin, vmax=vmax, clip=True)
    nodes = axis.scatter(
        longitudes,
        latitudes,
        c=np.maximum(values, 0.0) if clip_negative_values else values,
        cmap=cmap,
        norm=normalization,
        s=node_size,
        marker="o",
        edgecolors="#17324d",
        linewidths=0.45,
        zorder=3,
    )

    lon_padding = max((lon_max - lon_min) * 0.035, 0.1)
    lat_padding = max((lat_max - lat_min) * 0.035, 0.1)
    axis.set_xlim(lon_min - lon_padding, lon_max + lon_padding)
    axis.set_ylim(lat_min - lat_padding, lat_max + lat_padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    axis.set_title(
        f"{frame_label}\n{date_label}",
        fontsize=11,
        fontweight="normal",
        linespacing=1.0,
        pad=5,
    )
    # Keep an identical map viewport for every map. The reserved strip stays
    # blank when no colorbar is shown.
    figure.subplots_adjust(left=0.02, right=0.88, bottom=0.02, top=0.88)
    if show_colorbar:
        colorbar_axis = figure.add_axes((0.90, 0.13, 0.022, 0.74))
        colorbar = figure.colorbar(nodes, cax=colorbar_axis)
        if colorbar_label is not None:
            colorbar.set_label(colorbar_label)
        colorbar.outline.set_linewidth(0.6)
    figure.savefig(
        path,
        dpi=dpi,
        facecolor="white",
        transparent=False,
        bbox_inches=figure.bbox_inches,
        pad_inches=0,
    )
    plt.close(figure)


def _write_beamer_example(path: Path, digits: int, final_lead: int, fps: float) -> None:
    animation_frames = []
    residual_animation_frames = []
    for frame_number in range(1, final_lead + 1):
        filename = f"frame_{frame_number:0{digits}d}.png"
        animation_frames.append(
            rf"""  \makebox[\linewidth][c]{{%
    \includegraphics[width=0.48\linewidth]{{Predict/{filename}}}\hfill
    \includegraphics[width=0.48\linewidth]{{Real/{filename}}}%
            }}"""
        )
        residual_animation_frames.append(
            rf"""  \makebox[\linewidth][c]{{%
    \includegraphics[width=0.78\linewidth]{{Residual/{filename}}}%
  }}"""
        )
    body = "\n  \\newframe\n".join(animation_frames)
    residual_body = "\n  \\newframe\n".join(residual_animation_frames)
    text = rf"""\documentclass[aspectratio=169]{{beamer}}
\usepackage{{graphicx}}
\usepackage{{animate}}
\setbeamertemplate{{navigation symbols}}{{}}

\begin{{document}}
\begin{{frame}}{{Precipita\c{{c}}\~ao prevista e observada no RS}}
\centering
\begin{{minipage}}{{0.86\linewidth}}
\centering
\begin{{animateinline}}[
  label=rsrain,
  poster=first,
  autoplay,
  autopause,
  autoresume,
  loop,
  controls=all,
  controlsaligned=center
]{{{fps:g}}}
{body}
\end{{animateinline}}
\end{{minipage}}
\end{{frame}}
\begin{{frame}}{{Res\'iduo de precipita\c{{c}}\~ao: predito - real}}
\centering
\begin{{minipage}}{{0.86\linewidth}}
\centering
\begin{{animateinline}}[
  label=rsresidual,
  poster=first,
  autoplay,
  autopause,
  autoresume,
  loop,
  controls=all,
  controlsaligned=center
]{{{fps:g}}}
{residual_body}
\end{{animateinline}}
\end{{minipage}}
\end{{frame}}
\end{{document}}
"""
    path.write_text(text, encoding="utf-8")


def save_rs_precipitation_animation_frames(
    run_dir: Path | str,
    *,
    sample: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lead_day: int | None = None,
    output_dir: Path | str | None = None,
    predictions_csv: Path | str | None = None,
    boundary_geojson: Mapping[str, object] | None = None,
    boundary_path: Path | str | None = None,
    vmax: float | None = None,
    residual_vmax: float | None = None,
    node_size: float = 74.0,
    dpi: int = 190,
    figsize: tuple[float, float] = (7.2, 7.0),
    fps: float = 2.0,
) -> Path:
    """Save predicted, real, and residual RS precipitation animation frames.

    Parameters
    ----------
    run_dir:
        Trained run (or inference output) containing a prediction CSV. Station
        coordinates are read from the run's ``inference_state.json``.
    sample:
        Legacy sample mode: exact sample id in the CSV, or ``-1`` for the last
        sample. When the period parameters are omitted, ``None`` selects sample
        0 and each frame represents one lead day.
    start_date / end_date:
        Inclusive target-date period to animate. Both are required together
        with ``lead_day``. Each frame then represents one chronological target
        date at the fixed lead day.
    lead_day:
        Forecast lead to compare throughout ``start_date``..``end_date``.
    output_dir:
        Destination whose children will be ``Predict``, ``Real``, and
        ``Residual``. By default, ``<run>/rs_animation_frames/sample_XXXXX``
        is used.
    boundary_geojson / boundary_path:
        Optional RS GeoJSON supplied directly or from disk. When omitted, the
        same IBGE endpoint used by ``Seminar_GLSTM/RS_state_map.py`` is called.
    vmax:
        Shared maximum of the white-to-blue scale in millimetres. When omitted,
        it is computed over both predicted and real values in all lead days.
    residual_vmax:
        Positive magnitude of the symmetric residual scale in millimetres.
        Residuals are calculated as predicted minus real; positive values are
        red, negative values are blue, and values near zero are white. When
        omitted, the magnitude is computed across all selected frames.

    The destination must not already exist, preventing accidental replacement
    of previously generated presentation frames.
    """
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise NotADirectoryError(f"Run directory not found: {run_path}")
    if boundary_geojson is not None and boundary_path is not None:
        raise ValueError("Pass boundary_geojson or boundary_path, not both.")
    if node_size <= 0 or dpi <= 0 or fps <= 0:
        raise ValueError("node_size, dpi, and fps must be positive.")
    if len(figsize) != 2 or min(figsize) <= 0:
        raise ValueError("figsize must contain two positive values.")

    period_arguments = (start_date, end_date, lead_day)
    period_mode = any(value is not None for value in period_arguments)
    if period_mode and not all(value is not None for value in period_arguments):
        raise ValueError("start_date, end_date, and lead_day must be provided together.")
    if period_mode and sample is not None:
        raise ValueError("sample cannot be combined with start_date/end_date/lead_day.")

    csv_path = _resolve_predictions_csv(run_path, predictions_csv)
    if period_mode:
        assert start_date is not None and end_date is not None and lead_day is not None
        prediction_rows, actual_column, predicted_column, period_start, period_end = (
            _read_prediction_period(
                csv_path,
                start_date=start_date,
                end_date=end_date,
                lead_day=lead_day,
            )
        )
        resolved_sample = None
    else:
        selected_sample = 0 if sample is None else int(sample)
        prediction_rows, actual_column, predicted_column, resolved_sample = _read_prediction_sample(
            csv_path, selected_sample
        )
        period_start = period_end = None
    station_coordinates = _load_station_coordinates(run_path)
    if period_mode:
        lead_frames = _validated_period_frames(
            prediction_rows,
            station_coordinates,
            actual_column,
            predicted_column,
            int(lead_day),
        )
    else:
        lead_frames = _validated_lead_frames(
            prediction_rows,
            station_coordinates,
            actual_column,
            predicted_column,
        )

    if boundary_path is not None:
        with open(Path(boundary_path), "r", encoding="utf-8") as file:
            loaded_boundary = json.load(file)
        if not isinstance(loaded_boundary, Mapping):
            raise ValueError("boundary_path must contain a GeoJSON object.")
        boundary_geojson = loaded_boundary
    if boundary_geojson is None:
        boundary_geojson = load_rs_state_geojson()
    polygons = _geojson_polygons(boundary_geojson)

    all_values = np.concatenate(
        [
            np.asarray(frame[kind], dtype=float)
            for frame in lead_frames
            for kind in ("predicted", "actual")
        ]
    )
    inferred_vmax = float(max(np.max(np.maximum(all_values, 0.0)), 1.0))
    color_vmax = inferred_vmax if vmax is None else float(vmax)
    if not np.isfinite(color_vmax) or color_vmax <= 0:
        raise ValueError("vmax must be a positive finite precipitation value.")
    all_residuals = np.concatenate(
        [
            np.asarray(frame["predicted"], dtype=float)
            - np.asarray(frame["actual"], dtype=float)
            for frame in lead_frames
        ]
    )
    inferred_residual_vmax = float(max(np.max(np.abs(all_residuals)), 1.0))
    color_residual_vmax = (
        inferred_residual_vmax if residual_vmax is None else float(residual_vmax)
    )
    if not np.isfinite(color_residual_vmax) or color_residual_vmax <= 0:
        raise ValueError("residual_vmax must be a positive finite residual value.")

    if period_mode:
        assert period_start is not None and period_end is not None and lead_day is not None
        selection_directory = (
            f"lead_day_{int(lead_day):02d}__"
            f"{period_start.date().isoformat()}__{period_end.date().isoformat()}"
        )
    else:
        assert resolved_sample is not None
        selection_directory = f"sample_{resolved_sample:05d}"
    default_destination = run_path / "rs_animation_frames" / selection_directory
    destination = (
        default_destination if output_dir is None else Path(output_dir).expanduser().resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")

    latitudes = np.asarray([coordinates[0] for coordinates in station_coordinates.values()])
    longitudes = np.asarray([coordinates[1] for coordinates in station_coordinates.values()])
    digits = max(3, len(str(len(lead_frames))))
    temporary = destination.parent / f".{destination.name}_{uuid4().hex}"
    temporary.mkdir(exist_ok=False)
    try:
        predict_dir = temporary / "Predict"
        real_dir = temporary / "Real"
        residual_dir = temporary / "Residual"
        predict_dir.mkdir()
        real_dir.mkdir()
        residual_dir.mkdir()
        manifest_frames = []
        clipped_counts = {"predicted": 0, "actual": 0}
        for frame_number, frame in enumerate(lead_frames, start=1):
            filename = f"frame_{frame_number:0{digits}d}.png"
            lead_day = int(frame["lead_day"])
            target_time = pd.Timestamp(frame["target_time"]).date().isoformat()
            predicted = np.asarray(frame["predicted"], dtype=float)
            actual = np.asarray(frame["actual"], dtype=float)
            clipped_counts["predicted"] += int(np.sum(predicted < 0))
            clipped_counts["actual"] += int(np.sum(actual < 0))
            _save_map_frame(
                predict_dir / filename,
                polygons=polygons,
                latitudes=latitudes,
                longitudes=longitudes,
                values=predicted,
                vmin=0.0,
                vmax=color_vmax,
                cmap=PRECIPITATION_CMAP,
                frame_label="Predict",
                date_label=target_time,
                show_colorbar=False,
                colorbar_label=None,
                clip_negative_values=True,
                node_size=node_size,
                dpi=dpi,
                figsize=figsize,
            )
            _save_map_frame(
                real_dir / filename,
                polygons=polygons,
                latitudes=latitudes,
                longitudes=longitudes,
                values=actual,
                vmin=0.0,
                vmax=color_vmax,
                cmap=PRECIPITATION_CMAP,
                frame_label="Real",
                date_label=target_time,
                show_colorbar=True,
                colorbar_label="Precipitation (mm)",
                clip_negative_values=True,
                node_size=node_size,
                dpi=dpi,
                figsize=figsize,
            )
            residual = predicted - actual
            _save_map_frame(
                residual_dir / filename,
                polygons=polygons,
                latitudes=latitudes,
                longitudes=longitudes,
                values=residual,
                vmin=-color_residual_vmax,
                vmax=color_residual_vmax,
                cmap=RESIDUAL_CMAP,
                frame_label="Residual: Predicted - Real (mm)",
                date_label=target_time,
                show_colorbar=True,
                colorbar_label="Residual (mm)",
                clip_negative_values=False,
                node_size=node_size,
                dpi=dpi,
                figsize=figsize,
            )
            manifest_frames.append(
                {
                    "frame": frame_number,
                    "sample": int(frame.get("sample", resolved_sample)),
                    "lead_day": lead_day,
                    "target_time": target_time,
                    "predict": f"Predict/{filename}",
                    "real": f"Real/{filename}",
                    "residual": f"Residual/{filename}",
                }
            )

        _write_beamer_example(temporary / "beamer_animate_example.tex", digits, len(lead_frames), fps)
        manifest = {
            "schema_version": 2,
            "source_run": str(run_path),
            "predictions_csv": str(csv_path),
            "selection_mode": "fixed_lead_period" if period_mode else "sample_horizon",
            "sample": resolved_sample,
            "start_date": None if period_start is None else period_start.date().isoformat(),
            "end_date": None if period_end is None else period_end.date().isoformat(),
            "lead_day": None if not period_mode else int(lead_day),
            "frame_count": len(lead_frames),
            "frame_digits": digits,
            "frame_prefix": "frame_",
            "color_scale_mm": {"min": 0.0, "max": color_vmax},
            "residual_color_scale_mm": {
                "min": -color_residual_vmax,
                "max": color_residual_vmax,
            },
            "residual_formula": "predicted_mm - actual_mm",
            "negative_values_rendered_as_zero": clipped_counts,
            "node_marker": "circle",
            "edges_rendered": False,
            "frames": manifest_frames,
        }
        with open(temporary / "animation_manifest.json", "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def salvar_frames_animacao_rs(*args, **kwargs) -> Path:
    """Portuguese alias for :func:`save_rs_precipitation_animation_frames`."""
    return save_rs_precipitation_animation_frames(*args, **kwargs)
