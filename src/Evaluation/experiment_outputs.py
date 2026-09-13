"""Output writers for dated graph-temporal precipitation experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import re
import unicodedata

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from Data.temporal_dataset import target_standard_deviation_to_physical_scale
from output_layout import logs_directory
from Evaluation.metrics import (
    METRIC_STANDARD_MODIFIED,
    normalize_metric_standard,
    numpy_rain_classification_metrics,
    numpy_regression_metrics,
    validate_metric_threshold,
)
from Evaluation.plot_style import (
    REFERENCE_COLOR,
    apply_seaborn_theme,
    lead_day_legend_labels,
    prediction_palette,
    save_figure,
    style_axis,
    style_time_axis,
)
from Evaluation.rs_animation_maps import (
    _draw_state_boundary,
    _geojson_polygons,
    load_rs_state_geojson,
)


apply_seaborn_theme()


OVERSMOOTHING_DIRECTORY = "oversmoothing_diagnostics"
_RS_BOUNDARY_POLYGONS: list[list[np.ndarray]] | None = None


def _as_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _model_adjacency_matrix(model, normalized: bool = False) -> np.ndarray | None:
    adjacency_fn = getattr(model, "current_adjacency", None)
    if callable(adjacency_fn):
        return _as_numpy(adjacency_fn(normalized=normalized))

    first_cell = getattr(model, "cell_0", None)
    adjacency_fn = getattr(first_cell, "current_adjacency", None)
    if callable(adjacency_fn):
        return _as_numpy(adjacency_fn(normalized=normalized))

    return None


def _heatmap_ticks(n_items: int) -> np.ndarray:
    if n_items <= 30:
        return np.arange(n_items)
    step = int(np.ceil(n_items / 20))
    return np.arange(0, n_items, step)


def save_topology_heatmap(
    run_dir: Path,
    model,
    station_names: list[str] | tuple[str, ...],
    filename: str = "final_adjacency.png",
    title: str = "Final model adjacency matrix",
) -> Path | None:
    """Save a heatmap of the model adjacency matrix in its current state."""
    adjacency = _model_adjacency_matrix(model, normalized=False)
    if adjacency is None:
        return None

    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency matrix must be square. Got shape={adjacency.shape}.")

    run_dir = Path(run_dir)
    output_path = run_dir / filename
    labels = [str(name) for name in station_names]
    if len(labels) != adjacency.shape[0]:
        labels = [f"station_{idx}" for idx in range(adjacency.shape[0])]

    fig_size = max(8.0, min(18.0, 0.28 * adjacency.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    tick_step = 1 if adjacency.shape[0] <= 30 else int(np.ceil(adjacency.shape[0] / 20))
    sns.heatmap(
        pd.DataFrame(adjacency, index=labels, columns=labels),
        ax=ax,
        cmap=sns.color_palette("mako", as_cmap=True),
        square=True,
        linewidths=0.0,
        xticklabels=tick_step,
        yticklabels=tick_step,
        cbar_kws={"label": "Adjacency weight", "fraction": 0.046, "pad": 0.04},
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Destination station")
    ax.set_ylabel("Source station")
    ax.set_title(title)
    save_figure(fig, output_path, dpi=190)
    plt.close(fig)
    return output_path


def _rs_boundary_polygons(boundary_geojson: Mapping[str, object] | None = None) -> list[list[np.ndarray]]:
    """Return plot-ready RS polygons, caching the default IBGE response in memory."""
    if boundary_geojson is not None:
        return _geojson_polygons(boundary_geojson)

    global _RS_BOUNDARY_POLYGONS
    if _RS_BOUNDARY_POLYGONS is None:
        _RS_BOUNDARY_POLYGONS = _geojson_polygons(load_rs_state_geojson())
    return _RS_BOUNDARY_POLYGONS


def _validated_positions(pos: Mapping[int, object]) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    for raw_node, raw_coordinates in pos.items():
        node = int(raw_node)
        coordinates = np.asarray(raw_coordinates, dtype=float).reshape(-1)
        if coordinates.size < 2 or not np.all(np.isfinite(coordinates[:2])):
            raise ValueError(f"Node {node} must have finite longitude/latitude coordinates.")
        positions[node] = (float(coordinates[0]), float(coordinates[1]))

    expected_nodes = set(range(len(positions)))
    if set(positions) != expected_nodes:
        raise ValueError("pos keys must be the contiguous node indices 0..N-1.")
    return positions


def _undirected_edges(edge_index, n_nodes: int) -> list[tuple[int, int]]:
    edges = np.asarray(_as_numpy(edge_index), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E]. Got {edges.shape}.")

    undirected_edges: set[tuple[int, int]] = set()
    for source, destination in edges.T:
        source = int(source)
        destination = int(destination)
        if not (0 <= source < n_nodes and 0 <= destination < n_nodes):
            raise ValueError(f"edge_index contains a node outside 0..{n_nodes - 1}.")
        if source != destination:
            undirected_edges.add(tuple(sorted((source, destination))))
    return sorted(undirected_edges)


def _prepare_rs_graph_axis(
    positions: Mapping[int, tuple[float, float]],
    polygons: list[list[np.ndarray]],
):
    figure, axis = plt.subplots(figsize=(10.5, 9.5), facecolor="white")
    lon_min, lon_max, lat_min, lat_max = _draw_state_boundary(axis, polygons)
    lon_padding = max((lon_max - lon_min) * 0.035, 0.1)
    lat_padding = max((lat_max - lat_min) * 0.035, 0.1)
    axis.set_xlim(lon_min - lon_padding, lon_max + lon_padding)
    axis.set_ylim(lat_min - lat_padding, lat_max + lat_padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()

    graph = nx.Graph()
    graph.add_nodes_from(positions)
    return figure, axis, graph


def _draw_station_nodes(axis, graph: nx.Graph, positions: Mapping[int, tuple[float, float]]) -> None:
    """Draw unlabeled station markers for the geographic RS graph figures."""
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color="#2a9d8f",
        edgecolors="#17324d",
        linewidths=0.8,
        node_size=210,
        ax=axis,
    )


def save_graph_plot(
    run_dir: Path,
    edge_index,
    pos: Mapping[int, object],
    filename: str = "graph.png",
    *,
    boundary_geojson: Mapping[str, object] | None = None,
) -> Path:
    """Save the initial station graph over the Rio Grande do Sul boundary."""
    positions = _validated_positions(pos)
    edges = _undirected_edges(edge_index, len(positions))
    polygons = _rs_boundary_polygons(boundary_geojson)
    figure, axis, graph = _prepare_rs_graph_axis(positions, polygons)
    graph.add_edges_from(edges)
    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color="#4f8f83",
        width=2.2,
        alpha=0.82,
        ax=axis,
    )
    _draw_station_nodes(axis, graph, positions)

    output_path = Path(run_dir) / filename
    save_figure(figure, output_path, dpi=190)
    plt.close(figure)
    return output_path


def save_weighted_graph_plot(
    run_dir: Path,
    model,
    pos: Mapping[int, object],
    filename: str = "weighted_graph.png",
    *,
    boundary_geojson: Mapping[str, object] | None = None,
) -> Path | None:
    """Save an unlabeled weighted graph over the RS boundary.

    Edge color and width encode raw non-self adjacency magnitude. The colorbar
    deliberately retains numeric ticks while omitting a descriptive label so
    the figure can be used directly in a presentation.
    """
    adjacency = _model_adjacency_matrix(model, normalized=False)
    if adjacency is None:
        return None

    positions = _validated_positions(pos)
    adjacency = np.asarray(adjacency, dtype=float)
    expected_shape = (len(positions), len(positions))
    if adjacency.shape != expected_shape:
        raise ValueError(
            f"Adjacency matrix shape must match the station positions: "
            f"expected {expected_shape}, got {adjacency.shape}."
        )
    if not np.all(np.isfinite(adjacency)):
        raise ValueError("Adjacency matrix contains NaN or infinite values.")

    # The learned topologies are undirected. Taking the greater magnitude keeps
    # this visualization meaningful for compatible models that expose a matrix
    # with small numerical asymmetries.
    magnitudes = np.maximum(np.abs(adjacency), np.abs(adjacency.T))
    np.fill_diagonal(magnitudes, 0.0)
    source_nodes, destination_nodes = np.where(np.triu(magnitudes, k=1) > 0.0)
    edges = [
        (int(source), int(destination))
        for source, destination in zip(source_nodes, destination_nodes)
    ]
    edge_magnitudes = np.asarray(
        [magnitudes[source, destination] for source, destination in edges],
        dtype=float,
    )

    polygons = _rs_boundary_polygons(boundary_geojson)
    figure, axis, graph = _prepare_rs_graph_axis(positions, polygons)
    graph.add_edges_from(edges)

    if edge_magnitudes.size:
        magnitude_max = float(edge_magnitudes.max())
        color_min = 0.0
        color_max = magnitude_max if magnitude_max > color_min else color_min + 1.0
        normalization = Normalize(vmin=color_min, vmax=color_max)
        scaled = normalization(edge_magnitudes)
        edge_widths = 3.0 + 5.0 * np.asarray(scaled, dtype=float)
        colormap = sns.color_palette("viridis", as_cmap=True)
        nx.draw_networkx_edges(
            graph,
            positions,
            edge_color=edge_magnitudes,
            edge_cmap=colormap,
            edge_vmin=color_min,
            edge_vmax=color_max,
            width=edge_widths,
            alpha=0.9,
            ax=axis,
        )
        colorbar = figure.colorbar(
            ScalarMappable(norm=normalization, cmap=colormap),
            ax=axis,
            fraction=0.038,
            pad=0.015,
        )
        colorbar.outline.set_linewidth(0.6)
    else:
        axis.text(
            0.5,
            0.03,
            "No non-self adjacency weights",
            ha="center",
            va="bottom",
            transform=axis.transAxes,
            fontsize=10,
        )

    _draw_station_nodes(axis, graph, positions)
    output_path = Path(run_dir) / filename
    save_figure(figure, output_path, dpi=190)
    plt.close(figure)
    return output_path


def _inverse_target_scale(values: np.ndarray, target_scaler) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if target_scaler is None:
        return values
    return target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def _prepare_prediction_arrays(y_true, y_pred, target_scaler=None) -> tuple[np.ndarray, np.ndarray]:
    actual = _inverse_target_scale(_as_numpy(y_true), target_scaler)
    predicted = _inverse_target_scale(_as_numpy(y_pred), target_scaler)
    if actual.shape != predicted.shape:
        raise ValueError(f"Prediction shape mismatch: actual={actual.shape}, predicted={predicted.shape}")
    if actual.ndim != 3:
        raise ValueError("Expected prediction arrays with shape [sample, lead_day, station].")
    return actual, predicted


def _target_time_matrix(test_y, n_samples: int, n_leads: int) -> np.ndarray:
    if "target_time" not in test_y.coords:
        raise ValueError("test_y must carry a 'target_time' coordinate.")
    target_times = np.asarray(test_y.coords["target_time"].values)
    if target_times.shape != (n_samples, n_leads):
        raise ValueError(
            "target_time coordinate must have shape "
            f"({n_samples}, {n_leads}); got {target_times.shape}."
        )
    parsed = pd.to_datetime(target_times.reshape(-1), errors="raise")
    return parsed.to_numpy(dtype="datetime64[ns]").reshape(target_times.shape)


def _station_names(test_y, n_stations: int) -> list[str]:
    if "station" in test_y.coords:
        names = [str(value) for value in test_y.station.values]
        if len(names) == n_stations:
            return names
    return [f"station_{idx}" for idx in range(n_stations)]


def _safe_regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    metric_standard=None,
    metric_threshold=0.0,
) -> dict[str, float]:
    metrics = numpy_regression_metrics(
        actual,
        predicted,
        metric_standard=metric_standard,
        metric_threshold=metric_threshold,
    )
    return {
        "MSE": metrics["mse"],
        "RMSE": metrics["rmse"],
        "MAE": metrics["mae"],
        "R2": metrics["r2"],
        "bias": metrics["bias"],
        "n_metric_targets": metrics["count"],
    }


def _save_rain_confusion_matrix(
    run_dir: Path,
    classification_metrics: Mapping[str, object],
    *,
    threshold: float,
) -> Path:
    """Save an actual-by-predicted rain/no-rain confusion-matrix heatmap."""
    matrix = np.asarray(classification_metrics["confusion_matrix"], dtype=int)
    if matrix.shape != (2, 2):
        raise ValueError("Rain confusion matrix must have shape [2, 2].")

    labels = ["Não chove", "Chove"]
    figure, axis = plt.subplots(figsize=(6.8, 5.7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=sns.color_palette("Blues", as_cmap=True),
        cbar=False,
        square=True,
        linewidths=1.0,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
        annot_kws={"fontsize": 15, "fontweight": "semibold"},
    )
    axis.set_xlabel("Previsto")
    axis.set_ylabel("Real")
    axis.set_title(
        "Matriz de confusão: chove / não chove\n"
        f"Chove se precipitação > {threshold:g} mm"
    )
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    output_path = Path(run_dir) / "confusion_matrix.png"
    save_figure(figure, output_path, dpi=190)
    plt.close(figure)
    return output_path


def _cross_node_standard_deviation(values: np.ndarray) -> np.ndarray:
    """Return population standard deviation across stations for each sample/lead."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise ValueError("Expected values with shape [sample, lead_day, station].")
    finite = np.isfinite(values)
    counts = finite.sum(axis=2)
    safe_values = np.where(finite, values, 0.0)
    means = safe_values.sum(axis=2) / np.maximum(counts, 1)
    squared_deviation = np.where(finite, (values - means[:, :, None]) ** 2, 0.0)
    variance = squared_deviation.sum(axis=2) / np.maximum(counts, 1)
    return np.where(counts >= 2, np.sqrt(variance), np.nan)


def _cross_node_mean(values: np.ndarray) -> np.ndarray:
    """Return the finite station mean for each sample and forecast lead."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise ValueError("Expected values with shape [sample, lead_day, station].")
    finite = np.isfinite(values)
    counts = finite.sum(axis=2)
    safe_values = np.where(finite, values, 0.0)
    return np.where(counts > 0, safe_values.sum(axis=2) / np.maximum(counts, 1), np.nan)


def _node_standard_deviation_prediction_dataframe(
    actual: np.ndarray,
    predicted_node_std,
    target_times: np.ndarray,
    target_scaler=None,
) -> pd.DataFrame:
    """Build physical-scale learned node-spread predictions by sample/lead."""
    predicted_std = _as_numpy(predicted_node_std)
    expected_shape = actual.shape[:2]
    if predicted_std.shape != expected_shape:
        raise ValueError(
            "Predicted node standard deviation must have shape "
            f"[sample, lead_day]={expected_shape}; got {predicted_std.shape}."
        )
    if not np.all(np.isfinite(predicted_std)):
        raise ValueError("Predicted node standard deviation contains NaN or infinite values.")
    if np.any(predicted_std < 0.0):
        raise ValueError("Predicted node standard deviation must be non-negative.")

    predicted_std_mm = target_standard_deviation_to_physical_scale(
        predicted_std,
        target_scaler,
    )
    actual_std_mm = _cross_node_standard_deviation(actual)
    residual_mm = actual_std_mm - predicted_std_mm
    n_samples, n_leads = expected_shape
    timestamps = pd.to_datetime(target_times.reshape(-1), errors="raise").strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "sample": np.repeat(np.arange(n_samples, dtype=int), n_leads),
            "lead_day": np.tile(np.arange(1, n_leads + 1, dtype=int), n_samples),
            "target_time": timestamps,
            "actual_node_std_mm": actual_std_mm.reshape(-1),
            "predicted_node_std_mm": predicted_std_mm.reshape(-1),
            "residual_node_std_mm": residual_mm.reshape(-1),
            "absolute_error_node_std_mm": np.abs(residual_mm).reshape(-1),
            "squared_error_node_std_mm2": np.square(residual_mm).reshape(-1),
        }
    )


def _oversmoothing_dataframe(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_times: np.ndarray,
) -> pd.DataFrame:
    """Build one spatial-dispersion row per sample and forecast lead day."""
    actual_std = _cross_node_standard_deviation(actual)
    predicted_std = _cross_node_standard_deviation(predicted)
    actual_mean = _cross_node_mean(actual)
    predicted_mean = _cross_node_mean(predicted)
    with np.errstate(divide="ignore", invalid="ignore"):
        std_ratio = np.where(actual_std > 1e-8, predicted_std / actual_std, np.nan)

    n_samples, n_leads, _ = actual.shape
    timestamps = pd.to_datetime(target_times.reshape(-1), errors="raise").strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "sample": np.repeat(np.arange(n_samples, dtype=int), n_leads),
            "lead_day": np.tile(np.arange(1, n_leads + 1, dtype=int), n_samples),
            "target_time": timestamps,
            "actual_node_std_mm": actual_std.reshape(-1),
            "predicted_node_std_mm": predicted_std.reshape(-1),
            "actual_node_mean_mm": actual_mean.reshape(-1),
            "predicted_node_mean_mm": predicted_mean.reshape(-1),
            "node_std_ratio": std_ratio.reshape(-1),
        }
    )


def _resolve_diagnostic_adjacency(
    model,
    edge_index,
    n_stations: int,
) -> tuple[np.ndarray | None, str | None]:
    """Return a symmetric nonnegative adjacency for graph-smoothness diagnostics."""
    adjacency = _model_adjacency_matrix(model, normalized=False) if model is not None else None
    source = "final model adjacency" if adjacency is not None else None
    if adjacency is not None:
        adjacency = np.asarray(adjacency, dtype=float)
        if adjacency.shape != (n_stations, n_stations):
            adjacency = None
            source = None

    if adjacency is None and edge_index is not None:
        edges = np.asarray(_as_numpy(edge_index), dtype=int)
        if edges.ndim == 2 and edges.shape[0] == 2:
            adjacency = np.zeros((n_stations, n_stations), dtype=float)
            source_nodes, destination_nodes = edges
            valid = (
                (source_nodes >= 0)
                & (source_nodes < n_stations)
                & (destination_nodes >= 0)
                & (destination_nodes < n_stations)
                & (source_nodes != destination_nodes)
            )
            adjacency[source_nodes[valid], destination_nodes[valid]] = 1.0
            source = "base edge_index"

    if adjacency is None:
        return None, None

    adjacency = np.where(np.isfinite(adjacency), adjacency, 0.0)
    adjacency = np.maximum(adjacency, 0.0)
    np.fill_diagonal(adjacency, 0.0)
    adjacency = 0.5 * (adjacency + adjacency.T)
    if not np.any(adjacency > 0.0):
        return None, None
    return adjacency, source


def _graph_dirichlet_energy(values: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Compute mean squared variation over weighted undirected graph edges."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or adjacency.shape != (values.shape[2], values.shape[2]):
        raise ValueError("Graph energy expects [sample, lead_day, station] values and a matching adjacency.")

    valid = np.isfinite(values).all(axis=2)
    safe_values = np.where(np.isfinite(values), values, 0.0)
    degrees = adjacency.sum(axis=1)
    total_edge_weight = float(adjacency.sum() / 2.0)
    if total_edge_weight <= 0.0:
        return np.full(values.shape[:2], np.nan, dtype=float)

    diagonal_term = np.einsum("bhn,n,bhn->bh", safe_values, degrees, safe_values)
    adjacency_term = np.einsum("bhn,nm,bhm->bh", safe_values, adjacency, safe_values)
    energy = np.maximum(diagonal_term - adjacency_term, 0.0) / total_edge_weight
    return np.where(valid, energy, np.nan)


def _lead_grid(n_leads: int, *, width: float, height: float):
    n_columns = min(3, max(1, n_leads))
    n_rows = int(np.ceil(n_leads / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(width * n_columns, height * n_rows),
        squeeze=False,
    )
    for axis in axes.ravel()[n_leads:]:
        axis.set_visible(False)
    return fig, axes.ravel()


def _save_node_standard_deviation_by_time(
    output_dir: Path,
    diagnostics: pd.DataFrame,
    n_leads: int,
) -> Path:
    """Plot item 1: spatial precipitation spread across stations over time."""
    fig, axes = _lead_grid(n_leads, width=6.0, height=4.0)
    palette = prediction_palette()
    for lead_day, axis in zip(range(1, n_leads + 1), axes):
        lead_values = diagnostics[diagnostics["lead_day"] == lead_day].sort_values("target_time")
        dates = pd.to_datetime(lead_values["target_time"])
        window = min(30, max(1, len(lead_values)))
        for column, color, series_name in (
            ("actual_node_std_mm", palette["actual"], "Actual"),
            ("predicted_node_std_mm", palette["predicted"], "Prediction"),
        ):
            values = lead_values[column].to_numpy(dtype=float)
            rolling = pd.Series(values, index=dates).rolling(window=window, min_periods=1).mean()
            sns.lineplot(
                x=rolling.index,
                y=rolling.values,
                ax=axis,
                color=color,
                linewidth=2.0,
                label=f"{series_name} (30-day mean)",
                estimator=None,
                errorbar=None,
            )
        axis.set_title(f"Lead Day {lead_day}")
        axis.set_ylabel("SD across stations (mm)")
        axis.xaxis.set_major_locator(mdates.AutoDateLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
        axis.legend(fontsize=11)
        style_time_axis(axis)
    fig.suptitle("1. Cross-node precipitation standard deviation over time", y=1.01)
    fig.autofmt_xdate(rotation=25, ha="right")
    path = output_dir / "01_node_standard_deviation_by_time.png"
    save_figure(fig, path, dpi=190)
    plt.close(fig)
    return path


def _save_node_mean_by_time(
    output_dir: Path,
    diagnostics: pd.DataFrame,
    n_leads: int,
) -> Path:
    """Plot item 2: daily precipitation mean across stations over time."""
    fig, axes = _lead_grid(n_leads, width=6.0, height=4.0)
    palette = prediction_palette()
    for lead_day, axis in zip(range(1, n_leads + 1), axes):
        lead_values = diagnostics[diagnostics["lead_day"] == lead_day].sort_values("target_time")
        dates = pd.to_datetime(lead_values["target_time"])
        for column, color, series_name in (
            ("actual_node_mean_mm", palette["actual"], "Actual"),
            ("predicted_node_mean_mm", palette["predicted"], "Prediction"),
        ):
            sns.lineplot(
                x=dates,
                y=lead_values[column].to_numpy(dtype=float),
                ax=axis,
                color=color,
                linestyle="-",
                linewidth=2.0,
                label=f"{series_name} daily station mean",
                estimator=None,
                errorbar=None,
            )
        axis.set_title(f"Lead Day {lead_day}")
        axis.set_ylabel("Mean across stations (mm)")
        axis.xaxis.set_major_locator(mdates.AutoDateLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
        axis.legend(fontsize=11)
        style_time_axis(axis)
    fig.suptitle("2. Daily precipitation mean across stations over time", y=1.01)
    fig.autofmt_xdate(rotation=25, ha="right")
    path = output_dir / "02_node_mean_by_time.png"
    save_figure(fig, path, dpi=190)
    plt.close(fig)
    return path


def _save_spread_ratio_by_lead_day(output_dir: Path, diagnostics: pd.DataFrame) -> Path:
    """Plot item 3: retention of cross-node dispersion by forecast lead day."""
    ratio_values = diagnostics[["lead_day", "node_std_ratio"]].dropna()
    ratio_values = ratio_values[ratio_values["node_std_ratio"] >= 0.0].copy()
    ratio_values["ratio_for_plot"] = ratio_values["node_std_ratio"].clip(lower=1e-4)

    fig, axis = plt.subplots(figsize=(9.2, 5.7))
    if ratio_values.empty:
        axis.text(0.5, 0.5, "No finite node-dispersion ratios were available.", ha="center", va="center")
    else:
        lead_days = sorted(ratio_values["lead_day"].unique())
        sns.boxplot(
            data=ratio_values,
            x="lead_day",
            y="ratio_for_plot",
            color=prediction_palette()["scatter"],
            showfliers=False,
            ax=axis,
        )
        sns.lineplot(
            x=(-0.5, len(lead_days) - 0.5),
            y=(1.0, 1.0),
            ax=axis,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.3,
            label="Equal dispersion",
            estimator=None,
            errorbar=None,
        )
        axis.set_yscale("log")
        axis.set_xticks(range(len(lead_days)), [f"Lead Day {int(value)}" for value in lead_days])
        axis.legend(fontsize=11)
    axis.set_title("3. Cross-node dispersion retention by lead day")
    axis.set_xlabel("Forecast lead day")
    axis.set_ylabel("Predicted SD / actual SD (log scale)")
    style_axis(axis)
    path = output_dir / "03_node_dispersion_ratio_by_lead_day.png"
    save_figure(fig, path, dpi=190)
    plt.close(fig)
    return path


def _save_predicted_vs_actual_spread(
    output_dir: Path,
    diagnostics: pd.DataFrame,
    n_leads: int,
) -> Path:
    """Plot item 4: predicted-versus-actual spatial standard deviation scatter."""
    finite = diagnostics[["actual_node_std_mm", "predicted_node_std_mm"]].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite).all(axis=1)]
    upper = float(np.max(finite)) * 1.05 if finite.size else 1.0
    upper = max(upper, 1.0)

    fig, axes = _lead_grid(n_leads, width=5.0, height=4.5)
    scatter_color = prediction_palette()["scatter"]
    for lead_day, axis in zip(range(1, n_leads + 1), axes):
        lead_values = diagnostics[diagnostics["lead_day"] == lead_day]
        x_values = lead_values["actual_node_std_mm"].to_numpy(dtype=float)
        y_values = lead_values["predicted_node_std_mm"].to_numpy(dtype=float)
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        sns.scatterplot(
            x=x_values[mask],
            y=y_values[mask],
            ax=axis,
            color=scatter_color,
            alpha=0.34,
            s=15,
            edgecolor="none",
        )
        sns.lineplot(
            x=(0.0, upper),
            y=(0.0, upper),
            ax=axis,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.15,
            estimator=None,
            errorbar=None,
        )
        axis.set_xlim(0.0, upper)
        axis.set_ylim(0.0, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"Lead Day {lead_day}")
        axis.set_xlabel("Actual SD across stations (mm)")
        axis.set_ylabel("Predicted SD across stations (mm)")
        style_axis(axis)
    fig.suptitle("4. Predicted versus actual cross-node dispersion", y=1.01)
    path = output_dir / "04_predicted_vs_actual_node_standard_deviation.png"
    save_figure(fig, path, dpi=190)
    plt.close(fig)
    return path


def _save_dirichlet_energy_by_time(
    output_dir: Path,
    diagnostics: pd.DataFrame,
    n_leads: int,
    graph_source: str,
) -> Path:
    """Plot item 6: graph variation of actual and predicted fields over time."""
    fig, axes = _lead_grid(n_leads, width=6.0, height=4.0)
    palette = prediction_palette()
    for lead_day, axis in zip(range(1, n_leads + 1), axes):
        lead_values = diagnostics[diagnostics["lead_day"] == lead_day].sort_values("target_time")
        dates = pd.to_datetime(lead_values["target_time"])
        window = min(30, max(1, len(lead_values)))
        for column, color, label in (
            ("actual_graph_dirichlet_energy_mm2", palette["actual"], "Actual (30-day mean)"),
            ("predicted_graph_dirichlet_energy_mm2", palette["predicted"], "Prediction (30-day mean)"),
        ):
            values = lead_values[column].to_numpy(dtype=float)
            finite = np.isfinite(values)
            sns.lineplot(
                x=dates[finite],
                y=values[finite],
                ax=axis,
                color=color,
                alpha=0.18,
                linewidth=0.75,
                estimator=None,
                errorbar=None,
            )
            rolling = pd.Series(values, index=dates).rolling(window=window, min_periods=1).mean()
            sns.lineplot(
                x=rolling.index,
                y=rolling.values,
                ax=axis,
                color=color,
                linewidth=2.0,
                label=label,
                estimator=None,
                errorbar=None,
            )
        axis.set_title(f"Lead Day {lead_day}")
        axis.set_ylabel("Mean edge variation (mm²)")
        axis.xaxis.set_major_locator(mdates.AutoDateLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
        axis.legend(fontsize=11)
        style_time_axis(axis)
    fig.suptitle(f"6. Graph Dirichlet energy over time ({graph_source})", y=1.01)
    fig.autofmt_xdate(rotation=25, ha="right")
    path = output_dir / "06_graph_dirichlet_energy_by_time.png"
    save_figure(fig, path, dpi=190)
    plt.close(fig)
    return path


def _oversmoothing_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead_day, values in diagnostics.groupby("lead_day", sort=True):
        actual_std = values["actual_node_std_mm"].to_numpy(dtype=float)
        predicted_std = values["predicted_node_std_mm"].to_numpy(dtype=float)
        actual_energy = values["actual_graph_dirichlet_energy_mm2"].to_numpy(dtype=float)
        predicted_energy = values["predicted_graph_dirichlet_energy_mm2"].to_numpy(dtype=float)
        mean_actual_std = float(np.nanmean(actual_std))
        mean_predicted_std = float(np.nanmean(predicted_std))
        mean_actual_energy = float(np.nanmean(actual_energy))
        mean_predicted_energy = float(np.nanmean(predicted_energy))
        rows.append(
            {
                "lead_day": int(lead_day),
                "n_samples": int(len(values)),
                "actual_node_std_mean_mm": mean_actual_std,
                "predicted_node_std_mean_mm": mean_predicted_std,
                "node_std_retention": mean_predicted_std / mean_actual_std if mean_actual_std > 1e-8 else np.nan,
                "node_std_ratio_median": float(np.nanmedian(values["node_std_ratio"].to_numpy(dtype=float))),
                "fraction_predicted_std_lower": float(np.nanmean(predicted_std < actual_std)),
                "actual_graph_dirichlet_energy_mean_mm2": mean_actual_energy,
                "predicted_graph_dirichlet_energy_mean_mm2": mean_predicted_energy,
                "graph_energy_retention": mean_predicted_energy / mean_actual_energy if mean_actual_energy > 1e-8 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_oversmoothing_diagnostics(
    run_dir: Path,
    actual: np.ndarray,
    predicted: np.ndarray,
    target_times: np.ndarray,
    *,
    model=None,
    edge_index=None,
) -> dict[str, Path]:
    """Save spatial-collapse diagnostics for all station nodes in a test prediction set."""
    run_dir = Path(run_dir)
    output_dir = run_dir / OVERSMOOTHING_DIRECTORY
    output_dir.mkdir(exist_ok=True)
    logs_dir = logs_directory(run_dir, create=True) / OVERSMOOTHING_DIRECTORY
    logs_dir.mkdir(exist_ok=True)
    diagnostics = _oversmoothing_dataframe(actual, predicted, target_times)
    adjacency, graph_source = _resolve_diagnostic_adjacency(model, edge_index, actual.shape[2])
    if adjacency is None:
        diagnostics["actual_graph_dirichlet_energy_mm2"] = np.nan
        diagnostics["predicted_graph_dirichlet_energy_mm2"] = np.nan
        graph_source = "no graph topology available"
    else:
        diagnostics["actual_graph_dirichlet_energy_mm2"] = _graph_dirichlet_energy(actual, adjacency).reshape(-1)
        diagnostics["predicted_graph_dirichlet_energy_mm2"] = _graph_dirichlet_energy(predicted, adjacency).reshape(-1)

    diagnostics.to_csv(logs_dir / "node_dispersion_by_time.csv", index=False)
    _oversmoothing_summary(diagnostics).to_csv(
        logs_dir / "oversmoothing_summary_by_lead_day.csv",
        index=False,
    )
    outputs = {
        "node_standard_deviation": _save_node_standard_deviation_by_time(output_dir, diagnostics, actual.shape[1]),
        "node_mean": _save_node_mean_by_time(output_dir, diagnostics, actual.shape[1]),
        "node_dispersion_ratio": _save_spread_ratio_by_lead_day(output_dir, diagnostics),
        "predicted_vs_actual_node_standard_deviation": _save_predicted_vs_actual_spread(
            output_dir,
            diagnostics,
            actual.shape[1],
        ),
    }
    if adjacency is not None:
        outputs["graph_dirichlet_energy"] = _save_dirichlet_energy_by_time(
            output_dir,
            diagnostics,
            actual.shape[1],
            graph_source,
        )
    return outputs


def _axis_limits(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.ravel(actual), np.ravel(predicted)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
        padding = max(1.0, abs(high) * 0.05)
    else:
        padding = (high - low) * 0.05
    return low - padding, high + padding


def _station_mean(values: np.ndarray) -> np.ndarray:
    return np.nanmean(np.asarray(values, dtype=float), axis=2)


def _normalize_station_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_name = "".join(char for char in normalized if not unicodedata.combining(char))
    ascii_name = re.sub(r"[^A-Za-z0-9]+", " ", ascii_name).strip().upper()
    return re.sub(r"\s+", " ", ascii_name)


def _resolve_plot_station(
    requested_station: str | None,
    station_names: list[str],
) -> tuple[int, str]:
    if not station_names:
        raise ValueError("No stations are available for prediction plots.")
    if requested_station is None or str(requested_station).strip() == "":
        return 0, station_names[0]

    requested_key = _normalize_station_name(requested_station)
    normalized_names = [_normalize_station_name(name) for name in station_names]
    exact_matches = [idx for idx, key in enumerate(normalized_names) if key == requested_key]
    if len(exact_matches) == 1:
        idx = exact_matches[0]
        return idx, station_names[idx]

    partial_matches = [
        idx
        for idx, key in enumerate(normalized_names)
        if requested_key in key or key in requested_key
    ]
    if len(partial_matches) == 1:
        idx = partial_matches[0]
        return idx, station_names[idx]

    available = ", ".join(station_names)
    if partial_matches:
        matches = ", ".join(station_names[idx] for idx in partial_matches)
        raise ValueError(
            f"PLOT_STATION_NAME={requested_station!r} is ambiguous. Matches: {matches}."
        )
    raise ValueError(
        f"PLOT_STATION_NAME={requested_station!r} was not found among selected stations. "
        f"Available stations: {available}"
    )


def _station_series(values: np.ndarray, station_idx: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[:, :, station_idx]


def _prediction_dataframe(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_times: np.ndarray,
    station_names: list[str],
) -> pd.DataFrame:
    rows = []
    for sample_idx in range(actual.shape[0]):
        for lead_idx in range(actual.shape[1]):
            target_time = pd.Timestamp(target_times[sample_idx, lead_idx]).date().isoformat()
            for station_idx, station_name in enumerate(station_names):
                actual_value = float(actual[sample_idx, lead_idx, station_idx])
                predicted_value = float(predicted[sample_idx, lead_idx, station_idx])
                residual = actual_value - predicted_value
                rows.append(
                    {
                        "sample": int(sample_idx),
                        "lead_day": int(lead_idx + 1),
                        "target_time": target_time,
                        "station": station_name,
                        "actual": actual_value,
                        "predicted": predicted_value,
                        "residual": residual,
                        "actual_mm": actual_value,
                        "predicted_mm": predicted_value,
                        "residual_mm": residual,
                        "absolute_error_mm": abs(residual),
                        "squared_error_mm2": residual**2,
                    }
                )
    return pd.DataFrame(rows)


def save_dataset_contract(run_dir: Path, raw_X, raw_y, windowed, scaled_windowed, config) -> None:
    metric_standard = normalize_metric_standard(getattr(config, "metric_standard", None))
    summary = {
        "raw_X_dims": dict(raw_X.sizes),
        "raw_y_dims": dict(raw_y.sizes),
        "raw_time_start": pd.Timestamp(raw_X.time.values[0]).date().isoformat(),
        "raw_time_end": pd.Timestamp(raw_X.time.values[-1]).date().isoformat(),
        "features": [str(value) for value in raw_X.feature.values],
        "stations": [str(value) for value in raw_X.station.values],
        "windowed_dims": {
            "train_X": dict(windowed.train_X.sizes),
            "val_X": dict(windowed.val_X.sizes),
            "test_X": dict(windowed.test_X.sizes),
            "train_y": dict(windowed.train_y.sizes),
            "val_y": dict(windowed.val_y.sizes),
            "test_y": dict(windowed.test_y.sizes),
        },
        "scaled": {
            "features": config.normalize_features,
            "target": config.normalize_target,
        },
        "plots": {
            "plot_station_name": getattr(config, "plot_station_name", None),
        },
        "metrics": {
            "metric_standard": metric_standard,
            "metric_threshold_mm": getattr(config, "metric_threshold", None),
            "selection": (
                "target_mm > metric_threshold_mm"
                if metric_standard == METRIC_STANDARD_MODIFIED
                else "all finite target/prediction pairs"
            ),
        },
        "torch_boundary": "Only after dated xarray split/window/scaling, via to_torch_window_splits().",
    }
    with open(logs_directory(run_dir, create=True) / "dataset_contract.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _serialize_scaler_state(scaler) -> dict | None:
    """Serialize the fitted numeric state needed for transform/inverse_transform."""
    if scaler is None:
        return None

    scaler_name = type(scaler).__name__
    if scaler_name == "StandardScaler":
        return {
            "kind": "standard",
            "mean": np.asarray(scaler.mean_, dtype=float).reshape(-1).tolist(),
            "scale": np.asarray(scaler.scale_, dtype=float).reshape(-1).tolist(),
        }
    if scaler_name == "MinMaxScaler":
        return {
            "kind": "minmax",
            "min": np.asarray(scaler.min_, dtype=float).reshape(-1).tolist(),
            "scale": np.asarray(scaler.scale_, dtype=float).reshape(-1).tolist(),
        }
    raise TypeError(
        f"Unsupported fitted scaler {scaler_name!r}; expected StandardScaler or MinMaxScaler."
    )


def _model_build_state(model) -> dict:
    """Capture effective non-tensor constructor options that a state_dict omits."""
    class_name = type(model).__name__
    if class_name == "GLSTM_v2":
        first_cell = model.cell_0
        return {
            "model_class": class_name,
            "adjacency_scope": "shared" if model.share_adjacency else "per_layer",
            "kwargs": {
                "N": int(model.N),
                "in_channels": int(first_cell.W_i.in_features),
                "hidden_size": int(model.hidden_size),
                "out_channels": int(model.window),
                "lstm_layers": int(model.lstm_layers),
                "aggr": str(model.aggr),
                "learn_adj": bool(first_cell.learn_adj),
                "learn_self_att": bool(getattr(first_cell, "learn_self_att", False)),
                "lock_topology": bool(first_cell.lock_topology),
                "dropout": float(model.dropout),
                "cell_clip": None if first_cell.cell_clip is None else float(first_cell.cell_clip),
                "learn_std": bool(getattr(model, "learn_std", False)),
            },
        }
    if class_name == "NodewiseLSTM":
        return {
            "model_class": class_name,
            "kwargs": {
                "N": int(model.N),
                "in_channels": int(model.in_channels),
                "hidden_size": int(model.hidden_size),
                "out_channels": int(model.window),
                "lstm_layers": int(model.lstm_layers),
                "dropout": float(model.dropout),
            },
        }
    if class_name == "GraphTemporalTransformer_v1":
        day_layer = model.day_encoder.layers[0]
        return {
            "model_class": class_name,
            "kwargs": {
                "N": int(model.N),
                "in_channels": int(model.in_channels),
                "hidden_size": int(model.hidden_size),
                "out_channels": int(model.window),
                "learn_adj": bool(model.adj.learn_adj),
                "lock_topology": bool(model.adj.lock_topology),
                "target_dim": int(model.target_dim),
                "nhead": int(day_layer.self_attn.num_heads),
                "num_day_layers": int(len(model.day_encoder.layers)),
                "num_window_layers": int(1 + len(model.extra_graph_mixers)),
                "num_decoder_layers": int(len(model.decoder.layers)),
                "dim_feedforward": int(day_layer.linear1.out_features),
                "dropout": float(model.input_dropout.p),
                "max_window": int(model.max_window),
                "max_days": int(model.max_days),
                "squeeze_output": bool(model.squeeze_output),
                "four_dim_mode": str(model.four_dim_mode),
                "precip_col": int(model.precip_col),
                "output_activation": model.output_activation,
                "use_node_embeddings": bool(model.use_node_embeddings),
                "use_precip_residual": bool(model.use_precip_residual),
            },
        }
    raise TypeError(f"Unsupported model class for inference persistence: {class_name!r}.")


def save_inference_state(
    run_dir: Path,
    model,
    edge_index,
    stations: Mapping[str, object],
    raw_X,
    scaling_state,
) -> Path:
    """Persist topology, node coordinates, feature order, and fitted scalers."""
    station_names = [str(value) for value in raw_X.station.values]
    feature_names = [str(value) for value in raw_X.feature.values]
    station_coordinates = {}
    for station_name in station_names:
        if station_name not in stations:
            raise ValueError(f"Station {station_name!r} is missing from the coordinate mapping.")
        coordinates = list(stations[station_name])
        if len(coordinates) < 2:
            raise ValueError(f"Station {station_name!r} must provide latitude and longitude.")
        station_coordinates[station_name] = [float(coordinates[0]), float(coordinates[1])]

    edge_array = np.asarray(_as_numpy(edge_index), dtype=np.int64)
    if edge_array.ndim != 2 or edge_array.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E]. Got {edge_array.shape}.")

    payload = {
        "schema_version": 2,
        "checkpoint": "model_state_dict.pt",
        "config": "config.json",
        "dataset_contract": "dataset_contract.json",
        "model_build": _model_build_state(model),
        "stations": station_names,
        "features": feature_names,
        "station_coordinates": station_coordinates,
        "edge_index": edge_array.tolist(),
        "feature_scaler": _serialize_scaler_state(scaling_state.feature_scaler),
        "target_scaler": _serialize_scaler_state(scaling_state.target_scaler),
    }
    output_path = logs_directory(run_dir, create=True) / "inference_state.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def _save_prediction_overview(
    run_dir: Path,
    actual: np.ndarray,
    predicted: np.ndarray,
    target_times: np.ndarray,
    plot_station_idx: int,
    plot_station_name: str,
) -> None:
    plot_dir = run_dir / "prediction_overview"
    plot_dir.mkdir(exist_ok=True)

    actual_station = _station_series(actual, plot_station_idx)
    predicted_station = _station_series(predicted, plot_station_idx)
    final_lead = actual.shape[1] - 1
    era5_label, glstm_label = lead_day_legend_labels(final_lead + 1)
    dates = pd.to_datetime(target_times[:, final_lead])
    low, high = _axis_limits(actual, predicted)
    palette = prediction_palette()

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    sns.lineplot(
        x=dates,
        y=actual_station[:, final_lead],
        ax=axes[0],
        label=era5_label,
        color=palette["actual"],
        linewidth=2.2,
        estimator=None,
    )
    sns.lineplot(
        x=dates,
        y=predicted_station[:, final_lead],
        ax=axes[0],
        label=glstm_label,
        color=palette["predicted"],
        linewidth=2.2,
        estimator=None,
    )
    axes[0].set_title(f"{plot_station_name} - Lead Day {final_lead + 1}")
    axes[0].set_ylabel("Precipitation (mm)")
    axes[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    axes[0].legend()
    style_time_axis(axes[0])

    sns.scatterplot(
        x=actual.reshape(-1),
        y=predicted.reshape(-1),
        ax=axes[1],
        color=palette["scatter"],
        alpha=0.42,
        s=22,
        edgecolor="white",
        linewidth=0.2,
    )
    sns.lineplot(
        x=(low, high),
        y=(low, high),
        ax=axes[1],
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.2,
        estimator=None,
        errorbar=None,
    )
    axes[1].set_xlim(low, high)
    axes[1].set_ylim(low, high)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("Actual vs Predicted - All Lead Days and Stations")
    axes[1].set_xlabel("Actual precipitation (mm)")
    axes[1].set_ylabel("Predicted precipitation (mm)")
    style_axis(axes[1])

    fig.autofmt_xdate(rotation=30, ha="right")
    save_figure(fig, plot_dir / "02_predictions_vs_actual.png", dpi=190)
    plt.close(fig)


def _save_prediction_timeseries_splits(
    run_dir: Path,
    actual: np.ndarray,
    predicted: np.ndarray,
    target_times: np.ndarray,
    plot_station_idx: int,
    plot_station_name: str,
    n_splits: int = 4,
) -> None:
    plot_dir = run_dir / "prediction_timeseries_splits"
    plot_dir.mkdir(exist_ok=True)

    actual_station = _station_series(actual, plot_station_idx)
    predicted_station = _station_series(predicted, plot_station_idx)
    indices = np.arange(actual.shape[0])
    n_splits = max(1, int(n_splits))
    palette = prediction_palette()

    for lead_idx in range(actual.shape[1]):
        lead_day = lead_idx + 1
        era5_label, glstm_label = lead_day_legend_labels(lead_day)
        lead_dir = plot_dir / f"lead_day_{lead_day:02d}"
        lead_dir.mkdir(exist_ok=True)

        for split_index, split_indices in enumerate(np.array_split(indices, n_splits), start=1):
            fig, ax = plt.subplots(figsize=(14, 5.6))
            if split_indices.size:
                dates = pd.to_datetime(target_times[split_indices, lead_idx])
                sns.lineplot(
                    x=dates,
                    y=actual_station[split_indices, lead_idx],
                    ax=ax,
                    label=era5_label,
                    color=palette["actual"],
                    linewidth=2.0,
                    estimator=None,
                )
                sns.lineplot(
                    x=dates,
                    y=predicted_station[split_indices, lead_idx],
                    ax=ax,
                    label=glstm_label,
                    color=palette["predicted"],
                    linewidth=2.0,
                    estimator=None,
                )
                ax.legend()
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
                fig.autofmt_xdate(rotation=30, ha="right")
            else:
                ax.text(0.5, 0.5, "No samples in this split", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(
                f"{plot_station_name} - Lead Day {lead_day}: Predictions vs Actual - "
                f"Split {split_index} of {n_splits}"
            )
            ax.set_ylabel("Precipitation (mm)")
            style_time_axis(ax)
            save_figure(
                fig,
                lead_dir / f"02_predictions_timeseries_split_{split_index:02d}_of_{n_splits:02d}.png",
                dpi=190,
            )
            plt.close(fig)


def _save_absolute_error_boxplots(
    output_dir: Path,
    lead_df: pd.DataFrame,
    n_leads: int,
) -> Path:
    """Save absolute-error distributions for every lead day and all leads.

    Each observation in ``lead_df`` represents one station at one target time,
    so no station averaging is applied before plotting.
    """
    required_columns = {"lead_day", "actual_mm", "predicted_mm"}
    missing_columns = sorted(required_columns.difference(lead_df.columns))
    if missing_columns:
        raise ValueError(
            "Lead-day prediction data is missing required columns: "
            + ", ".join(missing_columns)
        )

    n_leads = int(n_leads)
    if n_leads < 1:
        raise ValueError("n_leads must be positive.")

    values = lead_df[["lead_day", "actual_mm", "predicted_mm"]].copy()
    values["lead_day"] = pd.to_numeric(values["lead_day"], errors="coerce")
    values["actual_mm"] = pd.to_numeric(values["actual_mm"], errors="coerce")
    values["predicted_mm"] = pd.to_numeric(values["predicted_mm"], errors="coerce")
    values["absolute_error_mm"] = np.abs(values["actual_mm"] - values["predicted_mm"])
    values = values.loc[
        values["lead_day"].between(1, n_leads)
        & np.isfinite(values["absolute_error_mm"])
    ].copy()
    values["lead_day"] = values["lead_day"].astype(int)

    lead_labels = [f"D+{lead_day}" for lead_day in range(1, n_leads + 1)]
    aggregate_label = "All lead days"
    category_order = [*lead_labels, aggregate_label]
    values["lead_label"] = values["lead_day"].map(
        {lead_day: f"D+{lead_day}" for lead_day in range(1, n_leads + 1)}
    )
    aggregate_values = values.assign(lead_label=aggregate_label)
    plot_values = pd.concat([values, aggregate_values], ignore_index=True)
    counts = plot_values.groupby("lead_label").size().reindex(category_order, fill_value=0)

    figure, axis = plt.subplots(figsize=(12.8, 6.4))
    if plot_values.empty:
        axis.text(
            0.5,
            0.5,
            "No finite absolute prediction errors were available.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        sns.boxplot(
            data=plot_values,
            x="lead_label",
            y="absolute_error_mm",
            order=category_order,
            ax=axis,
            color=prediction_palette()["scatter"],
            width=0.62,
            showfliers=False,
            medianprops={"color": REFERENCE_COLOR, "linewidth": 2.0},
        )
        axis.set_xticks(
            range(len(category_order)),
            [f"{label}\n(n={int(counts[label])})" for label in category_order],
        )

    axis.set_title("Absolute prediction error across all stations")
    axis.set_xlabel("")
    axis.set_ylabel("Absolute error |actual − predicted| (mm)")
    style_axis(axis, grid_axis="y")
    output_path = output_dir / "15_absolute_prediction_error_boxplots.png"
    save_figure(figure, output_path, dpi=190)
    plt.close(figure)
    return output_path


def _save_forecast_lead_day_diagnostics(
    run_dir: Path,
    lead_df: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    target_times: np.ndarray,
    plot_station_idx: int,
    plot_station_name: str,
    metric_standard=None,
    metric_threshold: float = 0.0,
) -> pd.DataFrame:
    diag_dir = run_dir / "forecast_horizon_diagnostics"
    diag_dir.mkdir(exist_ok=True)
    diag_logs_dir = logs_directory(run_dir, create=True) / "forecast_horizon_diagnostics"
    diag_logs_dir.mkdir(exist_ok=True)
    lead_df.to_csv(diag_logs_dir / "test_prediction_by_lead_day.csv", index=False)
    palette = prediction_palette()
    metric_scope = (
        f"targets > {metric_threshold:g} mm"
        if metric_standard == METRIC_STANDARD_MODIFIED
        else "all targets"
    )

    metric_rows = []
    for lead_day, lead_values in lead_df.groupby("lead_day", sort=True):
        metrics = _safe_regression_metrics(
            lead_values["actual_mm"].to_numpy(dtype=float),
            lead_values["predicted_mm"].to_numpy(dtype=float),
            metric_standard=metric_standard,
            metric_threshold=metric_threshold,
        )
        metric_rows.append(
            {
                "lead_day": int(lead_day),
                "n_test": int(len(lead_values)),
                "metric_standard": metric_standard,
                "metric_threshold_mm": (
                    metric_threshold
                    if metric_standard == METRIC_STANDARD_MODIFIED
                    else np.nan
                ),
                **metrics,
            }
        )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(diag_logs_dir / "test_prediction_metrics_by_lead_day.csv", index=False)

    _save_absolute_error_boxplots(
        diag_dir,
        lead_df,
        n_leads=actual.shape[1],
    )

    if not metrics_df.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        sns.lineplot(
            data=metrics_df,
            x="lead_day",
            y="RMSE",
            ax=ax,
            marker="o",
            linewidth=2.4,
            label="RMSE",
            color=palette["rmse"],
        )
        sns.lineplot(
            data=metrics_df,
            x="lead_day",
            y="MAE",
            ax=ax,
            marker="s",
            linewidth=2.4,
            label="MAE",
            color=palette["mae"],
        )
        ax.set_xlabel("Lead day after input window")
        ax.set_ylabel("Error (mm)")
        ax.set_title(f"Test Error by Forecast Lead Day ({metric_scope})")
        ax.set_xticks(metrics_df["lead_day"])
        ax.legend()
        style_axis(ax)
        save_figure(fig, diag_dir / "12_prediction_error_by_lead_day.png", dpi=190)
        plt.close(fig)

    n_leads = int(actual.shape[1])
    if n_leads:
        n_cols = min(3, n_leads)
        n_rows = int(np.ceil(n_leads / n_cols))
        low, high = _axis_limits(actual, predicted)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.1 * n_cols, 4.5 * n_rows), squeeze=False)
        for ax in axes.ravel()[n_leads:]:
            ax.set_visible(False)
        for ax, lead_day in zip(axes.ravel(), range(1, n_leads + 1)):
            values = lead_df[lead_df["lead_day"] == lead_day]
            sns.scatterplot(
                data=values,
                x="actual_mm",
                y="predicted_mm",
                ax=ax,
                color=palette["scatter"],
                s=24,
                alpha=0.46,
                edgecolor="white",
                linewidth=0.2,
            )
            sns.lineplot(
                x=(low, high),
                y=(low, high),
                ax=ax,
                color=REFERENCE_COLOR,
                linestyle="--",
                linewidth=1.1,
                estimator=None,
                errorbar=None,
            )
            metrics = metrics_df.loc[metrics_df["lead_day"] == lead_day].iloc[0]
            ax.set_title(
                f"Lead Day {lead_day}: RMSE={metrics['RMSE']:.3f}, "
                f"MAE={metrics['MAE']:.3f}, n={int(metrics['n_metric_targets'])}"
            )
            ax.set_xlabel("Actual precipitation (mm)")
            ax.set_ylabel("Predicted precipitation (mm)")
            style_axis(ax)
        fig.suptitle("True vs Predicted by Lead Day", y=1.0)
        save_figure(fig, diag_dir / "13_true_vs_predicted_by_lead_day.png", dpi=190)
        plt.close(fig)

        by_lead_dir = diag_dir / "true_vs_predicted_by_lead_day"
        by_lead_dir.mkdir(exist_ok=True)
        for lead_day in range(1, n_leads + 1):
            values = lead_df[lead_df["lead_day"] == lead_day]
            metrics = metrics_df.loc[metrics_df["lead_day"] == lead_day].iloc[0]
            fig, ax = plt.subplots(figsize=(7.2, 6.2))
            sns.scatterplot(
                data=values,
                x="actual_mm",
                y="predicted_mm",
                ax=ax,
                color=palette["scatter"],
                s=28,
                alpha=0.52,
                edgecolor="white",
                linewidth=0.22,
            )
            sns.lineplot(
                x=(low, high),
                y=(low, high),
                ax=ax,
                color=REFERENCE_COLOR,
                linestyle="--",
                linewidth=1.1,
                estimator=None,
                errorbar=None,
            )
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"Actual precipitation at Lead Day {lead_day} (mm)")
            ax.set_ylabel(f"Prediction at Lead Day {lead_day} (mm)")
            ax.set_title(
                f"True vs Predicted at Lead Day {lead_day}: RMSE={metrics['RMSE']:.3f}, "
                f"MAE={metrics['MAE']:.3f}, n={int(metrics['n_metric_targets'])}"
            )
            style_axis(ax)
            save_figure(fig, by_lead_dir / f"true_vs_predicted_lead_day_{lead_day:02d}.png", dpi=190)
            plt.close(fig)

        actual_station = _station_series(actual, plot_station_idx)
        predicted_station = _station_series(predicted, plot_station_idx)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 3.8 * n_rows), squeeze=False)
        for ax in axes.ravel()[n_leads:]:
            ax.set_visible(False)
        for ax, lead_idx in zip(axes.ravel(), range(n_leads)):
            lead_day = lead_idx + 1
            era5_label, glstm_label = lead_day_legend_labels(lead_day)
            dates = pd.to_datetime(target_times[:, lead_idx])
            sns.lineplot(
                x=dates,
                y=actual_station[:, lead_idx],
                ax=ax,
                label=era5_label,
                color=palette["actual"],
                linewidth=1.9,
                estimator=None,
            )
            sns.lineplot(
                x=dates,
                y=predicted_station[:, lead_idx],
                ax=ax,
                label=glstm_label,
                color=palette["predicted"],
                linewidth=1.9,
                alpha=0.92,
                estimator=None,
            )
            ax.set_title(f"{plot_station_name} - Lead Day {lead_day}")
            ax.set_ylabel("Precipitation (mm)")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
            ax.legend()
            style_time_axis(ax)
        fig.suptitle(
            f"Test Time Series: Prediction Compared With Each Lead Day - {plot_station_name}",
            y=1.0,
        )
        fig.autofmt_xdate(rotation=30, ha="right")
        save_figure(fig, diag_dir / "14_prediction_vs_actual_timeseries_by_lead_day.png", dpi=190)
        plt.close(fig)

    return metrics_df


def save_prediction_outputs(
    run_dir: Path,
    y_true,
    y_pred,
    test_y,
    target_scaler=None,
    n_time_splits: int = 4,
    plot_station_name: str | None = None,
    graph_model=None,
    edge_index=None,
    metric_standard=None,
    metric_threshold: float = 0.0,
    confusion_matrix_threshold: float = 0.0,
    predicted_node_std=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save prediction CSVs, metrics, and plots in the experiment output directory.

    The rain/no-rain classification uses physical precipitation values and
    ``confusion_matrix_threshold`` in millimetres, independently of the
    optional regression metric policy.
    """
    metric_standard = normalize_metric_standard(metric_standard)
    metric_threshold = validate_metric_threshold(metric_threshold)
    confusion_matrix_threshold = validate_metric_threshold(confusion_matrix_threshold)
    run_dir = Path(run_dir)
    actual, predicted = _prepare_prediction_arrays(y_true, y_pred, target_scaler=target_scaler)
    target_times = _target_time_matrix(test_y, n_samples=actual.shape[0], n_leads=actual.shape[1])
    station_names = _station_names(test_y, n_stations=actual.shape[2])
    plot_station_idx, resolved_plot_station = _resolve_plot_station(plot_station_name, station_names)
    node_std_predictions_df = (
        _node_standard_deviation_prediction_dataframe(
            actual,
            predicted_node_std,
            target_times,
            target_scaler=target_scaler,
        )
        if predicted_node_std is not None
        else None
    )

    predictions_df = _prediction_dataframe(actual, predicted, target_times, station_names)
    predictions_df["metric_eligible"] = (
        predictions_df["actual_mm"] > metric_threshold
        if metric_standard == METRIC_STANDARD_MODIFIED
        else True
    )
    logs_dir = logs_directory(run_dir, create=True)
    predictions_df.to_csv(logs_dir / "test_predictions_by_lead_day.csv", index=False)
    if node_std_predictions_df is not None:
        node_std_predictions_df.to_csv(
            logs_dir / "test_node_standard_deviation_predictions_by_lead_day.csv",
            index=False,
        )

    physical_metrics = _safe_regression_metrics(
        actual,
        predicted,
        metric_standard=metric_standard,
        metric_threshold=metric_threshold,
    )
    classification_metrics = numpy_rain_classification_metrics(
        actual,
        predicted,
        threshold=confusion_matrix_threshold,
    )
    confusion_matrix_payload = {
        "confusion_matrix_threshold_mm": confusion_matrix_threshold,
        "positive_class": f"Chove (precipitação > {confusion_matrix_threshold:g} mm)",
        "negative_class": f"Não chove (precipitação ≤ {confusion_matrix_threshold:g} mm)",
        "matrix_layout": "rows=actual [não chove, chove]; columns=predicted [não chove, chove]",
        **classification_metrics,
    }
    with open(logs_dir / "test_metrics_physical_scale.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric_standard": metric_standard,
                "metric_threshold_mm": (
                    metric_threshold
                    if metric_standard == METRIC_STANDARD_MODIFIED
                    else None
                ),
                "metric_units": {"MSE": "mm^2", "RMSE": "mm", "MAE": "mm", "bias": "mm"},
                **physical_metrics,
                **confusion_matrix_payload,
            },
            f,
            indent=2,
        )
    with open(logs_dir / "test_confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(confusion_matrix_payload, f, indent=2)
    _save_rain_confusion_matrix(
        run_dir,
        classification_metrics,
        threshold=confusion_matrix_threshold,
    )

    _save_prediction_overview(
        run_dir,
        actual,
        predicted,
        target_times,
        plot_station_idx=plot_station_idx,
        plot_station_name=resolved_plot_station,
    )
    _save_prediction_timeseries_splits(
        run_dir,
        actual,
        predicted,
        target_times,
        plot_station_idx=plot_station_idx,
        plot_station_name=resolved_plot_station,
        n_splits=n_time_splits,
    )
    metrics_df = _save_forecast_lead_day_diagnostics(
        run_dir,
        predictions_df,
        actual,
        predicted,
        target_times,
        plot_station_idx=plot_station_idx,
        plot_station_name=resolved_plot_station,
        metric_standard=metric_standard,
        metric_threshold=metric_threshold,
    )
    save_oversmoothing_diagnostics(
        run_dir,
        actual,
        predicted,
        target_times,
        model=graph_model,
        edge_index=edge_index,
    )
    return predictions_df, metrics_df
