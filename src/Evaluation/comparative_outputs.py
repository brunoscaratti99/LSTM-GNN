"""Sweep-level comparison figures and a LaTeX report for ``run_experiment``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from math import ceil
from pathlib import Path
import re
import unicodedata

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from Evaluation.metrics import (
    METRIC_STANDARD_MODIFIED,
    normalize_metric_standard,
    numpy_regression_metrics,
    validate_metric_threshold,
)
from Evaluation.plot_style import (
    REFERENCE_COLOR,
    apply_seaborn_theme,
    lead_day_legend_labels,
    save_figure,
    style_axis,
    style_time_axis,
)
from output_layout import logs_directory, resolve_run_artifact


REPORT_FILENAME = "report_compare.tex"
ANALYSIS_DIRECTORY = "comparative_analysis"
HISTORY_FIGURE = "03_training_history_comparison.png"
PARAMETER_METRICS_FIGURE = "04_comparative_parameter_metrics.png"
STATION_METRICS_CSV = "station_metrics_common_dates.csv"
RUN_INDICATOR_FILE = "hist.pt"


apply_seaborn_theme()


def _json_display(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _normalize_station_name(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_name = "".join(character for character in normalized if not unicodedata.combining(character))
    ascii_name = re.sub(r"[^A-Za-z0-9]+", " ", ascii_name).strip().upper()
    return re.sub(r"\s+", " ", ascii_name)


def _completed_records(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    runs = manifest.get("runs", [])
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        return []
    return [
        record
        for record in runs
        if isinstance(record, Mapping) and record.get("status") == "completed"
    ]


def _run_label_for_warning(record: Mapping[str, object]) -> str:
    return str(record.get("run_name") or record.get("run_dir") or _record_label(record))


def _run_directory(sweep_dir: Path, record: Mapping[str, object]) -> Path | None:
    relative_path = record.get("run_dir") or record.get("run_name")
    if not isinstance(relative_path, (str, Path)):
        return None
    candidate = Path(relative_path)
    return candidate if candidate.is_absolute() else sweep_dir / candidate


def _incomplete_run_reasons(sweep_dir: Path, record: Mapping[str, object]) -> list[str]:
    reasons = []
    run_dir = _run_directory(sweep_dir, record)
    if run_dir is None:
        reasons.append("run path is unavailable")
        return reasons
    if not run_dir.is_dir():
        reasons.append(f"run directory not found: {run_dir.name}")
        return reasons

    if not resolve_run_artifact(run_dir, RUN_INDICATOR_FILE).is_file():
        reasons.append(f"missing {RUN_INDICATOR_FILE}")
    return reasons


def filter_complete_run_records(
    sweep_dir: Path | str,
    manifest: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], list[dict[str, object]]]:
    """Return report-ready runs, using ``hist.pt`` as the completion indicator."""
    sweep_dir = Path(sweep_dir)
    runs = manifest.get("runs", [])
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        return [], [{"run_name": "<manifest>", "reasons": ["manifest has no valid runs list"]}]

    complete = []
    skipped = []
    for record in runs:
        if not isinstance(record, Mapping):
            skipped.append({"run_name": "<invalid record>", "reasons": ["run entry is not an object"]})
            continue
        reasons = _incomplete_run_reasons(sweep_dir, record)
        if reasons:
            skipped.append({"run_name": _run_label_for_warning(record), "reasons": reasons})
        else:
            complete.append(record)
    return complete, skipped


def _record_label(record: Mapping[str, object]) -> str:
    metadata = record.get("comparative")
    if isinstance(metadata, Mapping):
        varied = metadata.get("varied_parameters")
        if isinstance(varied, Mapping) and varied:
            details = ", ".join(f"{name}={_json_display(value)}" for name, value in varied.items())
            return details[:88]
        parameter = metadata.get("comparative_parameter")
        if parameter is not None:
            return f"{parameter}={_json_display(metadata.get('comparative_value'))}"
    return str(record.get("run_name", f"run_{record.get('run_index', '?')}"))


def _record_parameter_value(record: Mapping[str, object], parameter: str) -> object:
    metadata = record.get("comparative")
    if isinstance(metadata, Mapping):
        varied = metadata.get("varied_parameters")
        if isinstance(varied, Mapping) and parameter in varied:
            return varied[parameter]
        if metadata.get("comparative_parameter") == parameter:
            return metadata.get("comparative_value")
    parameters = record.get("parameters")
    if isinstance(parameters, Mapping) and parameter in parameters:
        return parameters[parameter]
    return record.get("run_index")


def _record_metric_policy(
    record: Mapping[str, object],
    run_dir: Path | None = None,
) -> tuple[str | None, float]:
    """Read metric policy from a manifest record, falling back to config.json."""
    parameters = record.get("parameters")
    values = dict(parameters) if isinstance(parameters, Mapping) else {}
    if ("metric_standard" not in values or "metric_threshold" not in values) and run_dir is not None:
        config_path = resolve_run_artifact(run_dir, "config.json")
        if config_path.is_file():
            try:
                config_values = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                config_values = {}
            if isinstance(config_values, Mapping):
                values = {**config_values, **values}

    metric_standard = normalize_metric_standard(values.get("metric_standard"))
    metric_threshold = validate_metric_threshold(values.get("metric_threshold", 0.0))
    return metric_standard, metric_threshold


def _row_identity(row: Mapping[str, object]) -> tuple[object, object, object]:
    return (row.get("run_index"), row.get("run_name"), row.get("label"))


def _history_from_run(run_dir: Path, warnings: list[str]) -> dict | None:
    history_path = resolve_run_artifact(run_dir, "hist.pt")
    if not history_path.exists():
        warnings.append(f"History not found: {history_path.name} ({run_dir.name}).")
        return None
    try:
        history = torch.load(history_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - report generation must tolerate old artifacts.
        warnings.append(f"Could not read {history_path.name} in {run_dir.name}: {exc}")
        return None
    if not isinstance(history, dict):
        warnings.append(f"Invalid history format in {run_dir.name}.")
        return None
    return history


def _history_values(history: Mapping[str, object], key: str, *, rmse: bool = False) -> np.ndarray:
    values = history.get(key, [])
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=float)
    if rmse:
        array = np.sqrt(np.maximum(array, 0.0))
    return array


def _save_training_history_comparison(
    completed: Sequence[Mapping[str, object]],
    sweep_dir: Path,
    output_dir: Path,
    warnings: list[str],
) -> Path:
    metric_specs = (
        ("Loss", "loss", False),
        ("RMSE", "mse", True),
        ("MAE", "mae", False),
        ("R2", "r2", False),
    )
    colors = sns.color_palette("colorblind", n_colors=10)
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.2), squeeze=False)
    has_history = False

    for run_index, record in enumerate(completed):
        run_dir = _run_directory(sweep_dir, record)
        if run_dir is None:
            warnings.append(f"Run path is unavailable for {_record_label(record)}.")
            continue
        history = _history_from_run(run_dir, warnings)
        if history is None:
            continue
        has_history = True
        color = colors[run_index % len(colors)]
        for axis, (title, metric_key, use_rmse) in zip(axes.ravel(), metric_specs):
            for split_name, linestyle in (("train", "-"), ("validation", "--")):
                key = f"{'val' if split_name == 'validation' else 'train'}_{metric_key}"
                values = _history_values(history, key, rmse=use_rmse)
                if values.size:
                    sns.lineplot(
                        x=np.arange(1, values.size + 1),
                        y=values,
                        ax=axis,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.85,
                        estimator=None,
                        errorbar=None,
                    )

    for axis, (title, _metric_key, _use_rmse) in zip(axes.ravel(), metric_specs):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        style_axis(axis)
        if not has_history:
            axis.text(0.5, 0.5, "No training history available", ha="center", va="center", transform=axis.transAxes)

    color_handles = [
        Line2D([0], [0], color=colors[index % len(colors)], linewidth=2.4, label=_record_label(record))
        for index, record in enumerate(completed)
    ]
    split_handles = [
        Line2D([0], [0], color="#4b5563", linewidth=2.0, linestyle="-", label="Train"),
        Line2D([0], [0], color="#4b5563", linewidth=2.0, linestyle="--", label="Validation"),
    ]
    fig.legend(
        handles=[*color_handles, *split_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=min(3, max(1, len(color_handles) + len(split_handles))),
        fontsize=9.5,
    )
    fig.suptitle("Training-history comparison", y=1.055, fontsize=15, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_path = output_dir / HISTORY_FIGURE
    save_figure(fig, output_path, dpi=190, tight=False)
    plt.close(fig)
    return output_path


def _requested_station(records: Sequence[Mapping[str, object]]) -> str | None:
    for record in records:
        parameters = record.get("parameters")
        if isinstance(parameters, Mapping):
            value = parameters.get("plot_station_name")
            if value is not None and str(value).strip():
                return str(value)
    return None


def _resolve_station(frame: pd.DataFrame, requested: str | None) -> str | None:
    stations = [str(value) for value in frame["station"].dropna().unique()]
    if not stations:
        return None
    if requested is None or not requested.strip():
        return stations[0]
    requested_key = _normalize_station_name(requested)
    normalized = {_normalize_station_name(station): station for station in stations}
    exact = normalized.get(requested_key)
    if exact is not None:
        return exact
    partial = [
        station
        for station in stations
        if requested_key in _normalize_station_name(station)
        or _normalize_station_name(station) in requested_key
    ]
    return partial[0] if len(partial) == 1 else None


def _prediction_columns(frame: pd.DataFrame) -> tuple[str, str] | None:
    actual_column = "actual_mm" if "actual_mm" in frame.columns else "actual"
    predicted_column = "predicted_mm" if "predicted_mm" in frame.columns else "predicted"
    required = {"station", "target_time", "lead_day", actual_column, predicted_column}
    return (actual_column, predicted_column) if required.issubset(frame.columns) else None


def _load_station_predictions(
    completed: Sequence[Mapping[str, object]],
    sweep_dir: Path,
    requested_station: str | None,
    warnings: list[str],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for record in completed:
        run_dir = _run_directory(sweep_dir, record)
        if run_dir is None:
            continue
        csv_path = resolve_run_artifact(run_dir, "test_predictions_by_lead_day.csv")
        if not csv_path.exists():
            warnings.append(f"Prediction CSV not found for {_record_label(record)}.")
            continue
        try:
            raw = pd.read_csv(csv_path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not read prediction CSV in {run_dir.name}: {exc}")
            continue
        columns = _prediction_columns(raw)
        if columns is None:
            warnings.append(f"Prediction CSV in {run_dir.name} lacks required columns.")
            continue
        station = _resolve_station(raw, requested_station)
        if station is None:
            warnings.append(
                f"Selected station {requested_station!r} was not resolved in {run_dir.name}; run omitted from station plots."
            )
            continue
        actual_column, predicted_column = columns
        data = raw.loc[raw["station"].astype(str) == station, ["target_time", "lead_day", actual_column, predicted_column]].copy()
        data = data.rename(columns={actual_column: "actual_mm", predicted_column: "predicted_mm"})
        data["target_time"] = pd.to_datetime(data["target_time"], errors="coerce")
        data["lead_day"] = pd.to_numeric(data["lead_day"], errors="coerce")
        data["actual_mm"] = pd.to_numeric(data["actual_mm"], errors="coerce")
        data["predicted_mm"] = pd.to_numeric(data["predicted_mm"], errors="coerce")
        data = data.dropna(subset=["target_time", "lead_day", "actual_mm", "predicted_mm"])
        if data.empty:
            warnings.append(f"No finite station predictions found in {run_dir.name}.")
            continue
        data["lead_day"] = data["lead_day"].astype(int)
        try:
            metric_standard, metric_threshold = _record_metric_policy(record, run_dir)
        except ValueError as exc:
            warnings.append(f"Invalid metric policy in {run_dir.name}: {exc}")
            continue
        entries.append(
            {
                "record": record,
                "label": _record_label(record),
                "station": station,
                "data": data,
                "metric_standard": metric_standard,
                "metric_threshold": metric_threshold,
            }
        )
    return entries


def _by_date(data: pd.DataFrame, lead_day: int) -> pd.DataFrame:
    selected = data.loc[data["lead_day"] == lead_day, ["target_time", "actual_mm", "predicted_mm"]]
    return (
        selected.groupby("target_time", as_index=True, sort=True)
        .mean(numeric_only=True)
        .sort_index()
    )


def _aligned_by_lead(entries: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    if not entries:
        return []
    lead_sets = [set(entry["data"]["lead_day"].unique()) for entry in entries]
    common_leads = set.intersection(*lead_sets) if lead_sets else set()
    aligned: list[dict[str, object]] = []
    for lead_day in sorted(int(value) for value in common_leads):
        by_run = [(entry, _by_date(entry["data"], lead_day)) for entry in entries]
        date_sets = [set(frame.index) for _entry, frame in by_run]
        common_dates = sorted(set.intersection(*date_sets)) if date_sets else []
        if not common_dates:
            continue
        aligned_models = []
        for entry, frame in by_run:
            values = frame.reindex(common_dates)
            aligned_models.append(
                {
                    "entry": entry,
                    "actual_mm": values["actual_mm"].to_numpy(dtype=float),
                    "predicted_mm": values["predicted_mm"].to_numpy(dtype=float),
                }
            )
        aligned.append(
            {
                "lead_day": lead_day,
                "dates": pd.DatetimeIndex(common_dates),
                "models": aligned_models,
            }
        )
    return aligned


def _regression_metrics(
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
        "RMSE": metrics["rmse"],
        "MAE": metrics["mae"],
        "R2": metrics["r2"],
        "n_metric_targets": metrics["count"],
    }


def _shared_limits(models: Sequence[Mapping[str, object]]) -> tuple[float, float]:
    values = np.concatenate(
        [
            np.concatenate(
                [np.asarray(model["actual_mm"], dtype=float), np.asarray(model["predicted_mm"], dtype=float)]
            )
            for model in models
        ]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = float(values.min()), float(values.max())
    padding = max((high - low) * 0.05, 0.5)
    return max(0.0, low - padding), high + padding


def _save_prediction_timeseries(
    aligned_predictions: Sequence[Mapping[str, object]],
    output_dir: Path,
    colors: Mapping[str, object],
) -> list[Path]:
    paths = []
    for lead in aligned_predictions:
        models = lead["models"]
        era5_label, glstm_label = lead_day_legend_labels(lead["lead_day"])
        fig, axis = plt.subplots(figsize=(14.0, 5.4))
        actual = np.asarray(models[0]["actual_mm"], dtype=float)
        sns.lineplot(
            x=lead["dates"],
            y=actual,
            ax=axis,
            color=REFERENCE_COLOR,
            linewidth=2.5,
            label=era5_label,
            estimator=None,
            errorbar=None,
        )
        for model in models:
            entry = model["entry"]
            label = entry["label"]
            sns.lineplot(
                x=lead["dates"],
                y=model["predicted_mm"],
                ax=axis,
                color=colors[label],
                linewidth=1.75,
                alpha=0.92,
                label=(glstm_label if len(models) == 1 else f"{label} Lead Day {lead['lead_day']}"),
                estimator=None,
                errorbar=None,
            )
        station = models[0]["entry"]["station"]
        axis.set_title(f"Prediction vs actual - {station} - Lead Day {lead['lead_day']}")
        axis.set_ylabel("Precipitation (mm)")
        axis.xaxis.set_major_locator(mdates.AutoDateLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
        axis.legend(loc="best", fontsize=11)
        style_time_axis(axis)
        fig.autofmt_xdate(rotation=25, ha="right")
        fig.tight_layout()
        output_path = output_dir / f"01_test_timeseries_comparison_lead_day_{lead['lead_day']:02d}.png"
        save_figure(fig, output_path, dpi=190, tight=False)
        plt.close(fig)
        paths.append(output_path)
    return paths


def _save_scatter_comparisons(
    aligned_predictions: Sequence[Mapping[str, object]],
    output_dir: Path,
    colors: Mapping[str, object],
) -> tuple[list[Path], list[dict[str, object]]]:
    paths = []
    metric_rows: list[dict[str, object]] = []
    for lead in aligned_predictions:
        models = lead["models"]
        n_columns = min(3, len(models))
        n_rows = ceil(len(models) / n_columns)
        fig, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(5.2 * n_columns, 4.8 * n_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        limits = _shared_limits(models)
        for axis, model in zip(axes.flat, models):
            entry = model["entry"]
            label = entry["label"]
            metric_standard = entry["metric_standard"]
            metric_threshold = entry["metric_threshold"]
            metrics = _regression_metrics(
                model["actual_mm"],
                model["predicted_mm"],
                metric_standard=metric_standard,
                metric_threshold=metric_threshold,
            )
            metric_rows.append(
                {
                    "run_index": entry["record"].get("run_index"),
                    "run_name": entry["record"].get("run_name"),
                    "label": label,
                    "station": entry["station"],
                    "lead_day": lead["lead_day"],
                    "n_common_dates": int(len(lead["dates"])),
                    "metric_standard": metric_standard,
                    "metric_threshold_mm": (
                        metric_threshold
                        if metric_standard == METRIC_STANDARD_MODIFIED
                        else np.nan
                    ),
                    **metrics,
                }
            )
            sns.scatterplot(
                x=model["actual_mm"],
                y=model["predicted_mm"],
                ax=axis,
                s=20,
                alpha=0.52,
                color=colors[label],
                edgecolor="none",
            )
            sns.lineplot(
                x=limits,
                y=limits,
                ax=axis,
                color=REFERENCE_COLOR,
                linestyle="--",
                linewidth=1.2,
                estimator=None,
                errorbar=None,
            )
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(
                f"{label}\nRMSE={metrics['RMSE']:.3g} | MAE={metrics['MAE']:.3g} | "
                f"R2={metrics['R2']:.3g} | n={metrics['n_metric_targets']}",
                fontsize=10.5,
            )
            axis.set_xlabel("Actual precipitation (mm)")
            axis.set_ylabel("Predicted precipitation (mm)")
            style_axis(axis)
        for axis in axes.flat[len(models) :]:
            axis.set_visible(False)
        station = models[0]["entry"]["station"]
        fig.suptitle(
            f"True vs predicted - {station} - Lead Day {lead['lead_day']} ({len(lead['dates'])} common dates)",
            y=1.01,
            fontsize=14,
            fontweight="semibold",
        )
        fig.tight_layout()
        output_path = output_dir / f"02_test_scatter_comparison_lead_day_{lead['lead_day']:02d}.png"
        save_figure(fig, output_path, dpi=190, tight=False)
        plt.close(fig)
        paths.append(output_path)
    return paths, metric_rows


def _numeric_parameter_value(value: object) -> float | None:
    if isinstance(value, (list, tuple, dict, set)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _save_parameter_metrics_plot(
    metric_rows: Sequence[Mapping[str, object]],
    completed: Sequence[Mapping[str, object]],
    pivot_parameter: str,
    output_dir: Path,
) -> Path:
    output_path = output_dir / PARAMETER_METRICS_FIGURE
    overview_rows = _overview_metric_rows(metric_rows)
    record_lookup = {_row_identity({"run_index": record.get("run_index"), "run_name": record.get("run_name"), "label": _record_label(record)}): record for record in completed}
    plot_rows = []
    for row in overview_rows:
        record = record_lookup.get(_row_identity(row))
        parameter_value = _record_parameter_value(record, pivot_parameter) if record is not None else row.get("label")
        numeric_value = _numeric_parameter_value(parameter_value)
        plot_rows.append(
            {
                **row,
                "parameter_value": parameter_value,
                "numeric_parameter_value": numeric_value,
                "parameter_label": _json_display(parameter_value),
            }
        )

    numeric_axis = bool(plot_rows) and all(row["numeric_parameter_value"] is not None for row in plot_rows)
    if numeric_axis:
        plot_rows = sorted(plot_rows, key=lambda row: (float(row["numeric_parameter_value"]), str(row["label"])))
        x_values = np.asarray([float(row["numeric_parameter_value"]) for row in plot_rows], dtype=float)
        x_labels = [row["parameter_label"] for row in plot_rows]
    else:
        plot_rows = sorted(plot_rows, key=lambda row: (row.get("run_index") is None, row.get("run_index"), str(row["label"])))
        x_values = np.arange(len(plot_rows), dtype=float)
        x_labels = [row["parameter_label"] for row in plot_rows]

    metric_specs = (("RMSE", "min"), ("MAE", "min"), ("R2", "max"))
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 9.0), sharex=True, squeeze=False)
    color_map = {"RMSE": "#2563eb", "MAE": "#059669", "R2": "#dc2626"}

    for axis, (metric, direction) in zip(axes.ravel(), metric_specs):
        if not plot_rows:
            axis.text(0.5, 0.5, "No overview metrics available", ha="center", va="center", transform=axis.transAxes)
            axis.set_ylabel(metric)
            style_axis(axis)
            continue

        y_values = np.asarray([row.get(metric, np.nan) for row in plot_rows], dtype=float)
        finite = np.isfinite(y_values)
        sns.lineplot(
            x=x_values[finite],
            y=y_values[finite],
            ax=axis,
            color=color_map[metric],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            estimator=None,
            errorbar=None,
        )
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            local_index = int(np.argmin(y_values[finite])) if direction == "min" else int(np.argmax(y_values[finite]))
            best_index = finite_indices[local_index]
            sns.scatterplot(
                x=[x_values[best_index]],
                y=[y_values[best_index]],
                ax=axis,
                marker="*",
                s=220,
                color="#f59e0b",
                edgecolor="#78350f",
                linewidth=0.9,
                zorder=5,
                label="Best",
            )
            axis.annotate(
                "best",
                xy=(x_values[best_index], y_values[best_index]),
                xytext=(6, 7),
                textcoords="offset points",
                fontsize=10,
            )
        axis.set_ylabel(metric)
        style_axis(axis)

    axes.ravel()[-1].set_xlabel(pivot_parameter)
    if numeric_axis:
        axes.ravel()[-1].set_xticks(x_values)
        axes.ravel()[-1].set_xticklabels(x_labels, rotation=0)
    else:
        axes.ravel()[-1].set_xticks(x_values)
        axes.ravel()[-1].set_xticklabels(x_labels, rotation=25, ha="right")
    fig.suptitle(f"{pivot_parameter} vs selected-station metrics", fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, output_path, dpi=190, tight=False)
    plt.close(fig)
    return output_path


def _latex_figure(filename: str, caption: str) -> list[str]:
    return [
        r"\begin{figure}[H]",
        r"\centering",
        rf"\IfFileExists{{\detokenize{{{filename}}}}}{{%",
        rf"  \includegraphics[width=\linewidth]{{\detokenize{{{filename}}}}}%",
        r"}{\fbox{\parbox{0.92\linewidth}{\centering Figure unavailable.}}}",
        rf"\caption{{{_latex_escape(caption)}}}",
        r"\end{figure}",
    ]


def _metric_value(value: object) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{value:.4g}" if np.isfinite(value) else "--"


def _metric_scope(row: Mapping[str, object]) -> str:
    if row.get("metric_standard") == METRIC_STANDARD_MODIFIED:
        return f"target > {_metric_value(row.get('metric_threshold_mm'))} mm"
    return "all targets"


def _best_metric_values(metric_rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    directions = {"RMSE": "min", "MAE": "min", "R2": "max"}
    best = {}
    for metric, direction in directions.items():
        values = []
        for row in metric_rows:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if values:
            best[metric] = min(values) if direction == "min" else max(values)
    return best


def _latex_metric_cell(value: object, best_value: float | None) -> str:
    formatted = _metric_value(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return formatted
    if best_value is not None and np.isfinite(numeric) and np.isclose(numeric, best_value, rtol=1e-10, atol=1e-12):
        return rf"\textbf{{{formatted}}}"
    return formatted


def _overview_metric_rows(metric_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Pool RMSE/MAE by common dates and compute the mean R2 across lead days."""
    grouped: dict[tuple[object, object, object], list[Mapping[str, object]]] = {}
    for row in metric_rows:
        key = (row.get("run_index"), row.get("run_name"), row.get("label"))
        grouped.setdefault(key, []).append(row)

    overview = []
    for (run_index, run_name, label), rows in grouped.items():
        total_dates = 0.0
        total_metric_targets = 0.0
        weighted_squared_error = 0.0
        weighted_absolute_error = 0.0
        weighted_r2 = 0.0
        r2_dates = 0.0
        for row in rows:
            try:
                n_dates = float(row["n_common_dates"])
                n_metric_targets = float(row.get("n_metric_targets", n_dates))
                rmse = float(row["RMSE"])
                mae = float(row["MAE"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                n_metric_targets <= 0
                or not np.isfinite(n_metric_targets)
                or not np.isfinite(rmse)
                or not np.isfinite(mae)
            ):
                continue
            total_dates += n_dates
            total_metric_targets += n_metric_targets
            weighted_squared_error += n_metric_targets * rmse**2
            weighted_absolute_error += n_metric_targets * mae
            try:
                r2 = float(row["R2"])
            except (KeyError, TypeError, ValueError):
                r2 = np.nan
            if np.isfinite(r2):
                weighted_r2 += n_metric_targets * r2
                r2_dates += n_metric_targets
        if total_metric_targets:
            first_row = rows[0]
            overview.append(
                {
                    "run_index": run_index,
                    "run_name": run_name,
                    "label": label,
                    "n_common_dates": int(total_dates),
                    "n_metric_targets": int(total_metric_targets),
                    "metric_standard": first_row.get("metric_standard"),
                    "metric_threshold_mm": first_row.get("metric_threshold_mm", np.nan),
                    "RMSE": float(np.sqrt(weighted_squared_error / total_metric_targets)),
                    "MAE": float(weighted_absolute_error / total_metric_targets),
                    "R2": float(weighted_r2 / r2_dates) if r2_dates else np.nan,
                }
            )
    return sorted(overview, key=lambda row: (row.get("run_index") is None, row.get("run_index")))


def _latex_overview_table(metric_rows: Sequence[Mapping[str, object]]) -> list[str]:
    if not metric_rows:
        return [r"\textit{No station-level common-date metrics were available.}"]
    overview_rows = _overview_metric_rows(metric_rows)
    best = _best_metric_values(overview_rows)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l l r r r r}",
        r"\toprule",
        r"Configuration & Metric scope & Targets & RMSE (pooled) & MAE (mean) & $R^2$ (mean) \\",
        r"\midrule",
    ]
    for row in overview_rows:
        lines.append(
            f"{_latex_escape(row['label'])} & {_latex_escape(_metric_scope(row))} & "
            f"{int(row.get('n_metric_targets', row['n_common_dates']))} & "
            f"{_latex_metric_cell(row['RMSE'], best.get('RMSE'))} & "
            f"{_latex_metric_cell(row['MAE'], best.get('MAE'))} & "
            f"{_latex_metric_cell(row['R2'], best.get('R2'))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Overview across all available lead days. RMSE is pooled over eligible date-lead targets; MAE and $R^2$ are eligible-target weighted means. Bold values identify the best configuration for each metric.}",
            r"\end{table}",
        ]
    )
    return lines


def _latex_lead_day_table(lead_day: int, metric_rows: Sequence[Mapping[str, object]]) -> list[str]:
    rows = sorted(metric_rows, key=lambda row: (row.get("run_index") is None, row.get("run_index")))
    best = _best_metric_values(rows)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l l r r r r}",
        r"\toprule",
        r"Configuration & Metric scope & Targets & RMSE & MAE & $R^2$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['label'])} & {_latex_escape(_metric_scope(row))} & "
            f"{int(row.get('n_metric_targets', row['n_common_dates']))} & "
            f"{_latex_metric_cell(row['RMSE'], best.get('RMSE'))} & "
            f"{_latex_metric_cell(row['MAE'], best.get('MAE'))} & "
            f"{_latex_metric_cell(row['R2'], best.get('R2'))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Selected-station metrics at lead day D+{lead_day}. Bold values identify the best configuration for each metric.}}",
            r"\end{table}",
        ]
    )
    return lines


def _latex_metrics_tables(metric_rows: Sequence[Mapping[str, object]]) -> list[str]:
    if not metric_rows:
        return [r"\textit{No station-level common-date metrics were available.}"]
    lines = [r"\subsection{Overview across all lead days}"]
    lines.extend(_latex_overview_table(metric_rows))
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in metric_rows:
        grouped.setdefault(int(row["lead_day"]), []).append(row)
    for lead_day in sorted(grouped):
        lines.extend([r"\clearpage", rf"\subsection{{Lead day D+{lead_day}}}"])
        lines.extend(_latex_lead_day_table(lead_day, grouped[lead_day]))
    return lines


def _write_latex_report(
    output_dir: Path,
    sweep_dir: Path,
    pivot_parameter: str,
    requested_station: str | None,
    aligned_predictions: Sequence[Mapping[str, object]],
    metric_rows: Sequence[Mapping[str, object]],
    completed: Sequence[Mapping[str, object]],
    parameter_metrics_path: Path | None,
) -> Path:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.72in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\usepackage{float}",
        r"\usepackage{graphicx}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{longtable}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\begin{document}",
        r"\title{Comparative Experiment Report}",
        r"\author{}",
        r"\date{}",
        r"\maketitle",
        r"\section{Summary}",
        rf"Sweep folder: \texttt{{{_latex_escape(sweep_dir.name)}}}\\",
        rf"Comparative parameter: \texttt{{{_latex_escape(pivot_parameter)}}}\\",
        rf"Compared runs: {len(completed)}\\",
        rf"Selected station: \texttt{{{_latex_escape(requested_station or 'first available station')}}}.",
        r"\section{Metric definitions}",
        r"For the standard policy, all finite target--prediction pairs are used. For the modified policy, each run uses only the strict target-defined subset",
        r"\[I_\tau=\{i:y_i>\tau\},\qquad n_\tau=|I_\tau|,\]",
        r"where $y_i$ is observed precipitation in millimetres and $\tau$ is that run's metric threshold. Predictions do not determine membership in $I_\tau$.",
        r"\[\operatorname{RMSE}_\tau=\sqrt{\frac{1}{n_\tau}\sum_{i\in I_\tau}(\hat y_i-y_i)^2},\qquad",
        r"\operatorname{MAE}_\tau=\frac{1}{n_\tau}\sum_{i\in I_\tau}|\hat y_i-y_i|,\]",
        r"\[\bar y_\tau=\frac{1}{n_\tau}\sum_{i\in I_\tau}y_i,\qquad",
        r"R^2_\tau=1-\frac{\sum_{i\in I_\tau}(y_i-\hat y_i)^2}{\sum_{i\in I_\tau}(y_i-\bar y_\tau)^2}.\]",
        r"Modified metrics are undefined when $n_\tau=0$; modified $R^2$ is also undefined when the selected targets have zero variance.",
        r"\section{Training History}",
        *_latex_figure(HISTORY_FIGURE, "Loss, RMSE, MAE, and R2 evolution. Solid lines are train curves; dashed lines are validation curves. Regression curves follow each run's configured metric policy."),
        r"\section{Comparative Parameter Response}",
    ]
    if parameter_metrics_path is not None:
        lines.extend(
            _latex_figure(
                parameter_metrics_path.name,
                f"{pivot_parameter} against overview RMSE, MAE, and R2 across all available lead days. Stars mark the best value for each metric.",
            )
        )
    else:
        lines.append(r"\textit{No comparative-parameter metric plot was available.}")
    lines.extend(
        [
        r"\clearpage",
        r"\section{Prediction vs Actual}",
        ]
    )
    if aligned_predictions:
        for lead in aligned_predictions:
            lead_day = int(lead["lead_day"])
            lines.append(rf"\subsection{{Lead day D+{lead_day}}}")
            lines.extend(
                _latex_figure(
                    f"01_test_timeseries_comparison_lead_day_{lead_day:02d}.png",
                    f"Selected-station prediction vs actual on {len(lead['dates'])} common target dates.",
                )
            )
    else:
        lines.append(r"\textit{No common selected-station prediction dates were available.}")

    lines.extend([r"\clearpage", r"\section{Scatter Comparisons}"])
    if aligned_predictions:
        for lead in aligned_predictions:
            lead_day = int(lead["lead_day"])
            lines.append(rf"\subsection{{Lead day D+{lead_day}}}")
            lines.extend(
                _latex_figure(
                    f"02_test_scatter_comparison_lead_day_{lead_day:02d}.png",
                    "One selected-station scatter panel per configuration, with a shared scale and identity line.",
                )
            )
    else:
        lines.append(r"\textit{No selected-station scatter data were available.}")

    lines.extend([r"\clearpage", r"\section{Selected-station metrics on common dates}"])
    lines.extend(_latex_metrics_tables(metric_rows))
    lines.extend([r"\end{document}", ""])
    report_path = output_dir / REPORT_FILENAME
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def save_comparative_outputs(sweep_dir: Path | str, manifest: Mapping[str, object]) -> dict[str, object]:
    """Create comparison figures and ``report_compare.tex`` after a completed grid."""
    sweep_dir = Path(sweep_dir)
    output_dir = sweep_dir / ANALYSIS_DIRECTORY
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_logs_dir = logs_directory(output_dir, create=True)
    apply_seaborn_theme()

    warnings: list[str] = []
    completed, skipped_runs = filter_complete_run_records(sweep_dir, manifest)
    for skipped in skipped_runs:
        reasons = "; ".join(str(reason) for reason in skipped["reasons"])
        warnings.append(f"Skipped incomplete run {skipped['run_name']}: {reasons}.")
    history_path = _save_training_history_comparison(completed, sweep_dir, output_dir, warnings)
    requested_station = _requested_station(completed)
    station_entries = _load_station_predictions(completed, sweep_dir, requested_station, warnings)
    aligned_predictions = _aligned_by_lead(station_entries)
    comparison_palette = sns.color_palette("colorblind", n_colors=max(1, len(station_entries)))
    colors = {
        entry["label"]: comparison_palette[index]
        for index, entry in enumerate(station_entries)
    }
    timeseries_paths = _save_prediction_timeseries(aligned_predictions, output_dir, colors)
    scatter_paths, metric_rows = _save_scatter_comparisons(aligned_predictions, output_dir, colors)
    pd.DataFrame(metric_rows).to_csv(analysis_logs_dir / STATION_METRICS_CSV, index=False)

    pivot_parameter = str(manifest.get("comparative_parameter", "comparative_parameter"))
    parameter_metrics_path = _save_parameter_metrics_plot(metric_rows, completed, pivot_parameter, output_dir)
    report_path = _write_latex_report(
        output_dir,
        sweep_dir,
        pivot_parameter,
        requested_station,
        aligned_predictions,
        metric_rows,
        completed,
        parameter_metrics_path,
    )
    return {
        "directory": str(output_dir.relative_to(sweep_dir)),
        "report_compare_tex": str(report_path.relative_to(sweep_dir)),
        "training_history_figure": str(history_path.relative_to(sweep_dir)),
        "parameter_metrics_figure": str(parameter_metrics_path.relative_to(sweep_dir)),
        "timeseries_figures": [str(path.relative_to(sweep_dir)) for path in timeseries_paths],
        "scatter_figures": [str(path.relative_to(sweep_dir)) for path in scatter_paths],
        "station_metrics_csv": str((output_dir / STATION_METRICS_CSV).relative_to(sweep_dir)),
        "selected_station": requested_station,
        "completed_runs": len(completed),
        "skipped_incomplete_runs": skipped_runs,
        "warnings": warnings,
    }
