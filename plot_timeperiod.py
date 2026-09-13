"""Create one real-versus-predicted time-series PNG for a trained run.

The command reloads the saved model, runs a historical backtest for the
requested target-date interval, and plots the selected station.  By default
all forecast lead days are shown as panels in one PNG; ``--lead-day`` limits
the output to one horizon.

Example
-------
    python plot_timeperiod.py \
        --run-dir "Experiments/run_experiment/<run>" \
        --start-date 2024-05-01 \
        --end-date 2024-06-01 \
        --station "Porto Alegre"
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from Evaluation.plot_style import (  # noqa: E402
    apply_seaborn_theme,
    lead_day_legend_labels,
    prediction_palette,
    save_figure,
    style_time_axis,
)
from Evaluation.experiment_outputs import _normalize_station_name  # noqa: E402
from inference import run_inference  # noqa: E402
from output_layout import resolve_run_artifact  # noqa: E402


RUN_DIR: Path | None = Path(r"C:\Local Repository\LSTM-GNN\Experiments\run_experiment\07_09_2026\glstm_sweep_20260909_124805")
START_DATE: str | None = "2024-05-01"
END_DATE: str | None = "2025-01-01"
STATION: str | None = "PORTO ALEGRE JARDIM BOTANICO"
LEAD_DAY: int | None = 1
OUTPUT: Path | None = Path(r"C:\Local Repository\LSTM-GNN\Experiments\run_experiment\07_09_2026\glstm_sweep_20260909_124805")
CATALOG_PATH: Path | None = None
DEVICE = "auto"
BATCH_SIZE: int | None = None
SHOW_PROGRESS = True
PROGRESS_TIME_CHUNK_DAYS = 30


apply_seaborn_theme()


def _parse_date(value: str, argument_name: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value).normalize()
    except Exception as exc:
        raise ValueError(f"{argument_name} must be a valid date: {value!r}.") from exc
    if pd.isna(parsed):
        raise ValueError(f"{argument_name} must be a valid date: {value!r}.")
    return parsed


def _resolve_station(requested: str | None, available: Sequence[str]) -> str:
    names = [str(value) for value in available]
    if not names:
        raise ValueError("The prediction file contains no stations.")
    if requested is None or not str(requested).strip():
        return names[0]

    key = _normalize_station_name(requested)
    normalized = [_normalize_station_name(name) for name in names]
    exact = [name for name, candidate in zip(names, normalized) if candidate == key]
    if len(exact) == 1:
        return exact[0]
    partial = [
        name
        for name, candidate in zip(names, normalized)
        if key in candidate or candidate in key
    ]
    if len(partial) == 1:
        return partial[0]
    if partial:
        raise ValueError(
            f"Station {requested!r} is ambiguous. Matches: {', '.join(partial)}."
        )
    raise ValueError(
        f"Station {requested!r} was not found. Available stations: {', '.join(names)}."
    )


def _prediction_columns(frame: pd.DataFrame) -> tuple[str, str]:
    actual_column = "actual_mm" if "actual_mm" in frame.columns else "actual"
    predicted_column = "predicted_mm" if "predicted_mm" in frame.columns else "predicted"
    required = {"lead_day", "target_time", "station", actual_column, predicted_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            "Prediction CSV is missing required columns: " + ", ".join(missing)
        )
    return actual_column, predicted_column


def save_timeperiod_plot(
    predictions: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    output_path: Path | str,
    station: str | None = None,
    lead_day: int | None = None,
    show_titles: bool = True,
) -> Path:
    """Save actual and predicted curves for a date period.

    Set ``show_titles=False`` to omit both the per-axis titles and the figure
    supertitle while preserving the precipitation axis and curve legend. Date
    ticks remain visible, but the redundant lower date-axis label is omitted.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must not be after end_date.")

    frame = predictions.copy()
    actual_column, predicted_column = _prediction_columns(frame)
    frame["target_time"] = pd.to_datetime(frame["target_time"], errors="raise").dt.normalize()
    frame["station"] = frame["station"].astype(str)
    frame["lead_day"] = pd.to_numeric(frame["lead_day"], errors="raise").astype(int)
    frame[actual_column] = pd.to_numeric(frame[actual_column], errors="coerce")
    frame[predicted_column] = pd.to_numeric(frame[predicted_column], errors="coerce")
    frame = frame[frame["target_time"].between(start, end)].copy()
    if frame.empty:
        raise ValueError(
            f"No predictions were found between {start.date()} and {end.date()}."
        )

    resolved_station = _resolve_station(station, frame["station"].drop_duplicates().tolist())
    frame = frame[frame["station"] == resolved_station].copy()
    if lead_day is not None:
        if int(lead_day) < 1:
            raise ValueError("lead_day must be positive.")
        frame = frame[frame["lead_day"] == int(lead_day)].copy()
    lead_days = sorted(frame["lead_day"].unique().tolist())
    if not lead_days:
        raise ValueError(
            f"No predictions were found for station {resolved_station!r} in the requested period."
        )

    n_columns = min(3, len(lead_days))
    n_rows = int(np.ceil(len(lead_days) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(8.0 * n_columns, 4.8 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for axis in axes_flat[len(lead_days) :]:
        axis.set_visible(False)

    palette = prediction_palette()
    for axis, current_lead in zip(axes_flat, lead_days):
        era5_label, glstm_label = lead_day_legend_labels(current_lead)
        values = frame[frame["lead_day"] == current_lead].copy()
        values = values.sort_values("target_time")
        values = values.groupby("target_time", as_index=False)[
            [actual_column, predicted_column]
        ].mean()
        finite = np.isfinite(values[[actual_column, predicted_column]].to_numpy(dtype=float)).all(axis=1)
        values = values.loc[finite]

        if values.empty:
            axis.text(0.5, 0.5, "No finite values in this period", ha="center", va="center", transform=axis.transAxes)
        else:
            dates = pd.DatetimeIndex(values["target_time"])
            sns.lineplot(
                x=dates,
                y=values[actual_column],
                ax=axis,
                color=palette["actual"],
                linewidth=2.0,
                label=era5_label,
                estimator=None,
                errorbar=None,
            )
            sns.lineplot(
                x=dates,
                y=values[predicted_column],
                ax=axis,
                color=palette["predicted"],
                linewidth=2.0,
                label=glstm_label,
                estimator=None,
                errorbar=None,
            )
            axis.legend(fontsize=11)
            axis.xaxis.set_major_locator(mdates.AutoDateLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
            axis.tick_params(axis="x", rotation=30)

        if show_titles:
            axis.set_title(f"{resolved_station} - D+{current_lead}: Predictions vs Actual")
        axis.set_ylabel("Precipitation (mm)")
        style_time_axis(axis)

    if show_titles:
        figure.suptitle(
            f"Real vs predicted precipitation - {resolved_station}\n"
            f"{start:%d/%m/%Y} to {end:%d/%m/%Y}",
            y=1.01,
        )
    output_path = Path(output_path)
    save_figure(figure, output_path, dpi=190)
    plt.close(figure)
    return output_path


def _forecast_horizon(run_dir: Path) -> int:
    config_path = resolve_run_artifact(run_dir, "config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Required run artifact not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    raw_horizon = config.get("forecast_horizon")
    if raw_horizon is None:
        summary_path = resolve_run_artifact(run_dir, "run_summary.json")
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as file:
                raw_horizon = json.load(file).get("horizon")
    try:
        horizon = int(raw_horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not read forecast_horizon from {config_path}.") from exc
    if horizon < 1:
        raise ValueError(f"Invalid forecast_horizon={horizon} in {config_path}.")
    return horizon


def create_timeperiod_plot(
    run_dir: Path | str,
    *,
    start_date: str,
    end_date: str,
    output_path: Path | str | None = None,
    station: str | None = None,
    lead_day: int | None = None,
    catalog_path: Path | str | None = None,
    device: str = "auto",
    batch_size: int | None = None,
    show_progress: bool = True,
    progress_time_chunk_days: int = 30,
) -> Path:
    """Run the model for a target period and save the combined time-series PNG."""
    resolved_run = Path(run_dir).resolve()
    if not resolved_run.is_dir():
        raise FileNotFoundError(f"Run directory not found: {resolved_run}")
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must not be after end_date.")

    horizon = _forecast_horizon(resolved_run)
    requested_horizon = horizon if lead_day is None else int(lead_day)
    if requested_horizon < 1 or requested_horizon > horizon:
        raise ValueError(f"lead_day must be between 1 and {horizon}.")

    # run_inference selects forecast origins. Expanding the origin start keeps
    # the requested target period complete even for D+2..D+H panels.
    inference_start = start - pd.Timedelta(days=requested_horizon - 1)
    if output_path is None:
        output_path = resolved_run / f"timeperiod_{start:%Y%m%d}_{end:%Y%m%d}.png"

    # Keep the transient inference CSV beside the run so the command works in
    # restricted environments where the system temporary directory is not
    # writable.  TemporaryDirectory removes it after the PNG is created.
    temporary = resolved_run / f".plot_timeperiod_{uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        inference_dir = Path(temporary) / "predictions"
        generated_dir = run_inference(
            resolved_run,
            mode="backtest",
            start_date=inference_start.date().isoformat(),
            end_date=end.date().isoformat(),
            output_dir=inference_dir,
            catalog_path=catalog_path,
            device=device,
            batch_size=batch_size,
            show_progress=show_progress,
            progress_time_chunk_days=progress_time_chunk_days,
        )
        predictions = pd.read_csv(
            resolve_run_artifact(generated_dir, "inference_predictions_by_lead_day.csv")
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    return save_timeperiod_plot(
        predictions,
        start_date=start,
        end_date=end,
        output_path=output_path,
        station=station,
        lead_day=lead_day,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a saved model and save one real-versus-predicted time-period PNG."
    )
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Trained model run directory.")
    parser.add_argument("--start-date", default=START_DATE, help="First target date, inclusive.")
    parser.add_argument("--end-date", default=END_DATE, help="Last target date, inclusive.")
    parser.add_argument("--station", default=STATION, help="Station name; defaults to the first station.")
    parser.add_argument("--lead-day", type=int, default=LEAD_DAY, help="One forecast horizon; default plots all horizons.")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Output PNG path.")
    parser.add_argument("--catalog-path", type=Path, default=CATALOG_PATH)
    parser.add_argument("--device", default=DEVICE, help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--show-progress", dest="show_progress", action="store_true")
    parser.add_argument("--quiet", dest="show_progress", action="store_false")
    parser.set_defaults(show_progress=SHOW_PROGRESS)
    parser.add_argument("--progress-time-chunk-days", type=int, default=PROGRESS_TIME_CHUNK_DAYS)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.run_dir is None:
        parser.error("Pass --run-dir with the trained model run directory.")
    if args.start_date is None or args.end_date is None:
        parser.error("Pass both --start-date and --end-date.")
    output = create_timeperiod_plot(
        args.run_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output,
        station=args.station,
        lead_day=args.lead_day,
        catalog_path=args.catalog_path,
        device=args.device,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
        progress_time_chunk_days=args.progress_time_chunk_days,
    )
    print(f"Time-period plot saved to: {output}")
    return output


if __name__ == "__main__":
    main()
