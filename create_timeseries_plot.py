"""Create one Porto Alegre true-versus-predicted plot per forecast lead.

The plots are built from ``test_predictions_by_lead_day.csv`` saved in a
completed experiment run.  Dates are target dates and both bounds are
inclusive.

Edit the configuration block below and run this file from Python or an IDE.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from Evaluation.experiment_outputs import _normalize_station_name  # noqa: E402
from plot_timeperiod import (  # noqa: E402
    _parse_date,
    _prediction_columns,
    save_timeperiod_plot,
)


# Configuration: edit these values directly in this file.
START_DATE = "2021-09-01"
END_DATE = "2021-11-01"

#START_DATE = "2024-03-01"
#END_DATE = "2024-06-01"

RUN_PATH = Path(
    r"C:\Local Repository\LSTM-GNN\Experiments\run_experiment\07_09_2026\glstm_sweep_20260909_124805"
)
LEAD_DAYS = [1, 5]


DEFAULT_LEAD_DAYS = tuple(LEAD_DAYS)
PORTO_ALEGRE_STATION = "PORTO ALEGRE JARDIM BOTANICO"
PLOTS_DIRECTORY = "plots_presentation"
PREDICTION_FILENAMES = (
    "test_predictions_by_lead_day.csv",
    "inference_predictions_by_lead_day.csv",
)
CSV_CHUNK_SIZE = 250_000


def _normalize_lead_days(lead_days: Sequence[int]) -> tuple[int, ...]:
    """Validate lead days and remove duplicates while preserving their order."""
    if isinstance(lead_days, (str, bytes)):
        raise ValueError("lead_days must be a non-empty sequence of positive integers.")
    try:
        values = list(lead_days)
    except TypeError as exc:
        raise ValueError(
            "lead_days must be a non-empty sequence of positive integers."
        ) from exc
    if not values:
        raise ValueError("lead_days must contain at least one lead day.")

    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("Every lead day must be a positive integer.")
        lead_day = int(value)
        if lead_day < 1:
            raise ValueError("Every lead day must be a positive integer.")
        if lead_day not in seen:
            normalized.append(lead_day)
            seen.add(lead_day)
    return tuple(normalized)


def _prediction_path(run_path: Path) -> Path:
    for filename in PREDICTION_FILENAMES:
        candidate = run_path / filename
        if candidate.is_file():
            return candidate
    expected = " or ".join(str(run_path / name) for name in PREDICTION_FILENAMES)
    raise FileNotFoundError(
        "The run has no saved prediction CSV. Expected " + expected + "."
    )


def _read_porto_alegre_period(
    prediction_path: Path,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    lead_days: tuple[int, ...],
) -> pd.DataFrame:
    """Read only the requested station, target period, and lead days."""
    header = pd.read_csv(prediction_path, nrows=0)
    actual_column, predicted_column = _prediction_columns(header)
    use_columns = [
        "lead_day",
        "target_time",
        "station",
        actual_column,
        predicted_column,
    ]

    requested_station_key = _normalize_station_name(PORTO_ALEGRE_STATION)
    selected_chunks: list[pd.DataFrame] = []
    station_names: set[str] = set()
    station_leads: set[int] = set()
    station_ranges: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    station_date_min: pd.Timestamp | None = None
    station_date_max: pd.Timestamp | None = None

    for chunk in pd.read_csv(
        prediction_path,
        usecols=use_columns,
        chunksize=CSV_CHUNK_SIZE,
    ):
        chunk["station"] = chunk["station"].astype(str)
        station_names.update(chunk["station"].drop_duplicates().tolist())

        target_times = pd.to_datetime(chunk["target_time"], errors="raise").dt.normalize()
        numeric_leads = pd.to_numeric(chunk["lead_day"], errors="raise")
        if not numeric_leads.mod(1).eq(0).all():
            raise ValueError(f"Prediction CSV has non-integer lead_day values: {prediction_path}")
        chunk["target_time"] = target_times
        chunk["lead_day"] = numeric_leads.astype(int)

        station_mask = chunk["station"].map(_normalize_station_name).eq(
            requested_station_key
        )
        if not station_mask.any():
            continue

        station_rows = chunk.loc[station_mask]
        station_leads.update(int(value) for value in station_rows["lead_day"].unique())
        for lead_day, lead_rows in station_rows.groupby("lead_day"):
            lead_day = int(lead_day)
            lead_min = lead_rows["target_time"].min()
            lead_max = lead_rows["target_time"].max()
            previous = station_ranges.get(lead_day)
            station_ranges[lead_day] = (
                lead_min if previous is None else min(previous[0], lead_min),
                lead_max if previous is None else max(previous[1], lead_max),
            )
        current_min = station_rows["target_time"].min()
        current_max = station_rows["target_time"].max()
        station_date_min = (
            current_min
            if station_date_min is None
            else min(station_date_min, current_min)
        )
        station_date_max = (
            current_max
            if station_date_max is None
            else max(station_date_max, current_max)
        )

        period_mask = station_rows["target_time"].between(start_date, end_date)
        lead_mask = station_rows["lead_day"].isin(lead_days)
        selected = station_rows.loc[period_mask & lead_mask]
        if not selected.empty:
            selected_chunks.append(selected.copy())

    if station_date_min is None or station_date_max is None:
        available = ", ".join(sorted(station_names)) or "none"
        raise ValueError(
            f"Station {PORTO_ALEGRE_STATION!r} was not found in {prediction_path}. "
            f"Available stations: {available}."
        )

    missing_from_run = [
        lead_day for lead_day in lead_days if lead_day not in station_leads
    ]
    if missing_from_run:
        available_text = ", ".join(
            f"D+{lead_day}" for lead_day in sorted(station_leads)
        ) or "none"
        missing_text = ", ".join(f"D+{lead_day}" for lead_day in missing_from_run)
        raise ValueError(
            f"The run has no predictions for {missing_text}. "
            f"Available lead days for Porto Alegre: {available_text}."
        )

    outside_ranges = []
    for lead_day in lead_days:
        lead_start, lead_end = station_ranges[lead_day]
        if start_date < lead_start or end_date > lead_end:
            outside_ranges.append(
                f"D+{lead_day} ({lead_start.date()} to {lead_end.date()})"
            )
    if outside_ranges:
        raise ValueError(
            f"Requested period {start_date.date()} to {end_date.date()} is outside "
            "the saved prediction range for " + ", ".join(outside_ranges) + "."
        )

    if not selected_chunks:
        raise ValueError(
            "No Porto Alegre predictions were found between "
            f"{start_date.date()} and {end_date.date()}. The saved station period is "
            f"{station_date_min.date()} to {station_date_max.date()}."
        )

    predictions = pd.concat(selected_chunks, ignore_index=True)
    available_in_period = set(int(value) for value in predictions["lead_day"].unique())
    missing = [lead_day for lead_day in lead_days if lead_day not in available_in_period]
    if missing:
        available_text = ", ".join(
            f"D+{lead_day}" for lead_day in sorted(station_leads)
        ) or "none"
        missing_text = ", ".join(f"D+{lead_day}" for lead_day in missing)
        raise ValueError(
            f"No predictions were found for {missing_text} in the requested period. "
            f"Available lead days for Porto Alegre: {available_text}."
        )

    numeric_values = predictions[[actual_column, predicted_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    predictions[[actual_column, predicted_column]] = numeric_values
    finite_pairs = np.isfinite(numeric_values.to_numpy(dtype=float)).all(axis=1)
    leads_without_values = [
        lead_day
        for lead_day in lead_days
        if not (finite_pairs & predictions["lead_day"].eq(lead_day).to_numpy()).any()
    ]
    if leads_without_values:
        lead_text = ", ".join(f"D+{lead_day}" for lead_day in leads_without_values)
        raise ValueError(f"No finite actual/predicted pairs were found for {lead_text}.")
    return predictions


def create_timeseries_plot(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    run_path: Path | str,
    lead_days: Sequence[int] = DEFAULT_LEAD_DAYS,
) -> list[Path]:
    """Save one Porto Alegre true-versus-predicted PNG per requested lead day."""
    resolved_run = Path(run_path).resolve()
    if not resolved_run.is_dir():
        raise FileNotFoundError(f"Run directory not found: {resolved_run}")

    start = _parse_date(str(start_date), "start_date")
    end = _parse_date(str(end_date), "end_date")
    if start > end:
        raise ValueError("start_date must not be after end_date.")
    requested_leads = _normalize_lead_days(lead_days)

    predictions = _read_porto_alegre_period(
        _prediction_path(resolved_run),
        start_date=start,
        end_date=end,
        lead_days=requested_leads,
    )

    output_directory = resolved_run / PLOTS_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for lead_day in requested_leads:
        output_path = output_directory / (
            "porto_alegre_true_vs_predicted_"
            f"lead_day_{lead_day:02d}_{start:%Y%m%d}_{end:%Y%m%d}.png"
        )
        output_paths.append(
            save_timeperiod_plot(
                predictions,
                start_date=start,
                end_date=end,
                output_path=output_path,
                station=PORTO_ALEGRE_STATION,
                lead_day=lead_day,
                show_titles=False,
            )
        )
    return output_paths


# The plural alias is convenient for callers while the singular name mirrors
# the requested script name and remains the canonical public entry point.
create_timeseries_plots = create_timeseries_plot


def main() -> list[Path]:
    """Create the plots using the configuration at the top of this file."""
    output_paths = create_timeseries_plot(
        START_DATE,
        END_DATE,
        RUN_PATH,
        LEAD_DAYS,
    )
    for output_path in output_paths:
        print(f"Time-series plot saved to: {output_path}")
    return output_paths


if __name__ == "__main__":
    main()
