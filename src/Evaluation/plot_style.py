"""Shared seaborn styling helpers for experiment figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


FIGURE_FACE = "#fbfcfe"
AXIS_FACE = "#ffffff"
GRID_COLOR = "#d7dde5"
TEXT_COLOR = "#1f2933"
REFERENCE_COLOR = "#111827"

PLOT_COLORS = {
    "actual": "#1f4e79",
    "predicted": "#d95f02",
    "scatter": "#2a9d8f",
    "rmse": "#7b3294",
    "mae": "#008837",
    "train": "#2b6cb0",
    "validation": "#e07a5f",
    "r2": "#6a4c93",
}


def apply_seaborn_theme() -> None:
    """Apply a high-contrast Seaborn theme suitable for projected figures."""
    sns.set_theme(
        context="talk",
        style="whitegrid",
        palette=[PLOT_COLORS["actual"], PLOT_COLORS["predicted"], PLOT_COLORS["scatter"], PLOT_COLORS["r2"]],
        font_scale=1.0,
        rc={
            "figure.facecolor": FIGURE_FACE,
            "axes.facecolor": AXIS_FACE,
            "axes.edgecolor": "#c8d0da",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "axes.titleweight": "semibold",
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 2.3,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": AXIS_FACE,
            "legend.edgecolor": "#d9dee7",
            "savefig.facecolor": FIGURE_FACE,
            "savefig.bbox": "tight",
        },
    )


def style_axis(ax, *, grid_axis: str = "both") -> None:
    """Apply the shared Seaborn-compatible axis treatment."""
    ax.grid(False, axis="both")
    ax.grid(True, axis=grid_axis, alpha=0.55)
    sns.despine(ax=ax, left=False, bottom=False)


def style_time_axis(ax) -> None:
    """Style a date axis without an x-label, leaving room for slide content."""
    ax.set_xlabel("")
    style_axis(ax, grid_axis="y")


def save_figure(fig, output_path, *, dpi: int = 190, tight: bool = True) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor=FIGURE_FACE, bbox_inches="tight")
    return output_path


def prediction_palette() -> dict[str, str]:
    return dict(PLOT_COLORS)


def lead_day_legend_labels(lead_day: int) -> tuple[str, str]:
    """Return the standard ERA5 and GLSTM legend labels for one lead day."""
    day = int(lead_day)
    if day < 1:
        raise ValueError("lead_day must be positive.")
    return f"ERA5 Lead day {day}", f"GLSTM Lead Day {day}"
