"""Command-line entry point for RS precipitation animation maps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from Evaluation.rs_animation_maps import save_rs_precipitation_animation_frames  # noqa: E402

ORIGIN_PATH = "Experiments/run_experiment/10_09_2026/glstm_sweep_20260910_170203"

# Editable defaults. Command-line arguments override them.
RUN_DIR: Path | None = Path(
   ORIGIN_PATH
)
SAMPLE: int | None = None  # Legacy mode; use -1 for the last test sample.
START_DATE: str | None = "2021-09-01"
END_DATE: str | None = "2021-10-15"
LEAD_DAY: int | None = 1
OUTPUT_DIR: Path | None = Path(
    ORIGIN_PATH + "/animation_beamer"
)
VMAX: float | None = None
RESIDUAL_VMAX: float | None = None


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export matching Predict/Real/Residual RS PNG frames for a fixed lead day and "
            "target-date period, for the LaTeX animate package."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Trained run directory.")
    parser.add_argument(
        "--sample",
        type=int,
        default=SAMPLE,
        help="Legacy mode: CSV sample id; use -1 for the last sample.",
    )
    parser.add_argument("--start-date", default=START_DATE, help="First target date, inclusive.")
    parser.add_argument("--end-date", default=END_DATE, help="Last target date, inclusive.")
    parser.add_argument("--lead-day", type=int, default=LEAD_DAY, help="Fixed lead day to animate.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--boundary-path", type=Path, help="Optional offline RS GeoJSON file.")
    parser.add_argument("--vmax", type=float, default=VMAX, help="Fixed common scale maximum in mm.")
    parser.add_argument(
        "--residual-vmax",
        type=float,
        default=RESIDUAL_VMAX,
        help="Fixed symmetric residual scale magnitude in mm.",
    )
    parser.add_argument("--node-size", type=float, default=74.0)
    parser.add_argument("--dpi", type=int, default=190)
    parser.add_argument("--fps", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.run_dir is None:
        parser.error("Set RUN_DIR in export_rs_animation_maps.py or pass --run-dir.")
    destination = save_rs_precipitation_animation_frames(
        args.run_dir,
        sample=args.sample,
        start_date=args.start_date,
        end_date=args.end_date,
        lead_day=args.lead_day,
        output_dir=args.output_dir,
        predictions_csv=args.predictions_csv,
        boundary_path=args.boundary_path,
        vmax=args.vmax,
        residual_vmax=args.residual_vmax,
        node_size=args.node_size,
        dpi=args.dpi,
        fps=args.fps,
    )
    print(f"Animation frames saved to: {destination}")
    return destination


if __name__ == "__main__":
    main()
