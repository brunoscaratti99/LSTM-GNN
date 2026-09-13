from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import matplotlib.image as mpimg
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plot_timeperiod import create_timeperiod_plot, save_timeperiod_plot  # noqa: E402


class TimePeriodPlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = ROOT / "tests" / f"_plot_timeperiod_test_{uuid4().hex}"
        self.temporary.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temporary, ignore_errors=True)

    def test_saves_one_png_for_all_lead_days_and_filters_dates(self):
        rows = []
        for target_time in pd.date_range("2024-04-30", "2024-05-03"):
            for lead_day in (1, 2):
                rows.append(
                    {
                        "lead_day": lead_day,
                        "target_time": target_time,
                        "station": "A",
                        "actual_mm": float(target_time.day + lead_day),
                        "predicted_mm": float(target_time.day + lead_day + 0.5),
                    }
                )
        output = save_timeperiod_plot(
            pd.DataFrame(rows),
            start_date="2024-05-01",
            end_date="2024-05-02",
            output_path=self.temporary / "period.png",
        )

        self.assertTrue(output.is_file())
        self.assertEqual(output.name, "period.png")
        image = mpimg.imread(output)
        self.assertGreater(image.shape[0], 0)
        self.assertGreater(image.shape[1], 0)

    def test_can_limit_output_to_one_lead_day_and_resolve_station(self):
        frame = pd.DataFrame(
            [
                {
                    "lead_day": lead_day,
                    "target_time": f"2024-05-0{day}",
                    "station": station,
                    "actual": float(day),
                    "predicted": float(day + 1),
                }
                for day in (1, 2)
                for lead_day in (1, 2)
                for station in ("Porto Alegre", "Caxias do Sul")
            ]
        )

        output = save_timeperiod_plot(
            frame,
            start_date="2024-05-01",
            end_date="2024-05-02",
            output_path=self.temporary / "d2.png",
            station="porto alegre",
            lead_day=2,
        )

        self.assertTrue(output.is_file())

    def test_date_axis_has_ticks_but_no_target_date_label(self):
        frame = pd.DataFrame(
            [
                {
                    "lead_day": 1,
                    "target_time": f"2024-05-0{day}",
                    "station": "A",
                    "actual_mm": float(day),
                    "predicted_mm": float(day + 1),
                }
                for day in (1, 2)
            ]
        )

        with patch("plot_timeperiod.save_figure", autospec=True) as save_figure:
            save_timeperiod_plot(
                frame,
                start_date="2024-05-01",
                end_date="2024-05-02",
                output_path=self.temporary / "period.png",
            )

        axis = save_figure.call_args.args[0].axes[0]
        self.assertEqual(axis.get_xlabel(), "")
        self.assertGreater(len(axis.get_xticks()), 0)
        self.assertEqual(
            [text.get_text() for text in axis.get_legend().get_texts()],
            ["ERA5 Lead day 1", "GLSTM Lead Day 1"],
        )

    def test_rejects_empty_date_period(self):
        frame = pd.DataFrame(
            [
                {
                    "lead_day": 1,
                    "target_time": "2024-05-01",
                    "station": "A",
                    "actual_mm": 1.0,
                    "predicted_mm": 2.0,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "No predictions"):
            save_timeperiod_plot(
                frame,
                start_date="2024-06-01",
                end_date="2024-06-02",
                output_path=self.temporary / "empty.png",
            )

    def test_model_run_is_used_to_generate_the_period_predictions(self):
        run_dir = self.temporary / "run"
        run_dir.mkdir()
        logs_dir = run_dir / "logs"
        logs_dir.mkdir()
        (logs_dir / "config.json").write_text('{"forecast_horizon": 2}', encoding="utf-8")

        def fake_run_inference(run, **kwargs):
            self.assertEqual(Path(run), run_dir.resolve())
            self.assertEqual(kwargs["start_date"], "2024-04-30")
            self.assertEqual(kwargs["end_date"], "2024-05-02")
            destination = Path(kwargs["output_dir"])
            destination_logs = destination / "logs"
            destination_logs.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "lead_day": 2,
                        "target_time": "2024-05-01",
                        "station": "A",
                        "actual_mm": 1.0,
                        "predicted_mm": 1.5,
                    },
                    {
                        "lead_day": 2,
                        "target_time": "2024-05-02",
                        "station": "A",
                        "actual_mm": 2.0,
                        "predicted_mm": 2.5,
                    },
                ]
            ).to_csv(destination_logs / "inference_predictions_by_lead_day.csv", index=False)
            return destination

        output = self.temporary / "model_period.png"
        with patch("plot_timeperiod.run_inference", side_effect=fake_run_inference):
            result = create_timeperiod_plot(
                run_dir,
                start_date="2024-05-01",
                end_date="2024-05-02",
                lead_day=2,
                output_path=output,
                show_progress=False,
            )

        self.assertEqual(result, output)
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
