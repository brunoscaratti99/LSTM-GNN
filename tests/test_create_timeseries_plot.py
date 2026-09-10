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

from create_timeseries_plot import (  # noqa: E402
    _prediction_path,
    _read_porto_alegre_period,
    create_timeseries_plot,
)


class CreateTimeseriesPlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = ROOT / "tests" / f"_create_timeseries_plot_test_{uuid4().hex}"
        self.temporary.mkdir()
        self.run_path = self.temporary / "run"
        self.run_path.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temporary, ignore_errors=True)

    def _write_predictions(self, *, value_columns=("actual_mm", "predicted_mm")):
        rows = []
        for date in pd.date_range("2024-04-30", "2024-05-03"):
            for lead_day in (1, 3, 5):
                for station, offset in (
                    ("PORTO ALEGRE - BELEM NOVO", 1_000.0),
                    ("PORTO ALEGRE - JARDIM BOTANICO", 0.0),
                    ("CAXIAS DO SUL", 2_000.0),
                ):
                    rows.append(
                        {
                            "sample": len(rows),
                            "lead_day": lead_day,
                            "target_time": date.date().isoformat(),
                            "station": station,
                            value_columns[0]: offset + date.day,
                            value_columns[1]: offset + date.day + 0.5,
                        }
                    )
        pd.DataFrame(rows).to_csv(
            self.run_path / "test_predictions_by_lead_day.csv", index=False
        )

    def test_default_creates_one_png_per_lead_in_presentation_directory(self):
        self._write_predictions()

        selected = _read_porto_alegre_period(
            _prediction_path(self.run_path),
            start_date=pd.Timestamp("2024-05-01"),
            end_date=pd.Timestamp("2024-05-02"),
            lead_days=(1, 5),
        )
        self.assertEqual(
            selected["station"].unique().tolist(),
            ["PORTO ALEGRE - JARDIM BOTANICO"],
        )
        self.assertEqual(selected["target_time"].min(), pd.Timestamp("2024-05-01"))
        self.assertEqual(selected["target_time"].max(), pd.Timestamp("2024-05-02"))
        self.assertLess(selected["actual_mm"].max(), 1_000.0)

        with (
            patch("matplotlib.axes.Axes.set_title") as set_title,
            patch("matplotlib.figure.Figure.suptitle") as suptitle,
        ):
            output_paths = create_timeseries_plot(
                "2024-05-01",
                "2024-05-02",
                self.run_path,
            )

        set_title.assert_not_called()
        suptitle.assert_not_called()

        self.assertEqual(len(output_paths), 2)
        self.assertEqual([path.parent.name for path in output_paths], ["plots_presentation"] * 2)
        self.assertIn("lead_day_01", output_paths[0].name)
        self.assertIn("lead_day_05", output_paths[1].name)
        for output_path in output_paths:
            self.assertTrue(output_path.is_file())
            image = mpimg.imread(output_path)
            self.assertGreater(image.shape[0], 0)
            self.assertGreater(image.shape[1], 0)

    def test_accepts_legacy_value_columns_and_deduplicates_leads(self):
        self._write_predictions(value_columns=("actual", "predicted"))

        output_paths = create_timeseries_plot(
            "2024-05-01",
            "2024-05-02",
            self.run_path,
            lead_days=[5, 1, 5],
        )

        self.assertEqual(len(output_paths), 2)
        self.assertIn("lead_day_05", output_paths[0].name)
        self.assertIn("lead_day_01", output_paths[1].name)

    def test_rejects_missing_lead_before_creating_output_directory(self):
        self._write_predictions()

        with self.assertRaisesRegex(ValueError, r"D\+2"):
            create_timeseries_plot(
                "2024-05-01",
                "2024-05-02",
                self.run_path,
                lead_days=[1, 2],
            )

        self.assertFalse((self.run_path / "plots_presentation").exists())

    def test_rejects_invalid_period_and_empty_lead_days(self):
        self._write_predictions()

        with self.assertRaisesRegex(ValueError, "start_date"):
            create_timeseries_plot("2024-05-03", "2024-05-01", self.run_path)
        with self.assertRaisesRegex(ValueError, "at least one"):
            create_timeseries_plot(
                "2024-05-01", "2024-05-02", self.run_path, lead_days=[]
            )

    def test_rejects_a_period_not_fully_covered_by_the_saved_run(self):
        self._write_predictions()

        with self.assertRaisesRegex(ValueError, "outside the saved prediction range"):
            create_timeseries_plot(
                "2024-04-29",
                "2024-05-02",
                self.run_path,
                lead_days=[1],
            )

        self.assertFalse((self.run_path / "plots_presentation").exists())


if __name__ == "__main__":
    unittest.main()
