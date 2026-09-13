from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dataset_analysis as analysis  # noqa: E402


class DatasetAnalysisPlotTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=24, freq="D")
        self.series = {
            "precipitation": xr.DataArray(
                np.linspace(0.0, 20.0, len(dates)), coords={"time": dates}, dims="time"
            ),
            "t2m": xr.DataArray(
                np.linspace(18.0, 30.0, len(dates)), coords={"time": dates}, dims="time"
            ),
            "d2m": xr.DataArray(
                np.linspace(10.0, 22.0, len(dates)), coords={"time": dates}, dims="time"
            ),
        }
        self.thresholds = {0.5: 10.0, 0.9: 18.0}
        self.figures = []

    def tearDown(self):
        for figure in self.figures:
            plt.close(figure)

    def _capture_figure(self, figure, filename):
        self.figures.append(figure)
        return Path(filename)

    def test_titles_only_name_the_plotted_variables(self):
        with patch("dataset_analysis._save_figure", side_effect=self._capture_figure):
            analysis.plot_histograms(
                self.series,
                "day",
                self.thresholds,
                "Station A",
                quantile_legend=False,
            )
            analysis.plot_timeseries(
                self.series,
                "day",
                self.thresholds,
                "Station A",
                quantile_legend=False,
            )
            analysis.plot_scatters(
                self.series,
                "day",
                self.thresholds,
                "Station A",
                quantile_legend=False,
            )
            analysis.plot_boxplots(
                self.series,
                "year",
                self.thresholds,
                "Station A",
                quantile_legend=False,
            )

        histogram_figures = self.figures[:3]
        timeseries_figure = self.figures[3]
        scatter_figures = self.figures[4:7]
        boxplot_figures = self.figures[7:]

        self.assertEqual(
            [figure.axes[0].get_title() for figure in histogram_figures],
            ["Precipitation", "Temperature", "Dew point"],
        )
        self.assertEqual(
            [axis.get_title() for axis in timeseries_figure.axes],
            ["Precipitation", "Temperature", "Dew point"],
        )
        self.assertIsNone(timeseries_figure._suptitle)
        self.assertEqual(
            [figure.axes[0].get_title() for figure in scatter_figures],
            [
                "Precipitation vs. Temperature",
                "Precipitation vs. Dew point",
                "Temperature vs. Dew point",
            ],
        )
        self.assertEqual(
            [figure.axes[0].get_title() for figure in boxplot_figures],
            ["Precipitation", "Temperature", "Dew point"],
        )

    def test_quantile_legend_false_hides_threshold_lines_and_legends(self):
        with patch("dataset_analysis._save_figure", side_effect=self._capture_figure):
            analysis.plot_histograms(
                {"precipitation": self.series["precipitation"]},
                "day",
                self.thresholds,
                "Station A",
                quantile_legend=False,
            )
            analysis.plot_timeseries(
                {"precipitation": self.series["precipitation"]},
                "day",
                self.thresholds,
                "Station A",
                quantile_legend=False,
            )
            analysis.plot_boxplots(
                {"precipitation": self.series["precipitation"]},
                "year",
                self.thresholds,
                "Station A",
                quantile_legend=False,
            )

        for figure in self.figures:
            axis = figure.axes[0]
            self.assertIsNone(axis.get_legend())
            self.assertFalse(any(line.get_label().startswith("Q") for line in axis.lines))

    def test_quantile_legend_true_draws_thresholds(self):
        with patch("dataset_analysis._save_figure", side_effect=self._capture_figure):
            analysis.plot_histograms(
                {"precipitation": self.series["precipitation"]},
                "day",
                self.thresholds,
                "Station A",
                quantile_legend=True,
            )

        axis = self.figures[0].axes[0]
        self.assertTrue(any(line.get_label().startswith("Q") for line in axis.lines))
        self.assertIsNotNone(axis.get_legend())

    def test_precipitation_histogram_uses_log_frequency_for_rare_high_values(self):
        precipitation = self.series["precipitation"].copy()
        precipitation[-1] = 250.0
        with patch("dataset_analysis._save_figure", side_effect=self._capture_figure):
            analysis.plot_histograms(
                {"precipitation": precipitation, "t2m": self.series["t2m"]},
                "day",
                {},
                "Station A",
            )

        precipitation_axis = self.figures[0].axes[0]
        temperature_axis = self.figures[1].axes[0]
        self.assertEqual(precipitation_axis.get_yscale(), "log")
        self.assertIn("log scale", precipitation_axis.get_ylabel())
        self.assertEqual(temperature_axis.get_yscale(), "linear")

    def test_quantile_legend_must_be_boolean(self):
        with patch.object(analysis, "quantile_legend", "false"):
            with self.assertRaisesRegex(ValueError, "quantile_legend"):
                analysis._validate_choices()


if __name__ == "__main__":
    unittest.main()
