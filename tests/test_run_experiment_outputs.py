from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_experiment as experiment_runner  # noqa: E402
from Evaluation import experiment_outputs  # noqa: E402
from Evaluation.plot_style import prediction_palette  # noqa: E402


class ExperimentParametersMarkdownTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = ROOT / "tests" / f"_parameters_markdown_test_{uuid4().hex}"
        self.output_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_table_contains_all_source_selectable_parameters(self):
        parameter_values = experiment_runner._default_run_parameter_values()
        parameter_values["state"] = "RS|SC"

        path = experiment_runner._write_parameters_markdown(
            self.output_dir,
            parameter_values,
            comparative_run=False,
            comparative_parameter="lstm_layers",
            output_root=self.output_dir,
            sweep_name=None,
        )

        content = path.read_text(encoding="utf-8")
        parameter_rows = [line for line in content.splitlines() if line.startswith("| `")]
        expected_names = {name.upper() for name in parameter_values}
        expected_names.update(
            {"COMPARATIVE_RUN", "COMPARATIVE_PARAMETER", "OUTPUT_ROOT", "SWEEP_NAME"}
        )

        self.assertEqual(path.name, "parameters.md")
        self.assertEqual(len(parameter_rows), len(expected_names))
        self.assertEqual(len(expected_names), 57)
        self.assertIn("EMPTY_GRAPH", expected_names)
        self.assertIn("LEARN_STD", expected_names)
        self.assertIn("LEARN_SELF_ATT", expected_names)
        self.assertIn("STATION_SIMILARITY", expected_names)
        self.assertIn("STATION_SIMILARITY_SIGMA_KM", expected_names)
        for name in expected_names:
            self.assertEqual(sum(f"| `{name}` |" in row for row in parameter_rows), 1, name)
        self.assertIn(r"`RS\|SC`", content)
        self.assertIn(
            f"| `LEARN_STD` | `{str(bool(experiment_runner.LEARN_STD)).lower()}` |",
            content,
        )
        self.assertIn(
            f"| `LEARN_SELF_ATT` | `{str(bool(experiment_runner.LEARN_SELF_ATT)).lower()}` |",
            content,
        )
        self.assertIn("| `EMPTY_GRAPH` | `false` |", content)
        self.assertIn("| `STATION_SIMILARITY` | `gaussian` |", content)
        self.assertIn("| `COMPARATIVE_RUN` | `false` |", content)
        self.assertIn("| `SWEEP_NAME` | `None` |", content)

    def test_materialized_values_and_structured_lists_remain_distinguishable(self):
        parameter_values = experiment_runner._default_run_parameter_values()
        parameter_values["lstm_layers"] = 2
        parameter_values["loss_quantiles"] = [0.7, 0.85]

        path = experiment_runner._write_parameters_markdown(
            self.output_dir,
            parameter_values,
            comparative_run=True,
            comparative_parameter="lstm_layers",
            output_root=self.output_dir,
            sweep_name="run_001__lstm_layers=2",
        )

        content = path.read_text(encoding="utf-8")
        self.assertIn("| `LSTM_LAYERS` | `2` |", content)
        self.assertIn("| `LOSS_QUANTILES` | `[0.7, 0.85]` |", content)
        self.assertIn("| `COMPARATIVE_RUN` | `true` |", content)

    def test_run_writes_parameter_snapshot_before_loading_data(self):
        parameter_values = {
            name: experiment_runner._parameter_options(name, value)[0]
            for name, value in experiment_runner._default_run_parameter_values().items()
        }
        parameter_values["lstm_layers"] = 2
        config = experiment_runner._config_from_parameter_values(parameter_values)

        with (
            patch.object(
                experiment_runner,
                "resolve_catalog_path",
                side_effect=RuntimeError("stop after output creation"),
            ),
            self.assertRaisesRegex(RuntimeError, "stop after output creation"),
        ):
            experiment_runner.run_experiment(
                config=config,
                output_root=self.output_dir,
                sweep_name="early_failure_run",
                show_console_info=False,
            )

        parameters_path = self.output_dir / "early_failure_run" / "parameters.md"
        self.assertTrue(parameters_path.is_file())
        content = parameters_path.read_text(encoding="utf-8")
        self.assertIn("| `LSTM_LAYERS` | `2` |", content)
        self.assertEqual(
            len([line for line in content.splitlines() if line.startswith("| `")]),
            57,
        )


class NodeStandardDeviationPlotTests(unittest.TestCase):
    def _diagnostics(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "lead_day": [1, 1, 1, 1],
                "target_time": pd.date_range("2026-01-01", periods=4, freq="D"),
                "actual_node_std_mm": [1.0, 3.0, np.nan, 8.0],
                "predicted_node_std_mm": [2.0, 4.0, 6.0, 8.0],
                "actual_node_mean_mm": [10.0, 20.0, np.nan, 40.0],
                "predicted_node_mean_mm": [15.0, 25.0, 35.0, 45.0],
            }
        )

    def test_standard_deviation_plot_uses_only_rolling_spread_curves(self):
        output_dir = ROOT / "tests" / f"_node_std_plot_test_{uuid4().hex}"
        output_dir.mkdir()
        try:
            with patch.object(experiment_outputs, "save_figure", autospec=True) as save_figure:
                path = experiment_outputs._save_node_standard_deviation_by_time(
                    output_dir,
                    self._diagnostics(),
                    n_leads=1,
                )

            figure = save_figure.call_args.args[0]
            lines = figure.axes[0].lines
            palette = prediction_palette()
            solid_lines = [line for line in lines if line.get_linestyle() == "-"]
            dotted_lines = [line for line in lines if line.get_linestyle() == ":"]

            self.assertEqual(path, output_dir / "01_node_standard_deviation_by_time.png")
            self.assertEqual(len(solid_lines), 2)
            self.assertEqual(len(dotted_lines), 0)
            self.assertFalse(any(line.get_alpha() == 0.18 for line in lines))
            self.assertEqual(
                [line.get_label() for line in solid_lines],
                ["Actual (30-day mean)", "Prediction (30-day mean)"],
            )
            self.assertEqual([line.get_color() for line in solid_lines], [palette["actual"], palette["predicted"]])
            self.assertEqual(figure.axes[0].get_title(), "Lead Day 1")
            self.assertEqual(figure.axes[0].get_xlabel(), "")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_daily_station_mean_plot_matches_standard_deviation_line_style(self):
        output_dir = ROOT / "tests" / f"_node_mean_plot_test_{uuid4().hex}"
        output_dir.mkdir()
        try:
            with patch.object(experiment_outputs, "save_figure", autospec=True) as save_figure:
                path = experiment_outputs._save_node_mean_by_time(
                    output_dir,
                    self._diagnostics(),
                    n_leads=1,
                )

            figure = save_figure.call_args.args[0]
            lines = figure.axes[0].lines
            solid_lines = [line for line in lines if line.get_linestyle() == "-"]
            self.assertEqual(path, output_dir / "02_node_mean_by_time.png")
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(solid_lines), 2)
            self.assertEqual(
                [line.get_label() for line in solid_lines],
                ["Actual daily station mean", "Prediction daily station mean"],
            )
            self.assertEqual([line.get_linewidth() for line in solid_lines], [2.0, 2.0])
            self.assertEqual([line.get_alpha() for line in solid_lines], [None, None])
            self.assertEqual(figure.axes[0].get_title(), "Lead Day 1")
            self.assertEqual(figure.axes[0].get_xlabel(), "")
            np.testing.assert_allclose(
                solid_lines[0].get_ydata(), [10.0, 20.0, 40.0]
            )
            np.testing.assert_allclose(solid_lines[1].get_ydata(), [15.0, 25.0, 35.0, 45.0])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_automatic_prediction_overview_omits_date_axis_label(self):
        output_dir = ROOT / "tests" / f"_prediction_overview_test_{uuid4().hex}"
        output_dir.mkdir()
        try:
            actual = np.array([[[1.0]], [[2.0]]])
            predicted = np.array([[[1.5]], [[2.5]]])
            target_times = np.array([["2026-01-01"], ["2026-01-02"]], dtype="datetime64[ns]")
            with patch.object(experiment_outputs, "save_figure", autospec=True) as save_figure:
                experiment_outputs._save_prediction_overview(
                    output_dir,
                    actual,
                    predicted,
                    target_times,
                    plot_station_idx=0,
                    plot_station_name="Station A",
                )

            axis = save_figure.call_args.args[0].axes[0]
            self.assertEqual(axis.get_xlabel(), "")
            self.assertEqual(
                [text.get_text() for text in axis.get_legend().get_texts()],
                ["ERA5 Lead day 1", "GLSTM Lead Day 1"],
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_absolute_error_boxplots_cover_every_lead_and_the_aggregate(self):
        output_dir = ROOT / "tests" / f"_absolute_error_boxplot_test_{uuid4().hex}"
        output_dir.mkdir()
        try:
            rows = []
            for lead_day in range(1, 6):
                for station, error in (("Station A", float(lead_day)), ("Station B", float(lead_day + 1))):
                    rows.append(
                        {
                            "lead_day": lead_day,
                            "station": station,
                            "actual_mm": 20.0,
                            "predicted_mm": 20.0 - error,
                        }
                    )

            with patch.object(experiment_outputs, "save_figure", autospec=True) as save_figure:
                path = experiment_outputs._save_absolute_error_boxplots(
                    output_dir,
                    pd.DataFrame(rows),
                    n_leads=5,
                )

            axis = save_figure.call_args.args[0].axes[0]
            self.assertEqual(path, output_dir / "15_absolute_prediction_error_boxplots.png")
            self.assertEqual(
                [label.get_text() for label in axis.get_xticklabels()],
                ["D+1\n(n=2)", "D+2\n(n=2)", "D+3\n(n=2)", "D+4\n(n=2)", "D+5\n(n=2)", "All lead days\n(n=10)"],
            )
            self.assertEqual(len(axis.patches), 6)
            self.assertEqual(axis.get_ylabel(), "Absolute error |actual − predicted| (mm)")
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_forecast_diagnostics_writes_the_absolute_error_boxplot(self):
        output_dir = ROOT / "tests" / f"_forecast_diagnostics_boxplot_test_{uuid4().hex}"
        output_dir.mkdir()
        try:
            actual = np.arange(1, 11, dtype=float).reshape(1, 5, 2)
            predicted = actual - 0.5
            target_times = np.array(
                [[f"2026-01-0{lead_day}" for lead_day in range(1, 6)]],
                dtype="datetime64[ns]",
            )
            lead_df = experiment_outputs._prediction_dataframe(
                actual,
                predicted,
                target_times,
                ["Station A", "Station B"],
            )
            with patch.object(experiment_outputs, "save_figure", autospec=True) as save_figure:
                experiment_outputs._save_forecast_lead_day_diagnostics(
                    output_dir,
                    lead_df,
                    actual,
                    predicted,
                    target_times,
                    plot_station_idx=0,
                    plot_station_name="Station A",
                )

            saved_paths = {call.args[1] for call in save_figure.call_args_list}
            self.assertIn(
                output_dir
                / "forecast_horizon_diagnostics"
                / "15_absolute_prediction_error_boxplots.png",
                saved_paths,
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
