from __future__ import annotations

import json
import math
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Evaluation import experiment_outputs  # noqa: E402
from Evaluation.metrics import numpy_rain_classification_metrics  # noqa: E402


class RainClassificationMetricTests(unittest.TestCase):
    def test_metrics_use_the_configured_strict_rain_threshold(self):
        metrics = numpy_rain_classification_metrics(
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 4.0, 2.0, 0.0],
            threshold=1.0,
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["n_classification_targets"], 4)
        self.assertTrue(math.isclose(metrics["precision"], 0.5))
        self.assertTrue(math.isclose(metrics["recall"], 0.5))
        self.assertTrue(math.isclose(metrics["accuracy"], 0.5))
        self.assertTrue(math.isclose(metrics["auc"], 0.375))

    def test_auc_is_undefined_when_test_data_has_one_class(self):
        metrics = numpy_rain_classification_metrics([0.0, 0.5], [0.0, 2.0], threshold=1.0)

        self.assertTrue(math.isnan(metrics["auc"]))
        self.assertTrue(math.isnan(metrics["recall"]))

    def test_metrics_accept_a_negative_threshold_for_standardized_targets(self):
        metrics = numpy_rain_classification_metrics(
            [-10.0, -8.0],
            [-11.0, -7.0],
            threshold=-9.0,
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [0, 1]])


class RainClassificationOutputTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = ROOT / "tests" / f"_rain_classification_test_{uuid4().hex}"
        self.output_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_prediction_outputs_write_metrics_and_confusion_matrix_artifacts(self):
        actual = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(4, 1, 1)
        predicted = torch.tensor([0.0, 4.0, 2.0, 0.0]).reshape(4, 1, 1)
        test_y = xr.DataArray(
            actual.numpy(),
            dims=("sample", "lead_day", "station"),
            coords={
                "sample": range(4),
                "lead_day": [1],
                "station": ["A"],
                "target_time": (
                    ("sample", "lead_day"),
                    np.asarray(
                        [["2026-01-01"], ["2026-01-02"], ["2026-01-03"], ["2026-01-04"]],
                        dtype="datetime64[ns]",
                    ),
                ),
            },
        )

        with (
            patch.object(experiment_outputs, "_save_prediction_overview", autospec=True),
            patch.object(experiment_outputs, "_save_prediction_timeseries_splits", autospec=True),
            patch.object(
                experiment_outputs,
                "_save_forecast_lead_day_diagnostics",
                autospec=True,
                return_value=pd.DataFrame(),
            ),
            patch.object(experiment_outputs, "save_oversmoothing_diagnostics", autospec=True),
        ):
            experiment_outputs.save_prediction_outputs(
                self.output_dir,
                actual,
                predicted,
                test_y,
                confusion_matrix_threshold=1.0,
            )

        physical_metrics = json.loads(
            (self.output_dir / "logs" / "test_metrics_physical_scale.json").read_text(encoding="utf-8")
        )
        confusion_metrics = json.loads(
            (self.output_dir / "logs" / "test_confusion_matrix.json").read_text(encoding="utf-8")
        )
        self.assertEqual(physical_metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(confusion_metrics["confusion_matrix_threshold_mm"], 1.0)
        self.assertEqual(confusion_metrics["matrix_layout"].split(";")[0], "rows=actual [não chove, chove]")
        self.assertTrue((self.output_dir / "confusion_matrix.png").is_file())


if __name__ == "__main__":
    unittest.main()
