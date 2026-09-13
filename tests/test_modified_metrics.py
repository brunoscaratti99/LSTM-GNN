import math
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Evaluation.metrics import (  # noqa: E402
    metric_threshold_in_target_scale,
    normalize_metric_standard,
    numpy_regression_metrics,
)
from Evaluation import experiment_outputs  # noqa: E402
from Evaluation import comparative_outputs  # noqa: E402
from Training.Training_Routines import (  # noqa: E402
    eval_with_loader_stable,
    resolve_training_metric_monitors,
)


class _ShiftScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float) - 10.0


class _IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, values):
        return values * self.scale


class ModifiedMetricFormulaTests(unittest.TestCase):
    def test_modified_metrics_use_only_strictly_above_threshold_targets(self):
        actual = np.asarray([0.0, 1.0, 2.0, 4.0])
        predicted = np.asarray([100.0, 100.0, 1.0, 6.0])

        metrics = numpy_regression_metrics(
            actual,
            predicted,
            metric_standard="modified",
            metric_threshold=1.0,
        )

        self.assertEqual(metrics["count"], 2)
        self.assertTrue(math.isclose(metrics["mse"], 2.5))
        self.assertTrue(math.isclose(metrics["rmse"], math.sqrt(2.5)))
        self.assertTrue(math.isclose(metrics["mae"], 1.5))
        self.assertTrue(math.isclose(metrics["r2"], -1.5))

    def test_standard_policy_preserves_all_finite_pairs(self):
        actual = np.asarray([0.0, 1.0, 2.0, np.nan])
        predicted = np.asarray([1.0, 1.0, 4.0, 5.0])

        metrics = numpy_regression_metrics(actual, predicted)

        self.assertEqual(metrics["count"], 3)
        self.assertTrue(math.isclose(metrics["mse"], 5.0 / 3.0))
        self.assertTrue(math.isclose(metrics["mae"], 1.0))

    def test_no_eligible_targets_returns_nan_metrics(self):
        metrics = numpy_regression_metrics(
            [0.0, 1.0],
            [5.0, 5.0],
            metric_standard="modified",
            metric_threshold=1.0,
        )

        self.assertEqual(metrics["count"], 0)
        for name in ("mse", "rmse", "mae", "r2", "bias"):
            self.assertTrue(math.isnan(metrics[name]), name)

    def test_threshold_is_transformed_for_normalized_target_tensors(self):
        self.assertEqual(metric_threshold_in_target_scale(1.0, _ShiftScaler()), -9.0)

    def test_only_supported_metric_standards_are_accepted(self):
        self.assertIsNone(normalize_metric_standard(None))
        self.assertEqual(normalize_metric_standard(" MODIFIED "), "modified")
        with self.assertRaisesRegex(ValueError, "None or 'modified'"):
            normalize_metric_standard("standard")


class ModifiedMetricPipelineTests(unittest.TestCase):
    def test_loader_evaluation_filters_rmse_mae_and_r2(self):
        predictions = torch.tensor([0.0, 100.0, 1.0, 6.0]).reshape(4, 1, 1)
        targets = torch.tensor([0.0, 1.0, 2.0, 4.0]).reshape(4, 1, 1)
        loader = DataLoader(TensorDataset(predictions, targets), batch_size=2)

        metrics = eval_with_loader_stable(
            _IdentityModel(),
            loader,
            criterion=torch.nn.MSELoss(),
            use_amp=False,
            amp_device="cpu",
            amp_dtype=torch.bfloat16,
            metric_standard="modified",
            metric_threshold=1.0,
            metric_threshold_mm=1.0,
        )

        self.assertEqual(metrics["metric_target_count"], 2)
        self.assertEqual(metrics["metric_target_count_by_step"], [2])
        self.assertTrue(math.isclose(metrics["mse"], 2.5))
        self.assertTrue(math.isclose(metrics["rmse"], math.sqrt(2.5)))
        self.assertTrue(math.isclose(metrics["mae"], 1.5))
        self.assertTrue(math.isclose(metrics["r2"], -1.5))
        # The optimization loss remains defined over every target.
        self.assertTrue(math.isclose(metrics["loss"], (0.5 + 4902.5) / 2.0))

    def test_modified_policy_uses_modified_metric_for_all_patience_decisions(self):
        modified = resolve_training_metric_monitors("modified", "loss")
        self.assertEqual(modified["scheduler_metric"], "rmse")
        self.assertEqual(modified["early_stopping_metric"], "rmse")

        selected = resolve_training_metric_monitors("modified", "r2")
        self.assertEqual(selected["scheduler_metric"], "r2")
        self.assertEqual(selected["early_stopping_metric"], "r2")
        self.assertEqual(selected["early_stopping_mode"], "max")

    def test_default_policy_preserves_legacy_patience_behavior(self):
        standard = resolve_training_metric_monitors(None, "mae")
        self.assertEqual(standard["scheduler_metric"], "mae")
        self.assertEqual(standard["early_stopping_metric"], "loss")

    def test_prediction_artifacts_record_physical_modified_metrics(self):
        actual = torch.tensor([1.0, 3.0]).reshape(2, 1, 1)
        predicted = torch.tensor([100.0, 5.0]).reshape(2, 1, 1)
        test_y = xr.DataArray(
            actual.numpy(),
            dims=("sample", "lead_day", "station"),
            coords={
                "sample": [0, 1],
                "lead_day": [1],
                "station": ["A"],
                "target_time": (
                    ("sample", "lead_day"),
                    np.asarray([["2026-01-01"], ["2026-01-02"]], dtype="datetime64[ns]"),
                ),
            },
        )

        temp_dir = ROOT / "tests" / f"_modified_metrics_test_{uuid4().hex}"
        temp_dir.mkdir()
        try:
            with (
                patch.object(experiment_outputs, "_save_prediction_overview", autospec=True),
                patch.object(experiment_outputs, "_save_prediction_timeseries_splits", autospec=True),
                patch.object(
                    experiment_outputs,
                    "_save_forecast_lead_day_diagnostics",
                    autospec=True,
                    return_value=pd.DataFrame(),
                ) as lead_diagnostics,
                patch.object(experiment_outputs, "save_oversmoothing_diagnostics", autospec=True),
            ):
                predictions_frame, _metrics_frame = experiment_outputs.save_prediction_outputs(
                    temp_dir,
                    actual,
                    predicted,
                    test_y,
                    metric_standard="modified",
                    metric_threshold=1.0,
                )

            self.assertEqual(predictions_frame["metric_eligible"].tolist(), [False, True])
            physical_metrics = json.loads(
                (temp_dir / "logs" / "test_metrics_physical_scale.json").read_text(encoding="utf-8")
            )
            self.assertTrue((temp_dir / "logs" / "test_predictions_by_lead_day.csv").is_file())
            self.assertFalse((temp_dir / "test_predictions_by_lead_day.csv").exists())
            self.assertEqual(physical_metrics["n_metric_targets"], 1)
            self.assertEqual(physical_metrics["RMSE"], 2.0)
            self.assertEqual(physical_metrics["MAE"], 2.0)
            self.assertEqual(lead_diagnostics.call_args.kwargs["metric_standard"], "modified")
            self.assertEqual(lead_diagnostics.call_args.kwargs["metric_threshold"], 1.0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_comparative_report_documents_formula_and_metric_scope(self):
        temp_dir = ROOT / "tests" / f"_modified_report_test_{uuid4().hex}"
        temp_dir.mkdir()
        metric_row = {
            "run_index": 1,
            "run_name": "run_001",
            "label": "modified run",
            "station": "A",
            "lead_day": 1,
            "n_common_dates": 3,
            "n_metric_targets": 2,
            "metric_standard": "modified",
            "metric_threshold_mm": 1.0,
            "RMSE": 2.0,
            "MAE": 1.5,
            "R2": 0.25,
        }
        try:
            report_path = comparative_outputs._write_latex_report(
                temp_dir,
                temp_dir,
                "hidden_dim",
                "A",
                [],
                [metric_row],
                [],
                None,
            )
            report = report_path.read_text(encoding="utf-8")
            self.assertIn(r"I_\tau=\{i:y_i>\tau\}", report)
            self.assertIn(r"\operatorname{RMSE}_\tau", report)
            self.assertIn("target > 1 mm", report)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
