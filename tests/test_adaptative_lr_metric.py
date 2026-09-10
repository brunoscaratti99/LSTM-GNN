import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Training.Training_Routines import (  # noqa: E402
    adaptative_lr_metric_value,
    resolve_adaptative_lr_metric,
)


class AdaptativeLearningRateMetricTests(unittest.TestCase):
    def test_error_metrics_are_minimized(self):
        for metric in ("loss", "mse", "rmse", "mae", "mape"):
            with self.subTest(metric=metric):
                self.assertEqual(resolve_adaptative_lr_metric(metric), (metric, "min"))

    def test_r2_metrics_are_maximized(self):
        for metric in ("r2", "r2_batch_mean"):
            with self.subTest(metric=metric):
                self.assertEqual(resolve_adaptative_lr_metric(metric), (metric, "max"))

    def test_rmse_is_derived_from_validation_mse(self):
        value = adaptative_lr_metric_value({"mse": 2.25}, "rmse")
        self.assertTrue(math.isclose(value, 1.5))

    def test_metric_aliases_are_normalized(self):
        self.assertEqual(resolve_adaptative_lr_metric("R2 score"), ("r2", "max"))
        self.assertEqual(
            resolve_adaptative_lr_metric("root-mean-squared-error"),
            ("rmse", "min"),
        )

    def test_unknown_metric_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Available metrics"):
            resolve_adaptative_lr_metric("accuracy")


if __name__ == "__main__":
    unittest.main()
