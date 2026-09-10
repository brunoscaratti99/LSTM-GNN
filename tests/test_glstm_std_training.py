from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Training.Training_Routines import (  # noqa: E402
    eval_with_loader_stable,
    node_standard_deviation_target,
    train_stable,
)


class _ForecastOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forecast_bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, values):
        return values * 0.0 + self.forecast_bias


class _ForecastAndStdModel(torch.nn.Module):
    def __init__(self, *, std_shape_offset: int = 0):
        super().__init__()
        self.forecast_bias = torch.nn.Parameter(torch.zeros(()))
        self.std_bias = torch.nn.Parameter(torch.zeros(()))
        self.std_shape_offset = std_shape_offset

    def forward(self, values):
        forecast = values * 0.0 + self.forecast_bias
        horizon = values.shape[1] + self.std_shape_offset
        predicted_std = values.new_zeros((values.shape[0], horizon)) + self.std_bias
        return forecast, predicted_std


def _loader():
    targets = torch.tensor([[[1.0, 3.0], [2.0, 6.0]]])
    inputs = torch.zeros_like(targets)
    return DataLoader(TensorDataset(inputs, targets), batch_size=1, shuffle=False)


def _evaluate(model, *, learn_std):
    return eval_with_loader_stable(
        model,
        _loader(),
        criterion=torch.nn.MSELoss(),
        use_amp=False,
        amp_device="cpu",
        amp_dtype=torch.bfloat16,
        learn_std=learn_std,
    )


class NodeStandardDeviationTargetTests(unittest.TestCase):
    def test_uses_population_standard_deviation_across_nodes(self):
        targets = torch.tensor(
            [
                [[1.0, 3.0], [2.0, 6.0]],
                [[5.0, 5.0], [0.0, 4.0]],
            ]
        )

        result = node_standard_deviation_target(targets)

        self.assertEqual(tuple(result.shape), (2, 2))
        torch.testing.assert_close(
            result,
            torch.tensor([[1.0, 2.0], [0.0, 2.0]]),
        )

    def test_rejects_tensors_without_batch_lead_and_node_axes(self):
        with self.assertRaisesRegex(ValueError, "batch, lead_day, node"):
            node_standard_deviation_target(torch.zeros(2, 3))


class StandardDeviationTrainingLossTests(unittest.TestCase):
    def test_evaluation_adds_auxiliary_mse_but_keeps_forecast_metrics_isolated(self):
        metrics = _evaluate(_ForecastAndStdModel(), learn_std=True)

        # Forecast MSE: mean([1, 9, 4, 36]) = 12.5.
        # Population std targets are [1, 2], so std MSE is 2.5.
        self.assertTrue(math.isclose(metrics["prediction_loss"], 12.5))
        self.assertTrue(math.isclose(metrics["std_loss"], 2.5))
        self.assertTrue(math.isclose(metrics["loss"], 15.0))
        self.assertTrue(math.isclose(metrics["mse"], 12.5))

    def test_false_mode_preserves_forecast_only_return_fields_and_loss(self):
        metrics = _evaluate(_ForecastOnlyModel(), learn_std=False)

        self.assertTrue(math.isclose(metrics["loss"], 12.5))
        self.assertTrue(math.isclose(metrics["mse"], 12.5))
        self.assertNotIn("prediction_loss", metrics)
        self.assertNotIn("std_loss", metrics)

    def test_true_mode_requires_one_std_value_per_sample_and_lead(self):
        with self.assertRaisesRegex(ValueError, "predicted_std must have shape"):
            _evaluate(_ForecastAndStdModel(std_shape_offset=1), learn_std=True)

    def test_training_records_auxiliary_losses_only_in_enabled_history_and_summary(self):
        loader = _loader()
        _model, history, summary = train_stable(
            model=_ForecastAndStdModel(),
            train_loader=loader,
            val_loader=loader,
            window_size=2,
            horizon=2,
            hidden_dim=1,
            epochs=1,
            lr=0.0,
            weight_decay=0.0,
            patience=1,
            criterion=torch.nn.MSELoss(),
            use_amp=False,
            learn_std=True,
        )

        self.assertEqual(len(history["train_prediction_loss"]), 1)
        self.assertEqual(len(history["train_std_loss"]), 1)
        self.assertEqual(len(history["val_prediction_loss"]), 1)
        self.assertEqual(len(history["val_std_loss"]), 1)
        self.assertTrue(
            math.isclose(
                history["train_loss"][0],
                history["train_prediction_loss"][0] + history["train_std_loss"][0],
            )
        )
        self.assertTrue(
            math.isclose(
                history["val_loss"][0],
                history["val_prediction_loss"][0] + history["val_std_loss"][0],
            )
        )
        self.assertTrue(summary["learn_std"])
        self.assertEqual(summary["std_loss_function"], "mse")
        self.assertEqual(
            summary["last_val_prediction_loss"], history["val_prediction_loss"][-1]
        )
        self.assertEqual(summary["last_val_std_loss"], history["val_std_loss"][-1])


if __name__ == "__main__":
    unittest.main()
