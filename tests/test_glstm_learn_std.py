from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Models.model import GLSTM_v2  # noqa: E402
from Training.experiment_runner import (  # noqa: E402
    ExperimentRunConfig,
    build_model,
    collect_model_predictions,
    unpack_model_output,
)


def _edge_index() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.long,
    )


def _model(*, learn_std: bool | None = None, layers: int = 2) -> GLSTM_v2:
    kwargs = {
        "N": 4,
        "edge_index": _edge_index(),
        "in_channels": 3,
        "hidden_size": 6,
        "out_channels": 2,
        "lstm_layers": layers,
        "learn_adj": True,
        "lock_topology": True,
        "dropout": 0.0,
    }
    if learn_std is not None:
        kwargs["learn_std"] = learn_std
    return GLSTM_v2(**kwargs)


def _config() -> ExperimentRunConfig:
    return ExperimentRunConfig(
        start_date="2020-01-01",
        end_date="2020-12-31",
        state="RS",
        max_stations=None,
        include_precipitation=True,
        include_temperature=False,
        include_specific_humidity=False,
        include_wind=False,
        include_vertical_velocity=False,
        window_size=3,
        forecast_horizon=2,
        train_ratio=0.6,
        val_ratio=0.2,
        normalize_features=False,
        feature_scaler="standard",
        normalize_target=False,
        target_scaler="standard",
        model_type="glstm",
        k_neighbors=2,
        hidden_dim=6,
        lstm_layers=2,
        learn_adj=True,
        lock_topology=True,
        dropout=0.0,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=1,
        adj_lr_factor=1.0,
        max_grad_norm=1.0,
        loss="mse",
        loss_quantiles=(0.5,),
        loss_quantile_weights="auto",
        loss_quantile_max_weight=10.0,
        random_seed=42,
        plot_station_name=None,
        use_daily_cache=False,
    )


class GLSTMLearnStandardDeviationTests(unittest.TestCase):
    def test_false_preserves_tensor_output_modules_and_state_dict(self):
        torch.manual_seed(101)
        default_model = _model()
        torch.manual_seed(101)
        explicit_false_model = _model(learn_std=False)

        self.assertFalse(default_model.learn_std)
        self.assertNotIn("std_head", default_model._modules)
        self.assertFalse(any(key.startswith("std_head.") for key in default_model.state_dict()))
        self.assertEqual(list(default_model.state_dict()), list(explicit_false_model.state_dict()))
        for key, value in default_model.state_dict().items():
            torch.testing.assert_close(value, explicit_false_model.state_dict()[key])

        default_model.eval()
        explicit_false_model.eval()
        inputs = torch.randn(2, 3, 4, 3)
        default_output = default_model(inputs)
        explicit_output = explicit_false_model(inputs)
        self.assertIsInstance(default_output, torch.Tensor)
        self.assertEqual(default_output.shape, (2, 2, 4))
        torch.testing.assert_close(default_output, explicit_output)

    def test_true_returns_forecast_and_nonnegative_population_std_head(self):
        torch.manual_seed(202)
        model = _model(learn_std=True, layers=1)
        model.eval()
        last_hidden = {}
        head_input = {}
        recurrent_hook = model.cell_0.register_forward_hook(
            lambda _module, _args, output: last_hidden.__setitem__("value", output[1].detach())
        )
        head_hook = model.std_head.register_forward_pre_hook(
            lambda _module, args: head_input.__setitem__("value", args[0].detach())
        )
        try:
            forecast, predicted_std = model(torch.randn(3, 4, 4, 3))
        finally:
            recurrent_hook.remove()
            head_hook.remove()

        self.assertEqual(forecast.shape, (3, 2, 4))
        self.assertEqual(predicted_std.shape, (3, 2))
        self.assertTrue(torch.isfinite(predicted_std).all())
        self.assertTrue((predicted_std >= 0.0).all())
        self.assertTrue(any(key.startswith("std_head.") for key in model.state_dict()))

        node_hidden = last_hidden["value"].transpose(1, 2)
        expected_features = torch.cat(
            (
                node_hidden.mean(dim=1),
                node_hidden.std(dim=1, correction=0),
            ),
            dim=-1,
        )
        torch.testing.assert_close(head_input["value"], expected_features)

    def test_true_std_head_receives_gradients_and_is_reset(self):
        torch.manual_seed(303)
        model = _model(learn_std=True)
        forecast, predicted_std = model(torch.randn(2, 3, 4, 3))
        (forecast.square().mean() + predicted_std.square().mean()).backward()

        for parameter in model.std_head.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

        with torch.no_grad():
            for parameter in model.std_head.parameters():
                parameter.fill_(123.0)
        model.reset_parameters()
        self.assertTrue(
            all(not torch.all(parameter == 123.0) for parameter in model.std_head.parameters())
        )


class ModelOutputHelperTests(unittest.TestCase):
    def test_unpack_accepts_legacy_tensor_and_two_tensor_tuple(self):
        forecast = torch.randn(2, 2, 4)
        predicted_std = torch.randn(2, 2)

        unpacked_forecast, unpacked_std = unpack_model_output(forecast)
        self.assertIs(unpacked_forecast, forecast)
        self.assertIsNone(unpacked_std)
        unpacked_forecast, unpacked_std = unpack_model_output((forecast, predicted_std))
        self.assertIs(unpacked_forecast, forecast)
        self.assertIs(unpacked_std, predicted_std)

    def test_unpack_rejects_malformed_outputs(self):
        with self.assertRaises(TypeError):
            unpack_model_output([torch.zeros(1), torch.zeros(1)])
        with self.assertRaises(ValueError):
            unpack_model_output((torch.zeros(1),))
        with self.assertRaises(TypeError):
            unpack_model_output((torch.zeros(1), None))

    def test_builder_propagates_flag_and_rejects_transformer(self):
        config = _config()
        legacy_model = build_model(
            "glstm",
            n_stations=4,
            n_features=3,
            edge_index=_edge_index(),
            config=config,
        )
        self.assertFalse(legacy_model.learn_std)

        std_config = replace(config, learn_std=True)
        std_model = build_model(
            "glstm",
            n_stations=4,
            n_features=3,
            edge_index=_edge_index(),
            config=std_config,
        )
        self.assertTrue(std_model.learn_std)
        with self.assertRaisesRegex(ValueError, "only supported"):
            build_model(
                "transformer",
                n_stations=4,
                n_features=3,
                edge_index=_edge_index(),
                config=replace(std_config, model_type="transformer"),
            )

    def test_collection_keeps_legacy_default_and_optionally_returns_std(self):
        torch.manual_seed(404)
        model = _model(learn_std=True)
        inputs = torch.randn(5, 3, 4, 3)
        targets = torch.randn(5, 2, 4)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)

        forecast_only = collect_model_predictions(model, loader)
        forecast, predicted_std = collect_model_predictions(model, loader, return_std=True)
        self.assertIsInstance(forecast_only, torch.Tensor)
        self.assertEqual(forecast_only.shape, (5, 2, 4))
        self.assertEqual(forecast.shape, (5, 2, 4))
        self.assertEqual(predicted_std.shape, (5, 2))
        torch.testing.assert_close(forecast_only, forecast)

        with self.assertRaisesRegex(ValueError, "return_std=True"):
            collect_model_predictions(_model(learn_std=False), loader, return_std=True)


if __name__ == "__main__":
    unittest.main()
