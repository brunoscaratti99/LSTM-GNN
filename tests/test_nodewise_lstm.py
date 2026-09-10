from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Evaluation.experiment_outputs import _model_build_state  # noqa: E402
from Models.model import NodewiseLSTM  # noqa: E402
from Training.experiment_runner import build_model  # noqa: E402
from inference import _build_reconstructed_model  # noqa: E402


def _config():
    return SimpleNamespace(
        empty_graph=True,
        learn_std=False,
        learn_self_att=False,
        hidden_dim=6,
        forecast_horizon=2,
        lstm_layers=2,
        dropout=0.0,
    )


def _edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


class NodewiseLSTMTests(unittest.TestCase):
    def test_builder_creates_one_unshared_lstm_per_station(self):
        model = build_model(
            "glstm",
            n_stations=3,
            n_features=4,
            edge_index=_edge_index(),
            config=_config(),
        )

        self.assertIsInstance(model, NodewiseLSTM)
        self.assertEqual(len(model.lstms), 3)
        self.assertEqual(len(model.heads), 3)
        self.assertFalse(any("a_logits" in name for name, _ in model.named_parameters()))
        torch.testing.assert_close(model.current_adjacency(), torch.eye(3))

        output = model(torch.randn(2, 5, 3, 4))
        self.assertEqual(output.shape, (2, 2, 3))

    def test_a_station_input_cannot_change_other_station_forecasts(self):
        torch.manual_seed(17)
        model = NodewiseLSTM(
            N=3,
            in_channels=2,
            hidden_size=6,
            out_channels=2,
            lstm_layers=1,
            dropout=0.0,
        ).eval()
        inputs = torch.randn(2, 4, 3, 2)
        changed_inputs = inputs.clone()
        changed_inputs[:, :, 1, :] += 100.0

        baseline = model(inputs)
        changed = model(changed_inputs)

        torch.testing.assert_close(baseline[:, :, 0], changed[:, :, 0])
        torch.testing.assert_close(baseline[:, :, 2], changed[:, :, 2])
        self.assertFalse(torch.allclose(baseline[:, :, 1], changed[:, :, 1]))

    def test_model_build_contract_reconstructs_nodewise_lstm(self):
        source = NodewiseLSTM(
            N=3,
            in_channels=2,
            hidden_size=6,
            out_channels=2,
            lstm_layers=2,
            dropout=0.0,
        ).eval()
        model_build = _model_build_state(source)
        restored = _build_reconstructed_model(
            _config(),
            n_stations=3,
            n_features=2,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            inference_state={"model_build": model_build},
            state_dict=source.state_dict(),
        ).eval()
        restored.load_state_dict(source.state_dict())

        self.assertEqual(model_build["model_class"], "NodewiseLSTM")
        inputs = torch.randn(2, 4, 3, 2)
        torch.testing.assert_close(source(inputs), restored(inputs))


if __name__ == "__main__":
    unittest.main()
