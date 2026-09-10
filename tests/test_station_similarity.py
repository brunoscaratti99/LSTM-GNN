from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Graph.graph_related_utils import (  # noqa: E402
    normalize_station_similarity,
    station_similarity_edge_weights,
)
from Models.model import GLSTM_v2  # noqa: E402
from Training.Training_Routines import train_stable  # noqa: E402


def _stations():
    # [latitude, longitude]; station 1 is much closer to 0 than station 2.
    return {
        "A": [0.0, 0.0],
        "B": [0.0, 0.5],
        "C": [0.0, 3.0],
    }


def _edge_index():
    return torch.tensor(
        [[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long
    )


class StationSimilarityTests(unittest.TestCase):
    def test_gaussian_decreases_with_haversine_distance(self):
        weights = station_similarity_edge_weights(
            _stations(),
            _edge_index(),
            station_similarity="gaussian",
            gaussian_sigma_km=100.0,
        )

        self.assertGreater(weights[0].item(), weights[2].item())
        torch.testing.assert_close(weights[0], weights[1])
        torch.testing.assert_close(weights[2], weights[3])

    def test_ones_assigns_one_to_every_selected_edge(self):
        weights = station_similarity_edge_weights(
            _stations(), _edge_index(), station_similarity="ones"
        )

        torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_inverse_distance_uses_raw_reciprocal_distance(self):
        weights = station_similarity_edge_weights(
            _stations(), _edge_index(), station_similarity="inverse_distance"
        )

        self.assertGreater(weights[0].item(), weights[2].item())
        self.assertGreater(weights[0].item() / weights[2].item(), 5.0)

    def test_climatology_correlation_uses_train_series_and_drops_negative_values(self):
        climatology = torch.tensor(
            [
                [1.0, 1.0, 4.0],
                [2.0, 2.0, 3.0],
                [3.0, 3.0, 2.0],
                [4.0, 4.0, 1.0],
            ]
        )
        weights = station_similarity_edge_weights(
            _stations(),
            _edge_index(),
            station_similarity="climatology_correlation",
            climatology=climatology,
        )

        torch.testing.assert_close(weights[:2], torch.ones(2))
        torch.testing.assert_close(weights[2:], torch.full((2,), 1e-4))

    def test_invalid_similarity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported station_similarity"):
            normalize_station_similarity("not_a_similarity")


class SimilarityPriorModelTests(unittest.TestCase):
    def test_explicit_prior_is_preserved_at_zero_residual(self):
        edge_index = _edge_index()
        weights = torch.tensor([0.8, 0.8, 0.2, 0.2])
        model = GLSTM_v2(
            N=3,
            edge_index=edge_index,
            edge_weight=weights,
            in_channels=2,
            hidden_size=4,
            out_channels=1,
            lstm_layers=2,
            learn_adj=True,
            lock_topology=True,
            dropout=0.0,
        )

        raw = model.current_adjacency(normalized=False)
        torch.testing.assert_close(raw.diagonal(), torch.ones(3))
        torch.testing.assert_close(raw[0, 1], torch.tensor(0.8))
        torch.testing.assert_close(raw[0, 2], torch.tensor(0.2))
        torch.testing.assert_close(model.cell_0.a_logits[0, 1], torch.tensor(0.0))
        torch.testing.assert_close(model.adjacency_anchor_loss(), torch.zeros(()))

        with torch.no_grad():
            model.cell_0.a_logits[0, 1] = 0.2
            model.cell_0.a_logits[1, 0] = 0.2
        adjusted = model.current_adjacency(normalized=False)
        torch.testing.assert_close(
            adjusted[0, 1], 0.8 * torch.exp(torch.tensor(0.2)),
        )
        self.assertGreater(model.adjacency_anchor_loss().item(), 0.0)

    def test_training_anchors_residuals_instead_of_decaying_absolute_weights(self):
        edge_index = _edge_index()
        model = GLSTM_v2(
            N=3,
            edge_index=edge_index,
            edge_weight=torch.full((4,), 0.4),
            in_channels=2,
            hidden_size=4,
            out_channels=1,
            learn_adj=True,
            lock_topology=True,
            dropout=0.0,
        )
        with torch.no_grad():
            model.cell_0.a_logits[model.cell_0.edge_mask.bool()] = 0.2
        expected_anchor = model.adjacency_anchor_loss().item()
        loader = DataLoader(
            TensorDataset(torch.randn(1, 2, 3, 2), torch.randn(1, 1, 3)),
            batch_size=1,
            shuffle=False,
        )

        _trained, history, summary = train_stable(
            model=model,
            train_loader=loader,
            val_loader=loader,
            window_size=2,
            horizon=1,
            hidden_dim=4,
            epochs=1,
            lr=0.0,
            weight_decay=0.25,
            patience=1,
            criterion=torch.nn.MSELoss(),
            use_amp=False,
        )

        self.assertAlmostEqual(
            history["train_adjacency_anchor_loss"][0], expected_anchor, places=6
        )
        self.assertEqual(summary["adjacency_anchor_strength"], 0.25)


if __name__ == "__main__":
    unittest.main()
