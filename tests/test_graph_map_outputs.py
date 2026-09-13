from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import matplotlib.image as mpimg
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Evaluation.experiment_outputs import save_graph_plot, save_weighted_graph_plot


_BOUNDARY = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-58.0, -34.0],
                        [-49.0, -34.0],
                        [-49.0, -27.0],
                        [-58.0, -27.0],
                        [-58.0, -34.0],
                    ]
                ],
            },
        }
    ],
}


class _AdjacencyModel:
    def __init__(self, adjacency):
        self.adjacency = torch.as_tensor(adjacency, dtype=torch.float32)

    def current_adjacency(self, normalized=True):
        if normalized:
            raise AssertionError("The graph must use the raw W_adj matrix.")
        return self.adjacency


class GraphMapOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = ROOT / "tests" / f"_graph_map_outputs_test_{uuid4().hex}"
        self.temporary.mkdir()
        self.positions = {
            0: (-56.0, -31.0),
            1: (-53.0, -29.5),
            2: (-50.5, -28.0),
        }

    def tearDown(self):
        shutil.rmtree(self.temporary, ignore_errors=True)

    def test_saves_initial_graph_over_supplied_boundary(self):
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

        path = save_graph_plot(
            self.temporary,
            edge_index,
            self.positions,
            boundary_geojson=_BOUNDARY,
        )

        self.assertEqual(path.name, "graph.png")
        self.assertTrue(path.is_file())
        image = mpimg.imread(path)
        self.assertGreater(image.shape[0], 0)
        self.assertGreater(image.shape[1], 0)

    def test_saves_weighted_graph_from_raw_adjacency_with_requested_name(self):
        model = _AdjacencyModel(
            [
                [1.0, 0.2, 0.0],
                [0.2, 1.0, 0.85],
                [0.0, 0.85, 1.0],
            ]
        )

        path = save_weighted_graph_plot(
            self.temporary,
            model,
            self.positions,
            filename="initial_graph.png",
            boundary_geojson=_BOUNDARY,
        )

        self.assertIsNotNone(path)
        self.assertEqual(path.name, "initial_graph.png")
        self.assertTrue(path.is_file())
        image = mpimg.imread(path)
        self.assertGreater(image.shape[0], 0)
        self.assertGreater(image.shape[1], 0)

    def test_weighted_graph_has_no_title_node_labels_or_colorbar_label(self):
        model = _AdjacencyModel(
            [
                [1.0, 0.2, 0.0],
                [0.2, 1.0, 0.85],
                [0.0, 0.85, 1.0],
            ]
        )

        with patch("Evaluation.experiment_outputs.save_figure", autospec=True) as save_figure:
            save_weighted_graph_plot(
                self.temporary,
                model,
                self.positions,
                boundary_geojson=_BOUNDARY,
            )

        figure = save_figure.call_args.args[0]
        map_axis, colorbar_axis = figure.axes
        self.assertEqual(map_axis.get_title(), "")
        self.assertEqual(len(map_axis.texts), 0)
        self.assertEqual(colorbar_axis.get_ylabel(), "")
        self.assertTrue(any(label.get_text() for label in colorbar_axis.get_yticklabels()))

    def test_weighted_graph_rejects_matrix_station_mismatch(self):
        model = _AdjacencyModel(np.eye(2))

        with self.assertRaisesRegex(ValueError, "shape must match"):
            save_weighted_graph_plot(
                self.temporary,
                model,
                self.positions,
                boundary_geojson=_BOUNDARY,
            )


if __name__ == "__main__":
    unittest.main()
