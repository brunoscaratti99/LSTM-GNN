from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest
from uuid import uuid4

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Evaluation.rs_animation_maps import RESIDUAL_CMAP, save_rs_precipitation_animation_frames


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


class RSAnimationMapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = ROOT / "tests" / f"_rs_animation_maps_test_{uuid4().hex}"
        self.temporary.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temporary, ignore_errors=True)

    def test_residual_colormap_is_blue_white_red(self):
        negative = np.asarray(RESIDUAL_CMAP(0.0))[:3]
        near_zero = np.asarray(RESIDUAL_CMAP(0.5))[:3]
        positive = np.asarray(RESIDUAL_CMAP(1.0))[:3]

        self.assertGreater(negative[2], negative[0])
        self.assertTrue(np.allclose(near_zero, 1.0))
        self.assertGreater(positive[0], positive[2])

    def test_saves_matching_predict_and_real_frames(self):
        run_dir = self.temporary / "run"
        run_dir.mkdir()
        coordinates = {
            "A": [-31.0, -56.0],
            "B": [-29.5, -53.0],
            "C": [-28.0, -50.5],
        }
        (run_dir / "inference_state.json").write_text(
            json.dumps({"station_coordinates": coordinates}), encoding="utf-8"
        )
        rows = []
        for sample in (0, 1):
            for lead_day in (1, 2):
                for station_index, station in enumerate(coordinates):
                    rows.append(
                        {
                            "sample": sample,
                            "lead_day": lead_day,
                            "target_time": f"2026-01-{sample * 2 + lead_day:02d}",
                            "station": station,
                            "actual_mm": float(station_index + lead_day),
                            "predicted_mm": float(station_index + lead_day + 0.5),
                        }
                    )
        pd.DataFrame(rows).to_csv(run_dir / "test_predictions_by_lead_day.csv", index=False)

        destination = save_rs_precipitation_animation_frames(
            run_dir,
            sample=1,
            boundary_geojson=_BOUNDARY,
            dpi=45,
            figsize=(4.0, 3.8),
        )

        predict_frames = sorted((destination / "Predict").glob("frame_*.png"))
        real_frames = sorted((destination / "Real").glob("frame_*.png"))
        residual_frames = sorted((destination / "Residual").glob("frame_*.png"))
        self.assertEqual([path.name for path in predict_frames], ["frame_001.png", "frame_002.png"])
        self.assertEqual([path.name for path in real_frames], ["frame_001.png", "frame_002.png"])
        self.assertEqual([path.name for path in residual_frames], ["frame_001.png", "frame_002.png"])
        predict_image = mpimg.imread(predict_frames[0])
        real_image = mpimg.imread(real_frames[0])
        residual_image = mpimg.imread(residual_frames[0])
        self.assertEqual(predict_image.shape, real_image.shape)
        self.assertEqual(predict_image.shape, residual_image.shape)

        height, width = predict_image.shape[:2]
        title_region = predict_image[: max(1, int(height * 0.18)), : int(width * 0.88), :3]
        self.assertTrue(np.any(title_region < 0.95))
        self.assertTrue(
            np.allclose(predict_image[height // 2, width // 2, :3], 1.0, atol=1 / 255)
        )

        colorbar_rows = slice(int(height * 0.15), int(height * 0.85))
        colorbar_columns = slice(int(width * 0.90), max(int(width * 0.923), 1))
        predict_colorbar_area = predict_image[colorbar_rows, colorbar_columns, :3]
        real_colorbar_area = real_image[colorbar_rows, colorbar_columns, :3]
        self.assertTrue(np.allclose(predict_colorbar_area, 1.0, atol=1 / 255))
        self.assertTrue(np.any(real_colorbar_area < 0.95))

        manifest = json.loads((destination / "animation_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sample"], 1)
        self.assertEqual(manifest["frame_count"], 2)
        self.assertEqual(manifest["color_scale_mm"], {"min": 0.0, "max": 4.5})
        self.assertEqual(manifest["residual_color_scale_mm"], {"min": -1.0, "max": 1.0})
        self.assertEqual(manifest["residual_formula"], "predicted_mm - actual_mm")
        self.assertEqual(manifest["node_marker"], "circle")
        self.assertFalse(manifest["edges_rendered"])
        self.assertEqual(
            [frame["target_time"] for frame in manifest["frames"]],
            ["2026-01-03", "2026-01-04"],
        )
        beamer_example = (destination / "beamer_animate_example.tex").read_text(encoding="utf-8")
        self.assertIn(r"\documentclass[aspectratio=169]{beamer}", beamer_example)
        self.assertIn(r"\begin{animateinline}", beamer_example)
        self.assertIn("Predict/frame_001.png", beamer_example)
        self.assertIn("Real/frame_002.png", beamer_example)
        self.assertIn("Residual/frame_002.png", beamer_example)
        self.assertIn("label=rsresidual", beamer_example)
        self.assertEqual(beamer_example.count(r"\newframe"), 2)
        self.assertIn(r"\end{document}", beamer_example)

    def test_forecast_without_real_values_is_rejected(self):
        run_dir = self.temporary / "run"
        run_dir.mkdir()
        (run_dir / "inference_state.json").write_text(
            json.dumps({"station_coordinates": {"A": [-30.0, -52.0]}}), encoding="utf-8"
        )
        pd.DataFrame(
            [
                {
                    "sample": 0,
                    "lead_day": 1,
                    "target_time": "2026-01-01",
                    "station": "A",
                    "actual_mm": float("nan"),
                    "predicted_mm": 2.0,
                }
            ]
        ).to_csv(run_dir / "inference_predictions_by_lead_day.csv", index=False)

        with self.assertRaisesRegex(ValueError, "Real frames require historical targets"):
            save_rs_precipitation_animation_frames(
                run_dir,
                boundary_geojson=_BOUNDARY,
                dpi=40,
                figsize=(3.0, 3.0),
            )

    def test_saves_chronological_period_for_one_fixed_lead_day(self):
        run_dir = self.temporary / "run"
        run_dir.mkdir()
        coordinates = {"A": [-31.0, -56.0], "B": [-29.0, -52.0]}
        (run_dir / "inference_state.json").write_text(
            json.dumps({"station_coordinates": coordinates}), encoding="utf-8"
        )
        rows = []
        for sample, first_target_day in ((0, 1), (1, 2), (2, 3)):
            for lead_day in (1, 2):
                for station_index, station in enumerate(coordinates):
                    rows.append(
                        {
                            "sample": sample,
                            "lead_day": lead_day,
                            "target_time": f"2026-02-{first_target_day + lead_day - 1:02d}",
                            "station": station,
                            "actual_mm": sample + lead_day + station_index,
                            "predicted_mm": sample + lead_day + station_index + 0.25,
                        }
                    )
        pd.DataFrame(rows).to_csv(run_dir / "test_predictions_by_lead_day.csv", index=False)

        destination = save_rs_precipitation_animation_frames(
            run_dir,
            start_date="2026-02-02",
            end_date="2026-02-03",
            lead_day=2,
            boundary_geojson=_BOUNDARY,
            dpi=40,
            figsize=(3.0, 3.0),
        )

        self.assertEqual(
            destination.name,
            "lead_day_02__2026-02-02__2026-02-03",
        )
        manifest = json.loads((destination / "animation_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["selection_mode"], "fixed_lead_period")
        self.assertEqual(manifest["lead_day"], 2)
        self.assertEqual(manifest["start_date"], "2026-02-02")
        self.assertEqual(manifest["end_date"], "2026-02-03")
        self.assertEqual([frame["sample"] for frame in manifest["frames"]], [0, 1])
        self.assertEqual(
            [frame["target_time"] for frame in manifest["frames"]],
            ["2026-02-02", "2026-02-03"],
        )
        self.assertEqual(len(list((destination / "Predict").glob("frame_*.png"))), 2)
        self.assertEqual(len(list((destination / "Real").glob("frame_*.png"))), 2)
        self.assertEqual(len(list((destination / "Residual").glob("frame_*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
