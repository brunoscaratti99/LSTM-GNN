from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_benchmark_model as benchmark
from Data.temporal_dataset import chronological_split, create_windowed_splits


class _FakeArimaResult:
    def __init__(self, last_value: float):
        self.last_value = float(last_value)

    def extend(self, values):
        self.last_value = float(np.asarray(values).reshape(-1)[-1])
        return self

    def forecast(self, steps: int):
        return np.repeat(self.last_value, steps)


class _FakeAutoArimaModel:
    order = (1, 0, 0)
    seasonal_order = (0, 0, 0, 0)

    def __init__(self, train):
        self.arima_res_ = _FakeArimaResult(train[-1])

    def aic(self):
        return 1.0


class BenchmarkModelTests(unittest.TestCase):
    def setUp(self):
        times = np.arange("2020-01-01", "2020-04-10", dtype="datetime64[D]")
        values = np.arange(times.size * 2, dtype=np.float32).reshape(times.size, 2)
        self.raw_y = xr.DataArray(
            values,
            dims=("time", "station"),
            coords={"time": times, "station": ["A", "B"]},
        )
        raw_X = xr.DataArray(
            values[..., None],
            dims=("time", "station", "feature"),
            coords={"time": times, "station": ["A", "B"], "feature": ["tp"]},
        )
        self.splits = chronological_split(raw_X, self.raw_y, train_ratio=0.6, val_ratio=0.2)
        self.windowed = create_windowed_splits(self.splits, window_size=5, horizon=3)
        self.config = replace(
            benchmark.BenchmarkConfig(),
            window_size=5,
            forecast_horizon=3,
            show_console_info=False,
        )

    def test_station_persistence_repeats_last_input(self):
        predicted = benchmark.persistence_station_predictions(self.windowed)
        last_input = self.windowed.test_X.sel(feature="tp").isel(lag=-1).values
        self.assertEqual(predicted.shape, self.windowed.test_y.shape)
        np.testing.assert_array_equal(predicted[:, 0, :], last_input)
        np.testing.assert_array_equal(predicted[:, 1, :], last_input)

    def test_auto_arima_rolling_origin_uses_only_available_observations(self):
        fake_auto_arima = lambda train, **kwargs: _FakeAutoArimaModel(train)
        with patch.object(benchmark, "_require_auto_arima", return_value=fake_auto_arima):
            predicted, details = benchmark.auto_arima_predictions(
                self.splits, self.windowed, self.config
            )

        expected = benchmark.persistence_station_predictions(self.windowed)
        np.testing.assert_array_equal(predicted, expected)
        self.assertTrue(all(item["status"] == "fitted" for item in details))

    def test_metrics_have_glstm_compatible_keys(self):
        actual = np.asarray(self.windowed.test_y.values)
        metrics = benchmark.regression_metrics(actual, actual.copy())
        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)
        self.assertEqual(len(metrics["r2_by_step"]), actual.shape[1])


if __name__ == "__main__":
    unittest.main()
