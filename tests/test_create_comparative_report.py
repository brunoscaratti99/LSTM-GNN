from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from create_comparative_report import (  # noqa: E402
    MANIFEST_FILENAME,
    _latest_sweep_dir,
    _report_manifest_for_path,
    _resolve_sweep_dir,
    create_comparative_report,
)


class CreateComparativeReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = ROOT / "tests" / f"_create_comparative_report_test_{uuid4().hex}"
        self.sweep_dir = self.temporary / "comparative_hidden_dim_20260826_120000"
        self.run_dir = self.sweep_dir / "run_001__hidden_dim=64"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "hist.pt").write_bytes(b"completed")
        self.manifest = {
            "comparative_parameter": "hidden_dim",
            "runs": [
                {
                    "run_name": self.run_dir.name,
                    "run_dir": self.run_dir.name,
                    "parameters": {"hidden_dim": 64},
                }
            ],
        }
        (self.sweep_dir / MANIFEST_FILENAME).write_text(
            json.dumps(self.manifest),
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.temporary, ignore_errors=True)

    def test_resolves_run_artifacts_to_their_containing_sweep(self):
        self.assertEqual(_resolve_sweep_dir(self.run_dir), self.sweep_dir.resolve())
        self.assertEqual(
            _resolve_sweep_dir(self.run_dir / "hist.pt"),
            self.sweep_dir.resolve(),
        )
        self.assertEqual(
            _resolve_sweep_dir(self.sweep_dir / MANIFEST_FILENAME),
            self.sweep_dir.resolve(),
        )
        self.assertEqual(
            _resolve_sweep_dir(self.sweep_dir / "comparative_analysis"),
            self.sweep_dir.resolve(),
        )

    def test_latest_sweep_accepts_search_root_or_individual_run(self):
        self.assertEqual(_latest_sweep_dir(self.temporary), self.sweep_dir.resolve())
        self.assertEqual(_latest_sweep_dir(self.run_dir), self.sweep_dir.resolve())

    def test_report_accepts_an_individual_run_path(self):
        expected = {
            "report_compare_tex": "comparative_analysis/report_compare.tex",
            "completed_runs": 1,
        }
        with patch(
            "create_comparative_report.save_comparative_outputs",
            return_value=expected,
        ) as save_outputs:
            result = create_comparative_report(self.run_dir)

        self.assertEqual(result, expected)
        save_outputs.assert_called_once_with(self.sweep_dir.resolve(), self.manifest)

    def test_report_aggregates_all_completed_runs_under_search_root(self):
        second_sweep = self.temporary / "comparative_window_size_20260826_121500"
        second_run = second_sweep / "run_001__window_size=30"
        second_run.mkdir(parents=True)
        (second_run / "hist.pt").write_bytes(b"completed")
        second_manifest = {
            "comparative_parameter": "window_size",
            "runs": [
                {
                    "run_name": second_run.name,
                    "run_dir": second_run.name,
                    "parameters": {"window_size": 30},
                }
            ],
        }
        (second_sweep / MANIFEST_FILENAME).write_text(
            json.dumps(second_manifest),
            encoding="utf-8",
        )
        expected = {
            "report_compare_tex": "comparative_analysis/report_compare.tex",
            "completed_runs": 2,
        }

        with patch(
            "create_comparative_report.save_comparative_outputs",
            return_value=expected,
        ) as save_outputs:
            result = create_comparative_report(self.temporary)

        self.assertTrue(result["aggregate_report"])
        self.assertEqual(result["completed_runs"], 2)
        aggregate_root, aggregate_manifest, aggregate = _report_manifest_for_path(self.temporary)
        self.assertEqual(aggregate_root, self.temporary.resolve())
        self.assertTrue(aggregate)
        self.assertEqual(len(aggregate_manifest["runs"]), 2)
        save_outputs.assert_called_once()
        called_root, called_manifest = save_outputs.call_args.args
        self.assertEqual(called_root, self.temporary.resolve())
        self.assertEqual(len(called_manifest["runs"]), 2)
        self.assertEqual(called_manifest["comparative_parameter"], "configuration")


if __name__ == "__main__":
    unittest.main()
