from __future__ import annotations

import csv
import subprocess
import sys
import unittest
from pathlib import Path


class TestPipelineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/run_pipeline.py"],
            cwd=cls.root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Pipeline failed:\n{result.stdout}\n{result.stderr}")

    def test_outputs_exist(self) -> None:
        required = [
            self.root / "data" / "model" / "model_artifact.json",
            self.root / "data" / "model" / "selected_model.pkl",
            self.root / "data" / "output" / "daily_risk_queue.csv",
            self.root / "data" / "output" / "training_summary.json",
            self.root / "data" / "output" / "monitoring_metrics.json",
            self.root / "data" / "output" / "data_quality_report.json",
            self.root / "data" / "output" / "sparql_results.json",
            self.root / "data" / "output" / "shap_global_importance.csv",
            self.root / "data" / "output" / "ops_report.html",
            self.root / "data" / "output" / "pipeline_status.json",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing output: {path}")

    def test_risk_queue_has_rows(self) -> None:
        queue_path = self.root / "data" / "output" / "daily_risk_queue.csv"
        with queue_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 10)


if __name__ == "__main__":
    unittest.main()
