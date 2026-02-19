from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import PIPELINE_STATUS_PATH
from control_tower.service_health import build_service_health_report, health_exit_code


class TestServiceHealth(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_pipeline.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Pipeline failed:\n{result.stdout}\n{result.stderr}")

    def test_service_health_report_structure(self) -> None:
        report = build_service_health_report(
            pipeline_status_path=PIPELINE_STATUS_PATH,
            max_pipeline_age_hours=24.0,
            min_model_auc=0.50,
            strict_queue_parity=True,
        )
        self.assertIn("overall_status", report)
        self.assertIn(report.get("overall_status"), {"pass", "warn", "fail"})
        self.assertIn("checks", report)
        self.assertTrue(isinstance(report.get("checks"), list))

        checks = report.get("checks", [])
        ids = {str(item.get("id", "")) for item in checks if isinstance(item, dict)}
        for required in [
            "pipeline_status",
            "pipeline_freshness",
            "quality_gate",
            "model_auc",
            "queue_parity",
            "audit_chain",
        ]:
            self.assertIn(required, ids)

    def test_health_exit_code(self) -> None:
        report = build_service_health_report(
            pipeline_status_path=PIPELINE_STATUS_PATH,
            max_pipeline_age_hours=24.0,
            min_model_auc=0.50,
            strict_queue_parity=True,
        )
        exit_code = health_exit_code(report, warn_as_error=False)
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()

