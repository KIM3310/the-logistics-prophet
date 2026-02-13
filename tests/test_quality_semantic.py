from __future__ import annotations

import json
import unittest
from pathlib import Path


class TestQualityAndSemantic(unittest.TestCase):
    def test_quality_gate_not_failed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        quality_path = root / "data" / "output" / "data_quality_report.json"
        self.assertTrue(quality_path.exists(), "Quality report missing. Run pipeline first.")

        with quality_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertNotEqual(report.get("status"), "fail", "Quality gate failed.")

    def test_sparql_queries_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sparql_path = root / "data" / "output" / "sparql_results.json"
        self.assertTrue(sparql_path.exists(), "SPARQL results missing. Run pipeline first.")

        with sparql_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.assertGreaterEqual(len(payload.get("queries", [])), 3)


if __name__ == "__main__":
    unittest.main()
