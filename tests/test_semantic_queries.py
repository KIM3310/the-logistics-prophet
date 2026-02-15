from __future__ import annotations

import subprocess
import sys
import unittest
import csv
from pathlib import Path


class TestSemanticQueries(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        src = cls.root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        from control_tower.config import RDF_INSTANCE_PATH

        if RDF_INSTANCE_PATH.exists():
            return

        result = subprocess.run(
            [sys.executable, "scripts/run_pipeline.py"],
            cwd=cls.root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Pipeline failed:\n{result.stdout}\n{result.stderr}")

    def test_query_shipment_evidence_returns_payload(self) -> None:
        src = self.root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        from control_tower.config import RDF_INSTANCE_PATH
        from control_tower.semantic_queries import load_instance_graph, query_shipment_evidence

        self.assertTrue(RDF_INSTANCE_PATH.exists(), f"Missing instance graph: {RDF_INSTANCE_PATH}")
        graph = load_instance_graph(RDF_INSTANCE_PATH)

        from control_tower.config import SHAP_LOCAL_PATH

        shipment_id = ""
        if SHAP_LOCAL_PATH.exists():
            with SHAP_LOCAL_PATH.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                first = next(reader, None)
                if first:
                    shipment_id = str(first.get("shipment_id", "") or "").strip()
        if not shipment_id:
            shipment_id = "S0000001"

        evidence = query_shipment_evidence(graph, shipment_id).as_dict()

        self.assertIn("shipment", evidence)
        self.assertIn("delay_events", evidence)
        self.assertIsInstance(evidence["delay_events"], list)


if __name__ == "__main__":
    unittest.main()
