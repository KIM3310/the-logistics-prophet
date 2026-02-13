from __future__ import annotations

import json
import unittest
from pathlib import Path


class TestModelArtifact(unittest.TestCase):
    def test_model_artifact_schema(self) -> None:
        root = Path(__file__).resolve().parents[1]
        artifact_path = root / "data" / "model" / "model_artifact.json"
        self.assertTrue(artifact_path.exists(), "Model artifact missing. Run pipeline first.")

        with artifact_path.open("r", encoding="utf-8") as f:
            artifact = json.load(f)

        required_keys = {
            "schema_version",
            "feature_names",
            "selected_model",
            "selected_threshold",
            "selection_reason",
            "train_metrics",
            "test_metrics",
        }
        self.assertTrue(required_keys.issubset(set(artifact.keys())))

        selected = artifact["selected_model"]
        self.assertIn(selected, {"baseline_logistic", "challenger_calibrated_hgb"})
        self.assertIn(selected, artifact["test_metrics"])
        self.assertGreaterEqual(float(artifact["selected_threshold"]), 0.3)
        self.assertLessEqual(float(artifact["selected_threshold"]), 0.8)


if __name__ == "__main__":
    unittest.main()
