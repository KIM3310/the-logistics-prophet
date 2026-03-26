from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.screening_agent import (
    BatchScreeningAgent,
    generate_action_items,
    generate_ops_summary,
    prioritize_shipments,
)


def _make_shipment(
    shipment_id: str,
    risk_band: str,
    risk_score: float,
    key_driver: str = "weather_severity",
    recommended_action: str = "",
) -> dict:
    return {
        "shipment_id": shipment_id,
        "risk_band": risk_band,
        "risk_score": risk_score,
        "key_driver": key_driver,
        "recommended_action": recommended_action or f"Action for {risk_band}",
    }


def _sample_batch() -> list:
    return [
        _make_shipment("SHP-010", "Low", 0.30, "distance_km"),
        _make_shipment(
            "SHP-011", "Critical", 0.92, "weather_severity", "Immediate escalation."
        ),
        _make_shipment("SHP-012", "High", 0.75, "carrier_reliability_score"),
        _make_shipment("SHP-013", "Medium", 0.55, "warehouse_load_pct"),
        _make_shipment(
            "SHP-014", "Critical", 0.88, "weather_severity", "Immediate escalation."
        ),
    ]


class TestPrioritizeShipments(unittest.TestCase):
    def test_critical_first(self) -> None:
        result = prioritize_shipments(_sample_batch())
        self.assertEqual(result[0]["risk_band"], "Critical")

    def test_low_last(self) -> None:
        result = prioritize_shipments(_sample_batch())
        self.assertEqual(result[-1]["risk_band"], "Low")

    def test_within_band_sorted_by_score_desc(self) -> None:
        result = prioritize_shipments(_sample_batch())
        critical = [r for r in result if r["risk_band"] == "Critical"]
        self.assertGreaterEqual(
            float(critical[0]["risk_score"]), float(critical[1]["risk_score"])
        )

    def test_empty_batch_returns_empty(self) -> None:
        self.assertEqual(prioritize_shipments([]), [])

    def test_single_item_returned(self) -> None:
        batch = [_make_shipment("SHP-999", "High", 0.70)]
        self.assertEqual(len(prioritize_shipments(batch)), 1)


class TestGenerateOpsSummary(unittest.TestCase):
    def test_empty_batch_returns_clear_message(self) -> None:
        summary = generate_ops_summary([])
        self.assertIn("No shipments", summary)

    def test_summary_contains_total_count(self) -> None:
        summary = generate_ops_summary(_sample_batch())
        self.assertIn("5", summary)

    def test_summary_mentions_critical_urgent(self) -> None:
        summary = generate_ops_summary(_sample_batch())
        self.assertIn("URGENT", summary)

    def test_summary_with_batch_label(self) -> None:
        summary = generate_ops_summary(_sample_batch(), batch_label="2024-01-15")
        self.assertIn("2024-01-15", summary)

    def test_no_critical_no_urgent(self) -> None:
        batch = [
            _make_shipment("SHP-020", "Medium", 0.50),
            _make_shipment("SHP-021", "Low", 0.30),
        ]
        summary = generate_ops_summary(batch)
        self.assertNotIn("URGENT", summary)

    def test_summary_mentions_top_drivers(self) -> None:
        summary = generate_ops_summary(_sample_batch())
        self.assertIn("weather_severity", summary)


class TestGenerateActionItems(unittest.TestCase):
    def test_returns_list(self) -> None:
        items = generate_action_items(_sample_batch())
        self.assertIsInstance(items, list)

    def test_action_items_count_matches_input(self) -> None:
        items = generate_action_items(_sample_batch())
        self.assertEqual(len(items), len(_sample_batch()))

    def test_action_item_has_required_keys(self) -> None:
        items = generate_action_items(_sample_batch())
        required = {
            "shipment_id",
            "risk_band",
            "risk_score",
            "key_driver",
            "action",
            "priority",
        }
        for item in items:
            self.assertTrue(required.issubset(item.keys()))

    def test_critical_items_have_lowest_priority_number(self) -> None:
        items = generate_action_items(_sample_batch())
        critical_items = [i for i in items if i["risk_band"] == "Critical"]
        for item in critical_items:
            self.assertEqual(item["priority"], 0)


class TestBatchScreeningAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = BatchScreeningAgent(use_llm_summary=False)

    def test_screen_returns_dict_with_required_keys(self) -> None:
        result = self.agent.screen(_sample_batch())
        required = {
            "prioritized_shipments",
            "ops_summary",
            "action_items",
            "risk_counts",
            "total_screened",
            "batch_label",
        }
        self.assertTrue(required.issubset(result.keys()))

    def test_screen_total_screened_correct(self) -> None:
        result = self.agent.screen(_sample_batch())
        self.assertEqual(result["total_screened"], 5)

    def test_screen_with_batch_label(self) -> None:
        result = self.agent.screen(_sample_batch(), batch_label="run-42")
        self.assertEqual(result["batch_label"], "run-42")

    def test_screen_empty_batch(self) -> None:
        result = self.agent.screen([])
        self.assertEqual(result["total_screened"], 0)
        self.assertIsInstance(result["ops_summary"], str)

    def test_screen_risk_counts_correct(self) -> None:
        result = self.agent.screen(_sample_batch())
        self.assertEqual(result["risk_counts"]["Critical"], 2)
        self.assertEqual(result["risk_counts"]["High"], 1)

    def test_llm_summary_falls_back_on_api_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        agent = BatchScreeningAgent(use_llm_summary=True)
        agent._client = mock_client
        result = agent.screen(_sample_batch())
        self.assertIsInstance(result["ops_summary"], str)
        self.assertTrue(len(result["ops_summary"]) > 0)

    @patch("control_tower.screening_agent.openai")
    def test_llm_summary_used_when_client_available(self, mock_openai_module) -> None:
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "LLM-generated summary."
        mock_client.chat.completions.create.return_value = mock_resp

        agent = BatchScreeningAgent(api_key="sk-test", use_llm_summary=True)
        agent._client = mock_client
        result = agent.screen(_sample_batch())
        self.assertEqual(result["ops_summary"], "LLM-generated summary.")


if __name__ == "__main__":
    unittest.main()
