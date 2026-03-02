from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.incident_llm import enrich_incident_recommendations, resolve_incident_llm_settings


class TestIncidentLLMResilience(unittest.TestCase):
    def setUp(self) -> None:
        self.recommendations = [
            {"rule_id": "r1", "severity": "SEV-1", "title": "A", "description": "A", "suggested_owner": "ops"},
            {"rule_id": "r2", "severity": "SEV-2", "title": "B", "description": "B", "suggested_owner": "ops"},
            {"rule_id": "r3", "severity": "SEV-3", "title": "C", "description": "C", "suggested_owner": "ops"},
        ]
        self.queue_summary = {"unresolved": 9, "critical_open": 2, "open_by_status": {"New": 4}}
        self.ops_health = {
            "overdue_eta": 3,
            "stale_24h": 2,
            "critical_unassigned": 1,
            "avg_unresolved_age_hours": 5.0,
        }

    def test_resolve_settings_is_stable(self) -> None:
        settings = resolve_incident_llm_settings()
        self.assertIn(str(settings.get("provider", "")), {"stub", "ollama"})
        self.assertGreater(float(settings.get("ollama_timeout_sec", 0.0)), 0.0)

    @patch("control_tower.incident_llm._generate_ollama_brief")
    def test_ollama_failure_is_short_circuited_after_first_error(self, mock_generate) -> None:
        mock_generate.side_effect = RuntimeError("connection refused")
        enriched = enrich_incident_recommendations(
            recommendations=self.recommendations,
            queue_summary=self.queue_summary,
            ops_health=self.ops_health,
            llm_provider="ollama",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="llama3.1:8b",
            ollama_timeout_sec=8.0,
            fail_on_llm_error=False,
        )
        self.assertEqual(mock_generate.call_count, 1)
        self.assertEqual(len(enriched), 3)
        self.assertEqual(str(enriched[0].get("llm_provider")), "stub_fallback")
        self.assertEqual(str(enriched[0].get("llm_error")), "connection refused")
        self.assertEqual(str(enriched[1].get("llm_provider")), "stub_fallback")
        self.assertTrue(str(enriched[1].get("llm_error", "")).startswith("ollama_disabled_after_error:"))
        self.assertEqual(str(enriched[2].get("llm_provider")), "stub_fallback")
        self.assertTrue(str(enriched[2].get("llm_error", "")).startswith("ollama_disabled_after_error:"))

    @patch("control_tower.incident_llm._generate_ollama_brief")
    def test_stub_provider_never_calls_ollama_generator(self, mock_generate) -> None:
        enriched = enrich_incident_recommendations(
            recommendations=self.recommendations,
            queue_summary=self.queue_summary,
            ops_health=self.ops_health,
            llm_provider="stub",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="llama3.1:8b",
            ollama_timeout_sec=8.0,
            fail_on_llm_error=False,
        )
        self.assertEqual(mock_generate.call_count, 0)
        self.assertEqual(len(enriched), 3)
        for item in enriched:
            self.assertEqual(str(item.get("llm_provider")), "stub")
            self.assertTrue(bool(item.get("llm_enriched")))
            self.assertIn("operator_brief", item)


if __name__ == "__main__":
    unittest.main()
