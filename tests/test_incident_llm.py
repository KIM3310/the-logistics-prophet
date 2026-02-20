from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.incident_llm import enrich_incident_recommendations, resolve_incident_llm_settings


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.content = b"{}"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self) -> dict:
        return self._payload


class TestIncidentLLM(unittest.TestCase):
    def setUp(self) -> None:
        self.recommendations = [
            {
                "rule_id": "critical_unassigned",
                "severity": "SEV-1",
                "title": "Critical items have no owner",
                "description": "Assign owner now.",
                "suggested_owner": "control-tower-lead",
            }
        ]
        self.queue_summary = {"unresolved": 42, "critical_open": 3, "open_by_status": {"New": 10}}
        self.ops_health = {
            "overdue_eta": 7,
            "stale_24h": 9,
            "critical_unassigned": 2,
            "avg_unresolved_age_hours": 13.4,
        }

    def test_resolve_settings_defaults_to_stub(self) -> None:
        settings = resolve_incident_llm_settings()
        self.assertEqual(settings["provider"], "stub")
        self.assertTrue(str(settings["ollama_base_url"]).startswith("http://"))
        self.assertTrue(str(settings["ollama_model"]))
        self.assertGreater(float(settings["ollama_timeout_sec"]), 0.0)

    def test_stub_provider_adds_deterministic_brief(self) -> None:
        enriched = enrich_incident_recommendations(
            recommendations=self.recommendations,
            queue_summary=self.queue_summary,
            ops_health=self.ops_health,
            llm_provider="stub",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="llama3.1:8b",
            ollama_timeout_sec=8.0,
        )
        self.assertEqual(len(enriched), 1)
        self.assertIn("operator_brief", enriched[0])
        self.assertEqual(str(enriched[0].get("llm_provider")), "stub")
        self.assertEqual(str(enriched[0].get("llm_model")), "deterministic-template")

    @patch("control_tower.incident_llm.requests.post")
    def test_ollama_provider_enriches_brief(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            {"message": {"role": "assistant", "content": "Escalate now and verify ETA recovery."}}
        )
        enriched = enrich_incident_recommendations(
            recommendations=self.recommendations,
            queue_summary=self.queue_summary,
            ops_health=self.ops_health,
            llm_provider="ollama",
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="llama3.1:8b",
            ollama_timeout_sec=8.0,
        )
        self.assertEqual(str(enriched[0].get("llm_provider")), "ollama")
        self.assertEqual(str(enriched[0].get("llm_model")), "llama3.1:8b")
        self.assertIn("Escalate now", str(enriched[0].get("operator_brief", "")))

    @patch("control_tower.incident_llm.requests.post")
    def test_ollama_failure_falls_back_to_stub(self, mock_post) -> None:
        mock_post.side_effect = requests.RequestException("connection refused")
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
        self.assertEqual(str(enriched[0].get("llm_provider")), "stub_fallback")
        self.assertIn("llm_error", enriched[0])
        self.assertIn("operator_brief", enriched[0])

    @patch("control_tower.incident_llm.requests.post")
    def test_ollama_failure_raises_when_fail_on_error(self, mock_post) -> None:
        mock_post.side_effect = requests.RequestException("connection refused")
        with self.assertRaises(RuntimeError):
            enrich_incident_recommendations(
                recommendations=self.recommendations,
                queue_summary=self.queue_summary,
                ops_health=self.ops_health,
                llm_provider="ollama",
                ollama_base_url="http://127.0.0.1:11434",
                ollama_model="llama3.1:8b",
                ollama_timeout_sec=8.0,
                fail_on_llm_error=True,
            )


if __name__ == "__main__":
    unittest.main()
