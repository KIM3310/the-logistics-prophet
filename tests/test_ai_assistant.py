from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.ai_assistant import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    LogisticsAIAssistant,
    _build_context_block,
    _stub_response,
)


def _sample_context() -> dict:
    return {
        "shipment_id": "SHP-001",
        "prediction": 1,
        "risk_score": 0.87,
        "risk_band": "Critical",
        "key_driver": "weather_severity",
        "driver_2": "carrier_reliability_score",
        "driver_3": "warehouse_load_pct",
        "recommended_action": "Immediate escalation: Pre-book alternate route.",
    }


def _sample_context_with_shap() -> dict:
    return {
        "shipment_id": "SHP-002",
        "prediction": 1,
        "risk_score": 0.72,
        "risk_band": "High",
        "shap_factors": {
            "weather_severity": 0.42,
            "carrier_reliability_score": -0.18,
            "warehouse_load_pct": 0.15,
        },
        "recommended_action": "Priority action: Escalate carrier.",
    }


class TestBuildContextBlock(unittest.TestCase):
    def test_includes_shipment_id(self) -> None:
        block = _build_context_block(_sample_context())
        self.assertIn("SHP-001", block)

    def test_includes_prediction_label_delayed(self) -> None:
        block = _build_context_block(_sample_context())
        self.assertIn("DELAYED", block)

    def test_includes_prediction_label_on_time(self) -> None:
        ctx = dict(_sample_context())
        ctx["prediction"] = 0
        block = _build_context_block(ctx)
        self.assertIn("ON TIME", block)

    def test_includes_risk_score(self) -> None:
        block = _build_context_block(_sample_context())
        self.assertIn("0.8700", block)

    def test_includes_risk_band(self) -> None:
        block = _build_context_block(_sample_context())
        self.assertIn("Critical", block)

    def test_includes_key_driver(self) -> None:
        block = _build_context_block(_sample_context())
        self.assertIn("weather_severity", block)

    def test_shap_factors_rendered(self) -> None:
        block = _build_context_block(_sample_context_with_shap())
        self.assertIn("SHAP", block)
        self.assertIn("weather_severity", block)

    def test_empty_context_returns_string(self) -> None:
        block = _build_context_block({})
        self.assertIsInstance(block, str)
        self.assertTrue(len(block) > 0)


class TestStubResponse(unittest.TestCase):
    def test_why_question_mentions_risk_band(self) -> None:
        resp = _stub_response("Why is this shipment delayed?", _sample_context())
        self.assertIn("Critical", resp)

    def test_action_question_mentions_recommended_action(self) -> None:
        resp = _stub_response("What actions should we take?", _sample_context())
        self.assertIn("escalation", resp.lower())

    def test_historical_question_mentions_pattern(self) -> None:
        resp = _stub_response("Compare to historical patterns", _sample_context())
        self.assertIn("Historical", resp)

    def test_generic_question_returns_summary(self) -> None:
        resp = _stub_response("Tell me about this shipment", _sample_context())
        self.assertIsInstance(resp, str)
        self.assertTrue(len(resp) > 0)

    def test_score_formatted_in_why_response(self) -> None:
        resp = _stub_response("Why is this delayed?", _sample_context())
        self.assertIn("0.87", resp)


class TestLogisticsAIAssistantNoKey(unittest.TestCase):
    """Tests using stub responses (no API key)."""

    def setUp(self) -> None:
        self.assistant = LogisticsAIAssistant(api_key="")

    def test_chat_returns_string(self) -> None:
        resp = self.assistant.chat("Why is this delayed?", _sample_context())
        self.assertIsInstance(resp, str)
        self.assertTrue(len(resp) > 0)

    def test_chat_empty_question_returns_prompt(self) -> None:
        resp = self.assistant.chat("   ", _sample_context())
        self.assertIn("question", resp.lower())

    def test_session_created(self) -> None:
        sid = self.assistant.create_session()
        self.assertIn(sid, self.assistant.list_sessions())

    def test_session_custom_id(self) -> None:
        sid = self.assistant.create_session("my-session")
        self.assertEqual(sid, "my-session")

    def test_session_history_empty_at_start(self) -> None:
        sid = self.assistant.create_session()
        self.assertEqual(self.assistant.get_session_history(sid), [])

    def test_session_history_populated_after_chat(self) -> None:
        sid = self.assistant.create_session()
        self.assistant.chat("Why is this delayed?", _sample_context(), session_id=sid)
        history = self.assistant.get_session_history(sid)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

    def test_clear_session(self) -> None:
        sid = self.assistant.create_session()
        self.assistant.chat("Why is this delayed?", _sample_context(), session_id=sid)
        self.assistant.clear_session(sid)
        self.assertEqual(self.assistant.get_session_history(sid), [])

    def test_multi_turn_accumulates_history(self) -> None:
        sid = self.assistant.create_session()
        self.assistant.chat("Why is this delayed?", _sample_context(), session_id=sid)
        self.assistant.chat(
            "What actions should we take?", _sample_context(), session_id=sid
        )
        history = self.assistant.get_session_history(sid)
        self.assertEqual(len(history), 4)

    def test_chat_without_session_does_not_store(self) -> None:
        resp = self.assistant.chat("Why is this delayed?", _sample_context())
        self.assertIsInstance(resp, str)
        self.assertEqual(len(self.assistant.list_sessions()), 0)

    def test_unknown_session_id_does_not_crash(self) -> None:
        resp = self.assistant.chat(
            "Why?", _sample_context(), session_id="ghost-session"
        )
        self.assertIsInstance(resp, str)


class TestLogisticsAIAssistantWithMockedAPI(unittest.TestCase):
    """Tests with mocked OpenAI client."""

    def _make_mock_response(self, content: str) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = content
        return mock_resp

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})
    @patch("control_tower.ai_assistant.openai")
    def test_api_called_when_client_available(self, mock_openai_module) -> None:
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "Escalate carrier immediately."
        )

        assistant = LogisticsAIAssistant(api_key="sk-test-key")
        assistant._client = mock_client
        resp = assistant.chat("What should we do?", _sample_context())

        self.assertEqual(resp, "Escalate carrier immediately.")
        mock_client.chat.completions.create.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})
    @patch("control_tower.ai_assistant.openai")
    def test_api_failure_falls_back_to_stub(self, mock_openai_module) -> None:
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API timeout")

        assistant = LogisticsAIAssistant(api_key="sk-test-key")
        assistant._client = mock_client
        resp = assistant.chat("Why is this delayed?", _sample_context())

        self.assertIsInstance(resp, str)
        self.assertTrue(len(resp) > 0)

    def test_default_model_constant(self) -> None:
        self.assertEqual(DEFAULT_MODEL, "gpt-4o-mini")

    def test_default_temperature_constant(self) -> None:
        self.assertAlmostEqual(DEFAULT_TEMPERATURE, 0.3)


if __name__ == "__main__":
    unittest.main()
