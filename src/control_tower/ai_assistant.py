from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

_openai: Any | None = None
try:  # pragma: no cover - exercised when optional dependency is installed
    import openai as _openai  # type: ignore[import]
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    pass

openai: Any | None = _openai

logger = logging.getLogger("control_tower.ai_assistant")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.3

SYSTEM_PROMPT = (
    "You are a logistics control tower AI assistant specializing in shipment delay analysis. "
    "You receive delay prediction results including risk scores, SHAP feature contributions, and "
    "shipment metadata. Provide concise, actionable operational guidance. "
    "Focus on: root cause analysis, recommended actions, and historical pattern comparisons. "
    "Be precise and operational. No markdown formatting."
)


def _build_context_block(prediction_context: Dict[str, object]) -> str:
    """Build a human-readable context block from prediction results."""
    lines: List[str] = ["Shipment delay prediction context:"]

    shipment_id = prediction_context.get("shipment_id")
    if shipment_id:
        lines.append(f"  Shipment ID: {shipment_id}")

    prediction = prediction_context.get("prediction")
    if prediction is not None:
        label = "DELAYED" if int(prediction) == 1 else "ON TIME"
        lines.append(f"  Prediction: {label}")

    risk_score = prediction_context.get("risk_score")
    if risk_score is not None:
        lines.append(f"  Risk score: {float(risk_score):.4f}")

    risk_band = prediction_context.get("risk_band")
    if risk_band:
        lines.append(f"  Risk band: {risk_band}")

    shap_factors = prediction_context.get("shap_factors")
    if isinstance(shap_factors, dict) and shap_factors:
        lines.append("  SHAP top factors:")
        for feature, value in list(shap_factors.items())[:5]:
            lines.append(f"    {feature}: {value}")
    elif prediction_context.get("key_driver"):
        lines.append(f"  Key driver: {prediction_context['key_driver']}")
        if prediction_context.get("driver_2"):
            lines.append(f"  Driver 2: {prediction_context['driver_2']}")
        if prediction_context.get("driver_3"):
            lines.append(f"  Driver 3: {prediction_context['driver_3']}")

    recommended_action = prediction_context.get("recommended_action")
    if recommended_action:
        lines.append(f"  Recommended action: {recommended_action}")

    return "\n".join(lines)


def _stub_response(question: str, prediction_context: Dict[str, object]) -> str:
    """Return a deterministic stub response when the OpenAI API is unavailable."""
    q_lower = question.lower()
    risk_band = str(prediction_context.get("risk_band", "Unknown"))
    key_driver = str(prediction_context.get("key_driver", "unknown factor"))
    risk_score = prediction_context.get("risk_score")
    score_str = f"{float(risk_score):.2f}" if risk_score is not None else "N/A"

    if any(w in q_lower for w in ("why", "cause", "reason", "delayed")):
        return (
            f"The shipment shows a {risk_band} delay risk (score: {score_str}). "
            f"The primary contributing factor is {key_driver}. "
            "Review the SHAP factors provided for a complete breakdown of contributing signals."
        )
    if any(w in q_lower for w in ("action", "should", "do", "recommend", "fix")):
        recommended = str(
            prediction_context.get(
                "recommended_action", "Monitor and escalate as needed."
            )
        )
        return (
            f"Given the {risk_band} risk level, the recommended action is: {recommended} "
            "Ensure owner assignment and ETA validation are completed promptly."
        )
    if any(w in q_lower for w in ("histor", "pattern", "compare", "similar", "past")):
        return (
            f"Shipments with {risk_band} risk driven by {key_driver} typically experience "
            "delays of 1-3 days. Historical patterns suggest early carrier escalation "
            "reduces impact by approximately 40%. Review recent similar cases in the queue."
        )
    return (
        f"Shipment risk band: {risk_band} (score: {score_str}). "
        f"Primary driver: {key_driver}. "
        "Use specific questions about delay causes, recommended actions, or historical patterns "
        "for more targeted guidance."
    )


class LogisticsAIAssistant:
    """Multi-turn conversational assistant for shipment delay analysis."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._client: Optional[object] = None

        resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if resolved_key:
            try:
                if openai is None:
                    raise ImportError("openai package not installed")
                self._client = openai.OpenAI(api_key=resolved_key)
                logger.info("OpenAI client initialised: model=%s", model)
            except ImportError:
                logger.warning("openai package not installed; using stub responses")
            except Exception as exc:
                logger.warning(
                    "OpenAI client init failed: %s; using stub responses", exc
                )
        else:
            logger.info("OPENAI_API_KEY not set; using stub responses")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session and return its ID."""
        sid = session_id or str(uuid.uuid4())
        self._sessions[sid] = []
        logger.debug("Session created: %s", sid)
        return sid

    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return the message history for a session."""
        return list(self._sessions.get(session_id, []))

    def clear_session(self, session_id: str) -> None:
        """Clear the message history for a session."""
        if session_id in self._sessions:
            self._sessions[session_id] = []
            logger.debug("Session cleared: %s", session_id)

    def list_sessions(self) -> List[str]:
        """Return all active session IDs."""
        return list(self._sessions.keys())

    # ------------------------------------------------------------------
    # Core chat
    # ------------------------------------------------------------------

    def chat(
        self,
        question: str,
        prediction_context: Dict[str, object],
        session_id: Optional[str] = None,
    ) -> str:
        """Answer a question about a shipment delay given prediction context.

        Args:
            question: The operator's question.
            prediction_context: Dict with prediction, risk_score, shap_factors, etc.
            session_id: Optional session ID for multi-turn continuity.

        Returns:
            The assistant's response as a plain string.
        """
        if not question or not question.strip():
            return "Please provide a question about the shipment."

        context_block = _build_context_block(prediction_context)
        user_message = f"{context_block}\n\nQuestion: {question.strip()}"

        # Build message history
        history: List[Dict[str, str]] = []
        if session_id and session_id in self._sessions:
            history = list(self._sessions[session_id])

        history.append({"role": "user", "content": user_message})

        response = self._call_api(history)

        # Persist to session
        if session_id and session_id in self._sessions:
            self._sessions[session_id].append({"role": "user", "content": user_message})
            self._sessions[session_id].append(
                {"role": "assistant", "content": response}
            )

        logger.info(
            "chat: session=%s, question_len=%d, response_len=%d",
            session_id,
            len(question),
            len(response),
        )
        return response

    def _call_api(self, history: List[Dict[str, str]]) -> str:
        """Call OpenAI API or fall back to stub."""
        if self._client is None:
            # Extract prediction context from latest user message for stub
            last_content = history[-1]["content"] if history else ""
            return self._stub_from_message(last_content)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        try:
            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            text = response.choices[0].message.content or ""
            return text.strip()
        except Exception as exc:
            logger.warning("OpenAI API call failed: %s; returning stub response", exc)
            last_content = history[-1]["content"] if history else ""
            return self._stub_from_message(last_content)

    def _stub_from_message(self, message_content: str) -> str:
        """Extract question and context from message content for stub response."""
        # Parse question from message (after "Question: " prefix)
        question = ""
        if "Question: " in message_content:
            question = message_content.split("Question: ", 1)[-1].strip()

        # Build minimal context from message text for stub
        ctx: Dict[str, object] = {}
        for line in message_content.splitlines():
            line = line.strip()
            if line.startswith("Risk score:"):
                try:
                    ctx["risk_score"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Risk band:"):
                ctx["risk_band"] = line.split(":", 1)[1].strip()
            elif line.startswith("Key driver:"):
                ctx["key_driver"] = line.split(":", 1)[1].strip()
            elif line.startswith("Recommended action:"):
                ctx["recommended_action"] = line.split(":", 1)[1].strip()
            elif line.startswith("Prediction:"):
                pred_text = line.split(":", 1)[1].strip()
                ctx["prediction"] = 1 if pred_text == "DELAYED" else 0

        return _stub_response(question, ctx)
