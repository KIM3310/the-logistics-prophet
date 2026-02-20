from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import requests

DEFAULT_LLM_PROVIDER = "stub"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_TIMEOUT_SEC = 8.0
ALLOWED_PROVIDERS = {"stub", "ollama"}


def _safe_float(value: object, default: float) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed


def _extract_ollama_text(payload: Dict[str, object]) -> str:
    if not isinstance(payload, dict):
        return ""
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    response_text = payload.get("response")
    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()
    return ""


def resolve_incident_llm_settings(
    *,
    llm_provider: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
    ollama_timeout_sec: Optional[float] = None,
) -> Dict[str, object]:
    provider_raw = str(llm_provider or os.getenv("LP_LLM_PROVIDER", DEFAULT_LLM_PROVIDER)).strip().lower()
    provider = provider_raw if provider_raw in ALLOWED_PROVIDERS else DEFAULT_LLM_PROVIDER
    base_url = str(ollama_base_url or os.getenv("LP_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)).strip()
    model = str(ollama_model or os.getenv("LP_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)).strip()
    timeout_sec = _safe_float(
        ollama_timeout_sec if ollama_timeout_sec is not None else os.getenv("LP_OLLAMA_TIMEOUT_SEC"),
        default=DEFAULT_OLLAMA_TIMEOUT_SEC,
    )
    return {
        "provider": provider,
        "ollama_base_url": base_url or DEFAULT_OLLAMA_BASE_URL,
        "ollama_model": model or DEFAULT_OLLAMA_MODEL,
        "ollama_timeout_sec": timeout_sec,
    }


def check_ollama_health(*, base_url: str, timeout_sec: float) -> Dict[str, object]:
    endpoint = f"{str(base_url).rstrip('/')}/api/tags"
    try:
        response = requests.get(endpoint, timeout=max(1.0, float(timeout_sec)))
        response.raise_for_status()
        payload = response.json() if response.content else {}
    except requests.RequestException as exc:
        return {
            "ok": False,
            "endpoint": endpoint,
            "error": str(exc),
            "models": [],
        }
    except ValueError:
        return {
            "ok": False,
            "endpoint": endpoint,
            "error": "invalid_json",
            "models": [],
        }

    models: List[str] = []
    for item in payload.get("models", []) if isinstance(payload, dict) else []:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                models.append(name)
    return {
        "ok": True,
        "endpoint": endpoint,
        "error": "",
        "models": models,
    }


def _build_stub_brief(
    *,
    recommendation: Dict[str, object],
    queue_summary: Dict[str, object],
    ops_health: Dict[str, object],
) -> str:
    severity = str(recommendation.get("severity", "SEV-3"))
    unresolved = int(queue_summary.get("unresolved", 0) or 0)
    critical_open = int(queue_summary.get("critical_open", 0) or 0)
    overdue_eta = int(ops_health.get("overdue_eta", 0) or 0)
    stale_24h = int(ops_health.get("stale_24h", 0) or 0)
    return (
        f"{severity} action: prioritize owner assignment and ETA recovery. "
        f"Current signals unresolved={unresolved}, critical_open={critical_open}, "
        f"overdue_eta={overdue_eta}, stale_24h={stale_24h}."
    )


def _build_ollama_prompt(
    *,
    recommendation: Dict[str, object],
    queue_summary: Dict[str, object],
    ops_health: Dict[str, object],
) -> str:
    compact = {
        "recommendation": {
            "rule_id": recommendation.get("rule_id"),
            "severity": recommendation.get("severity"),
            "title": recommendation.get("title"),
            "description": recommendation.get("description"),
            "suggested_owner": recommendation.get("suggested_owner"),
        },
        "signals": {
            "queue_summary": {
                "unresolved": queue_summary.get("unresolved", 0),
                "critical_open": queue_summary.get("critical_open", 0),
                "open_by_status": queue_summary.get("open_by_status", {}),
            },
            "ops_health": {
                "overdue_eta": ops_health.get("overdue_eta", 0),
                "stale_24h": ops_health.get("stale_24h", 0),
                "critical_unassigned": ops_health.get("critical_unassigned", 0),
                "avg_unresolved_age_hours": ops_health.get("avg_unresolved_age_hours", 0.0),
            },
        },
    }
    return (
        "Write a concise incident operator brief in English. "
        "Return 2-3 short sentences: (1) immediate action, (2) owner/escalation, (3) verification step. "
        "No markdown.\n\n"
        + json.dumps(compact, ensure_ascii=True)
    )


def _generate_ollama_brief(
    *,
    recommendation: Dict[str, object],
    queue_summary: Dict[str, object],
    ops_health: Dict[str, object],
    base_url: str,
    model: str,
    timeout_sec: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a logistics control tower incident copilot. Be precise and operational.",
            },
            {
                "role": "user",
                "content": _build_ollama_prompt(
                    recommendation=recommendation,
                    queue_summary=queue_summary,
                    ops_health=ops_health,
                ),
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 180,
        },
    }
    endpoint = f"{str(base_url).rstrip('/')}/api/chat"
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=max(1.0, float(timeout_sec)),
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"ollama_request_failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("ollama_invalid_json") from exc

    text = _extract_ollama_text(result if isinstance(result, dict) else {})
    if not text:
        raise RuntimeError("ollama_empty_response")
    return text


def enrich_incident_recommendations(
    *,
    recommendations: List[Dict[str, object]],
    queue_summary: Dict[str, object],
    ops_health: Dict[str, object],
    llm_provider: str,
    ollama_base_url: str,
    ollama_model: str,
    ollama_timeout_sec: float,
    fail_on_llm_error: bool = False,
) -> List[Dict[str, object]]:
    provider = str(llm_provider or DEFAULT_LLM_PROVIDER).strip().lower()
    if provider not in ALLOWED_PROVIDERS:
        provider = DEFAULT_LLM_PROVIDER

    enriched: List[Dict[str, object]] = []
    for recommendation in recommendations:
        item = dict(recommendation)
        if provider == "ollama":
            try:
                brief = _generate_ollama_brief(
                    recommendation=item,
                    queue_summary=queue_summary,
                    ops_health=ops_health,
                    base_url=ollama_base_url,
                    model=ollama_model,
                    timeout_sec=ollama_timeout_sec,
                )
                item["operator_brief"] = brief
                item["llm_provider"] = "ollama"
                item["llm_model"] = ollama_model
                item["llm_enriched"] = True
            except RuntimeError as exc:
                if fail_on_llm_error:
                    raise
                item["operator_brief"] = _build_stub_brief(
                    recommendation=item,
                    queue_summary=queue_summary,
                    ops_health=ops_health,
                )
                item["llm_provider"] = "stub_fallback"
                item["llm_model"] = "deterministic-template"
                item["llm_enriched"] = False
                item["llm_error"] = str(exc)
        else:
            item["operator_brief"] = _build_stub_brief(
                recommendation=item,
                queue_summary=queue_summary,
                ops_health=ops_health,
            )
            item["llm_provider"] = "stub"
            item["llm_model"] = "deterministic-template"
            item["llm_enriched"] = True
        enriched.append(item)

    return enriched
