#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import SERVICE_DB_PATH
from control_tower.incident_llm import check_ollama_health, resolve_incident_llm_settings
from control_tower.service_store import derive_incident_recommendations, upsert_incident_from_recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="List or apply incident recommendations.")
    parser.add_argument("--apply", action="store_true", help="Create/update incidents from recommendations.")
    parser.add_argument("--owner", default="auto-ops", help="Incident owner when --apply is used.")
    parser.add_argument("--actor", default="auto-ops", help="Actor identity for audit.")
    parser.add_argument("--actor-role", default="operator", choices=["operator", "admin"], help="Actor role for apply.")
    parser.add_argument(
        "--llm-provider",
        default=None,
        choices=["stub", "ollama"],
        help="Recommendation brief provider (default: LP_LLM_PROVIDER or stub).",
    )
    parser.add_argument("--ollama-base-url", default=None, help="Ollama base URL (default: LP_OLLAMA_BASE_URL).")
    parser.add_argument("--ollama-model", default=None, help="Ollama model (default: LP_OLLAMA_MODEL).")
    parser.add_argument(
        "--ollama-timeout-sec",
        type=float,
        default=None,
        help="Ollama HTTP timeout seconds (default: LP_OLLAMA_TIMEOUT_SEC or 8).",
    )
    parser.add_argument(
        "--ollama-healthz",
        action="store_true",
        help="Probe Ollama /api/tags before recommendation generation.",
    )
    parser.add_argument(
        "--fail-on-llm-error",
        action="store_true",
        help="Exit with non-zero when Ollama call fails instead of stub fallback.",
    )
    args = parser.parse_args()

    runtime = resolve_incident_llm_settings(
        llm_provider=args.llm_provider,
        ollama_base_url=args.ollama_base_url,
        ollama_model=args.ollama_model,
        ollama_timeout_sec=args.ollama_timeout_sec,
    )

    health = None
    if args.ollama_healthz or str(runtime["provider"]) == "ollama":
        health = check_ollama_health(
            base_url=str(runtime["ollama_base_url"]),
            timeout_sec=float(runtime["ollama_timeout_sec"]),
        )
        if not bool(health.get("ok")) and args.fail_on_llm_error and str(runtime["provider"]) == "ollama":
            raise RuntimeError(f"ollama health check failed: {health.get('error', 'unknown_error')}")

    recommendations = derive_incident_recommendations(
        path=SERVICE_DB_PATH,
        max_items=5,
        llm_provider=str(runtime["provider"]),
        ollama_base_url=str(runtime["ollama_base_url"]),
        ollama_model=str(runtime["ollama_model"]),
        ollama_timeout_sec=float(runtime["ollama_timeout_sec"]),
        fail_on_llm_error=bool(args.fail_on_llm_error),
    )
    applied = []
    if args.apply:
        for recommendation in recommendations:
            result = upsert_incident_from_recommendation(
                recommendation=recommendation,
                owner=args.owner,
                actor=args.actor,
                actor_role=args.actor_role,
                path=SERVICE_DB_PATH,
            )
            applied.append(result)

    payload = {
        "count": len(recommendations),
        "llm_runtime": runtime,
        "ollama_health": health,
        "recommendations": recommendations,
        "applied_count": len(applied),
        "applied": applied,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
