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
from control_tower.service_store import derive_incident_recommendations, upsert_incident_from_recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="List or apply incident recommendations.")
    parser.add_argument("--apply", action="store_true", help="Create/update incidents from recommendations.")
    parser.add_argument("--owner", default="auto-ops", help="Incident owner when --apply is used.")
    parser.add_argument("--actor", default="auto-ops", help="Actor identity for audit.")
    parser.add_argument("--actor-role", default="operator", choices=["operator", "admin"], help="Actor role for apply.")
    args = parser.parse_args()

    recommendations = derive_incident_recommendations(path=SERVICE_DB_PATH, max_items=5)
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
        "recommendations": recommendations,
        "applied_count": len(applied),
        "applied": applied,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
