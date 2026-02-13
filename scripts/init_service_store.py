#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import SERVICE_DB_PATH
from control_tower.service_store import init_service_store, list_pipeline_runs, list_users


def main() -> None:
    init_service_store(SERVICE_DB_PATH)
    runs = list_pipeline_runs(SERVICE_DB_PATH, limit=3)
    users = list_users(SERVICE_DB_PATH, include_inactive=True)
    payload = {
        "service_db": str(SERVICE_DB_PATH),
        "users": users,
        "existing_runs": len(runs),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
