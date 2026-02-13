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
from control_tower.service_store import fetch_workflow_sla_snapshot


def main() -> None:
    snapshot = fetch_workflow_sla_snapshot(path=SERVICE_DB_PATH, candidate_limit=12)
    print(json.dumps(snapshot, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
