#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import DAILY_RISK_QUEUE_PATH, SERVICE_DB_PATH
from control_tower.queue_sync import sync_queue_from_csv


def main() -> None:
    synced = sync_queue_from_csv(
        queue_csv_path=DAILY_RISK_QUEUE_PATH,
        service_db_path=SERVICE_DB_PATH,
    )
    print(
        json.dumps(
            {
                "queue_csv": str(DAILY_RISK_QUEUE_PATH),
                "service_db": str(SERVICE_DB_PATH),
                "synced_rows": int(synced),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
