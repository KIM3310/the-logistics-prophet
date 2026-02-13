#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import DAILY_RISK_QUEUE_PATH, MODEL_BINARY_PATH, SERVICE_DB_PATH, SQLITE_PATH
from control_tower.data_access import fetch_latest_scoring_batch
from control_tower.modeling import load_model_bundle
from control_tower.scoring import score_latest_batch
from control_tower.service_store import fetch_queue_summary


def main() -> None:
    latest_date, batch_frame = fetch_latest_scoring_batch(SQLITE_PATH)
    bundle = load_model_bundle(MODEL_BINARY_PATH)

    ranked = score_latest_batch(
        batch_frame=batch_frame,
        model_bundle=bundle,
        output_path=DAILY_RISK_QUEUE_PATH,
        service_db_path=SERVICE_DB_PATH,
        top_n=50,
    )
    summary = fetch_queue_summary(SERVICE_DB_PATH)

    payload = {
        "latest_date": latest_date,
        "scored_rows": int(len(batch_frame)),
        "output_rows": len(ranked),
        "selected_model": bundle.get("selected_name"),
        "service_queue": summary,
        "output_path": str(DAILY_RISK_QUEUE_PATH),
        "top_risk": ranked[0] if ranked else None,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
