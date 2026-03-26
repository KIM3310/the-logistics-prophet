from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

from .config import DAILY_RISK_QUEUE_PATH, SERVICE_DB_PATH
from .service_store import upsert_queue_rows

logger = logging.getLogger("control_tower.queue_sync")

REQUIRED_QUEUE_FIELDS = (
    "shipment_id",
    "ship_date",
    "order_id",
    "risk_score",
    "risk_band",
    "prediction",
    "key_driver",
    "driver_2",
    "driver_3",
    "recommended_action",
)


def _normalize_queue_row(row: Dict[str, str]) -> Dict[str, object]:
    return {
        "shipment_id": str(row.get("shipment_id", "")).strip(),
        "ship_date": str(row.get("ship_date", "")).strip(),
        "order_id": str(row.get("order_id", "")).strip(),
        "risk_score": float(row.get("risk_score", 0.0) or 0.0),
        "risk_band": str(row.get("risk_band", "Low")).strip() or "Low",
        "prediction": int(float(row.get("prediction", 0) or 0)),
        "key_driver": str(row.get("key_driver", "")).strip(),
        "driver_2": str(row.get("driver_2", "")).strip(),
        "driver_3": str(row.get("driver_3", "")).strip(),
        "recommended_action": str(row.get("recommended_action", "")).strip(),
    }


def load_queue_rows(
    queue_csv_path: Path = DAILY_RISK_QUEUE_PATH,
) -> List[Dict[str, object]]:
    """Load and validate queue rows from CSV."""
    logger.info("Loading queue rows from %s", queue_csv_path)
    if not queue_csv_path.exists():
        logger.error("Queue CSV missing: %s", queue_csv_path)
        raise FileNotFoundError(f"queue csv missing: {queue_csv_path}")

    with queue_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [field for field in REQUIRED_QUEUE_FIELDS if field not in fieldnames]
        if missing:
            logger.error("Queue CSV missing required fields: %s", missing)
            raise ValueError(f"queue csv missing required fields: {', '.join(missing)}")

        rows: List[Dict[str, object]] = []
        for raw_row in reader:
            normalized = _normalize_queue_row(raw_row)
            if not normalized["shipment_id"]:
                continue
            rows.append(normalized)

    if not rows:
        logger.error("Queue CSV contains no valid shipment rows")
        raise ValueError("queue csv contains no valid shipment rows")

    logger.info("Loaded %d queue rows from CSV", len(rows))
    return rows


def sync_queue_from_csv(
    queue_csv_path: Path = DAILY_RISK_QUEUE_PATH,
    service_db_path: Path = SERVICE_DB_PATH,
) -> int:
    """Sync the risk queue from CSV into the service store database."""
    logger.info("Syncing queue from %s to %s", queue_csv_path, service_db_path)
    rows = load_queue_rows(queue_csv_path)
    count = upsert_queue_rows(rows, path=service_db_path)
    logger.info("Queue sync complete: %d rows upserted", count)
    return count
