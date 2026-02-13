from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import DAILY_RISK_QUEUE_PATH, FEATURE_COLUMNS, SERVICE_DB_PATH
from .modeling import feature_contributions, predict_scores
from .service_store import upsert_queue_rows


def _risk_band(score: float) -> str:
    if score >= 0.82:
        return "Critical"
    if score >= 0.68:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def _recommended_action(score: float, driver: str) -> str:
    actions = {
        "weather_severity": "Pre-book alternate route and push proactive ETA update.",
        "warehouse_load_pct": "Shift picking capacity and move overflow to nearby warehouse.",
        "carrier_reliability_score": "Escalate carrier and reserve backup carrier slot.",
        "distance_km": "Prioritize line-haul planning and insert checkpoint monitoring.",
    }
    generic = "Monitor watchlist and validate upstream signals before dispatch."

    if score >= 0.82:
        prefix = "Immediate escalation: "
    elif score >= 0.68:
        prefix = "Priority action: "
    elif score >= 0.45:
        prefix = "Preventive action: "
    else:
        prefix = "No escalation: "

    return prefix + actions.get(driver, generic)


def _top_drivers(feature_names: List[str], contrib: np.ndarray, top_k: int = 3) -> List[str]:
    idxs = np.argsort(np.abs(contrib))[::-1][:top_k]
    return [feature_names[int(i)] for i in idxs]


def score_latest_batch(
    batch_frame: pd.DataFrame,
    model_bundle: Dict[str, object],
    output_path: Path = DAILY_RISK_QUEUE_PATH,
    service_db_path: Path = SERVICE_DB_PATH,
    top_n: int = 50,
) -> List[Dict[str, object]]:
    x = batch_frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    probs = predict_scores(model_bundle, x)
    threshold = float(model_bundle["selected_threshold"])

    contrib = feature_contributions(model_bundle, x)

    ranked: List[Dict[str, object]] = []
    for idx in range(len(batch_frame)):
        row = batch_frame.iloc[idx]
        score = float(probs[idx])

        drivers = _top_drivers(FEATURE_COLUMNS, contrib[idx], top_k=3)
        key_driver = drivers[0] if drivers else FEATURE_COLUMNS[0]

        ranked.append(
            {
                "ship_date": str(row["ship_date"]),
                "shipment_id": str(row["shipment_id"]),
                "order_id": str(row["order_id"]),
                "risk_score": round(score, 4),
                "risk_band": _risk_band(score),
                "prediction": int(score >= threshold),
                "key_driver": key_driver,
                "driver_2": drivers[1] if len(drivers) > 1 else "",
                "driver_3": drivers[2] if len(drivers) > 2 else "",
                "recommended_action": _recommended_action(score, key_driver),
            }
        )

    ranked.sort(key=lambda r: r["risk_score"], reverse=True)
    selected = ranked[:top_n]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ship_date",
                "shipment_id",
                "order_id",
                "risk_score",
                "risk_band",
                "prediction",
                "key_driver",
                "driver_2",
                "driver_3",
                "recommended_action",
            ],
        )
        writer.writeheader()
        writer.writerows(selected)

    upsert_queue_rows(selected, path=service_db_path)
    return selected
