from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger("control_tower.data_access")


def fetch_feature_frame(sqlite_path: Path) -> pd.DataFrame:
    """Load the full model_features table into a DataFrame."""
    logger.info("Fetching feature frame from %s", sqlite_path)
    cols = [
        "shipment_id",
        "order_id",
        "ship_date",
        "order_date",
        "customer_id",
        "product_id",
        "warehouse_id",
        "carrier_id",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
        "delay_hours",
    ]
    sql = f"SELECT {', '.join(cols)} FROM model_features ORDER BY ship_date"

    conn = sqlite3.connect(sqlite_path)
    try:
        frame = pd.read_sql_query(sql, conn)
    finally:
        conn.close()

    if frame.empty:
        logger.error("No rows found in model_features at %s", sqlite_path)
        raise ValueError("No rows found in model_features")

    for col in FEATURE_COLUMNS + [TARGET_COLUMN, "delay_hours"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["ship_date"] = pd.to_datetime(frame["ship_date"], errors="coerce")
    logger.info("Loaded %d rows from model_features", len(frame))
    return frame


def fetch_shipment_feature_row(
    sqlite_path: Path, shipment_id: str
) -> Dict[str, object]:
    """Fetch a single shipment row from model_features."""
    shipment_id = str(shipment_id or "").strip()
    if not shipment_id:
        logger.warning("fetch_shipment_feature_row called with empty shipment_id")
        raise ValueError("shipment_id is required")

    logger.debug("Fetching feature row for shipment_id=%s", shipment_id)
    cols = [
        "shipment_id",
        "order_id",
        "ship_date",
        "order_date",
        "customer_id",
        "product_id",
        "warehouse_id",
        "carrier_id",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
        "delay_hours",
    ]
    sql = f"SELECT {', '.join(cols)} FROM model_features WHERE shipment_id = ? LIMIT 1"

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(sql, (shipment_id,)).fetchone()
    finally:
        conn.close()

    if row is None:
        logger.debug("No feature row found for shipment_id=%s", shipment_id)
        return {}

    payload = dict(row)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN, "delay_hours"]:
        try:
            payload[col] = float(payload[col]) if payload.get(col) is not None else None
        except (TypeError, ValueError):
            payload[col] = None
    return payload


def split_frame_by_time(
    frame: pd.DataFrame, test_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/test by time-based ordering."""
    if not 0.0 < test_fraction < 1.0:
        logger.warning("Invalid test_fraction=%.3f, clamping to 0.2", test_fraction)
        test_fraction = 0.2

    sorted_frame = frame.sort_values("ship_date").reset_index(drop=True)
    split_idx = int(len(sorted_frame) * (1 - test_fraction))
    split_idx = max(1, min(split_idx, len(sorted_frame) - 1))
    train = sorted_frame.iloc[:split_idx].copy()
    test = sorted_frame.iloc[split_idx:].copy()
    logger.info(
        "Split frame: train=%d rows, test=%d rows (fraction=%.2f)",
        len(train),
        len(test),
        test_fraction,
    )
    return train, test


def fetch_latest_scoring_batch(sqlite_path: Path) -> Tuple[str, pd.DataFrame]:
    """Fetch the most recent day's shipments for scoring."""
    logger.info("Fetching latest scoring batch from %s", sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    try:
        latest_date_row = conn.execute(
            "SELECT MAX(ship_date) FROM model_features"
        ).fetchone()
        if latest_date_row is None or latest_date_row[0] is None:
            logger.error("No latest ship date found in model_features")
            raise ValueError("No latest ship date found in model_features")
        latest_date = str(latest_date_row[0])

        cols = ["shipment_id", "order_id", "ship_date", *FEATURE_COLUMNS]
        sql = f"SELECT {', '.join(cols)} FROM model_features WHERE ship_date = ? ORDER BY shipment_id"
        frame = pd.read_sql_query(sql, conn, params=(latest_date,))
    finally:
        conn.close()

    if frame.empty:
        logger.error("No rows found for latest scoring date %s", latest_date)
        raise ValueError("No rows found for latest scoring date")

    for col in FEATURE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    logger.info("Scoring batch: date=%s, rows=%d", latest_date, len(frame))
    return latest_date, frame


def fetch_kpi_snapshot(sqlite_path: Path) -> Dict[str, object]:
    """Fetch the latest KPI snapshot including rolling 7-day metrics."""
    logger.info("Fetching KPI snapshot from %s", sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        latest = conn.execute(
            "SELECT MAX(ship_date) AS max_d FROM kpi_daily"
        ).fetchone()["max_d"]
        latest_row = conn.execute(
            "SELECT * FROM kpi_daily WHERE ship_date = ?", (latest,)
        ).fetchone()
        rolling7 = conn.execute(
            """
            SELECT
                AVG(on_time_rate) AS on_time_rate_7d,
                AVG(avg_delay_hours) AS avg_delay_hours_7d,
                SUM(sla_breach_count) AS breaches_7d
            FROM (
                SELECT * FROM kpi_daily ORDER BY ship_date DESC LIMIT 7
            )
            """
        ).fetchone()
        top_causes = conn.execute(
            "SELECT event_type, cnt FROM delay_root_cause ORDER BY cnt DESC LIMIT 5"
        ).fetchall()
    finally:
        conn.close()

    latest_kpi = dict(latest_row) if latest_row else {}
    if latest:
        latest_kpi["ship_date"] = latest

    logger.info("KPI snapshot: latest_date=%s", latest)
    return {
        "latest_date": latest,
        "latest_kpi": latest_kpi,
        "rolling_7d": dict(rolling7) if rolling7 else {},
        "top_delay_causes": [dict(r) for r in top_causes],
    }


def fetch_recent_kpi_series(
    sqlite_path: Path, limit: int = 30
) -> List[Dict[str, object]]:
    """Fetch recent daily KPI rows for trend display."""
    logger.debug("Fetching recent KPI series (limit=%d) from %s", limit, sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM kpi_daily ORDER BY ship_date DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    series = [dict(r) for r in rows]
    series.reverse()
    logger.debug("Returned %d KPI series rows", len(series))
    return series
