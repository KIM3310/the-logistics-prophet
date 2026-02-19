from __future__ import annotations

import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import FEATURE_COLUMNS, QUALITY_REPORT_PATH, RAW_DIR, SQLITE_PATH, TARGET_COLUMN


@dataclass
class QualityCheck:
    name: str
    status: str
    detail: str


EXPECTED_RAW_COLUMNS: Dict[str, List[str]] = {
    "customers.csv": ["customer_id", "segment", "region", "join_date"],
    "products.csv": ["product_id", "category", "weight_kg", "price_usd"],
    "warehouses.csv": ["warehouse_id", "region", "capacity_tier", "avg_pick_minutes"],
    "carriers.csv": ["carrier_id", "name", "mode", "reliability_score"],
    "orders.csv": [
        "order_id",
        "customer_id",
        "product_id",
        "warehouse_id",
        "order_date",
        "promised_days",
        "order_value_usd",
        "peak_flag",
        "quantity",
    ],
    "shipments.csv": [
        "shipment_id",
        "order_id",
        "carrier_id",
        "ship_date",
        "delivery_date",
        "distance_km",
        "weather_severity",
        "warehouse_load_pct",
        "transit_days_actual",
        "delivered_on_time",
        "delay_hours",
    ],
    "delay_events.csv": ["event_id", "shipment_id", "event_type", "severity", "event_time"],
}


def _read_header(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return 0.0

    q = np.linspace(0.0, 1.0, bins + 1)
    cuts = np.quantile(expected, q)
    cuts = np.unique(cuts)
    if len(cuts) <= 2:
        return 0.0

    expected_hist, _ = np.histogram(expected, bins=cuts)
    actual_hist, _ = np.histogram(actual, bins=cuts)

    eps = 1e-6
    expected_pct = expected_hist / max(1, expected_hist.sum()) + eps
    actual_pct = actual_hist / max(1, actual_hist.sum()) + eps

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _check_raw_schema(raw_dir: Path) -> List[QualityCheck]:
    checks: List[QualityCheck] = []
    for filename, expected in EXPECTED_RAW_COLUMNS.items():
        path = raw_dir / filename
        if not path.exists():
            checks.append(QualityCheck(name=f"raw_schema:{filename}", status="fail", detail="file missing"))
            continue

        header = _read_header(path)
        if header != expected:
            checks.append(
                QualityCheck(
                    name=f"raw_schema:{filename}",
                    status="fail",
                    detail=f"header mismatch expected={expected} got={header}",
                )
            )
        else:
            checks.append(QualityCheck(name=f"raw_schema:{filename}", status="pass", detail="ok"))
    return checks


def _check_sql_integrity(sqlite_path: Path) -> List[QualityCheck]:
    checks: List[QualityCheck] = []

    conn = sqlite3.connect(sqlite_path)
    try:
        required_tables = [
            "customers",
            "products",
            "warehouses",
            "carriers",
            "orders",
            "shipments",
            "delay_events",
            "model_features",
        ]
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        missing = [t for t in required_tables if t not in existing]
        checks.append(
            QualityCheck(
                name="table_presence",
                status="pass" if not missing else "fail",
                detail="all present" if not missing else f"missing={missing}",
            )
        )

        if missing:
            return checks

        row_count = conn.execute("SELECT COUNT(*) FROM model_features").fetchone()[0]
        checks.append(
            QualityCheck(
                name="row_count:model_features",
                status="pass" if row_count >= 10000 else "fail",
                detail=f"rows={row_count}",
            )
        )

        null_checks = []
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            n = conn.execute(f"SELECT COUNT(*) FROM model_features WHERE {col} IS NULL").fetchone()[0]
            null_checks.append((col, n))
        null_fail = [f"{col}:{n}" for col, n in null_checks if n > 0]
        checks.append(
            QualityCheck(
                name="nulls:model_features",
                status="pass" if not null_fail else "fail",
                detail="none" if not null_fail else ", ".join(null_fail),
            )
        )

        fk_failures = []
        fk_queries = {
            "orders.customer_id->customers": "SELECT COUNT(*) FROM orders o LEFT JOIN customers c ON o.customer_id=c.customer_id WHERE c.customer_id IS NULL",
            "orders.product_id->products": "SELECT COUNT(*) FROM orders o LEFT JOIN products p ON o.product_id=p.product_id WHERE p.product_id IS NULL",
            "orders.warehouse_id->warehouses": "SELECT COUNT(*) FROM orders o LEFT JOIN warehouses w ON o.warehouse_id=w.warehouse_id WHERE w.warehouse_id IS NULL",
            "shipments.order_id->orders": "SELECT COUNT(*) FROM shipments s LEFT JOIN orders o ON s.order_id=o.order_id WHERE o.order_id IS NULL",
            "shipments.carrier_id->carriers": "SELECT COUNT(*) FROM shipments s LEFT JOIN carriers c ON s.carrier_id=c.carrier_id WHERE c.carrier_id IS NULL",
        }
        for name, q in fk_queries.items():
            cnt = conn.execute(q).fetchone()[0]
            if cnt > 0:
                fk_failures.append(f"{name}:{cnt}")

        checks.append(
            QualityCheck(
                name="referential_integrity",
                status="pass" if not fk_failures else "fail",
                detail="ok" if not fk_failures else ", ".join(fk_failures),
            )
        )

        range_rules = {
            "weather_severity": (0.0, 1.0),
            "warehouse_load_pct": (0.0, 1.0),
            "carrier_reliability_score": (0.0, 1.0),
            "distance_km": (1.0, 10000.0),
            "promised_days": (1.0, 30.0),
            "order_value_usd": (0.01, 1_000_000.0),
            "avg_pick_minutes": (1.0, 300.0),
            "product_weight_kg": (0.01, 2000.0),
        }

        range_issues = []
        for col, (lo, hi) in range_rules.items():
            cnt = conn.execute(
                f"SELECT COUNT(*) FROM model_features WHERE {col} < ? OR {col} > ?",
                (lo, hi),
            ).fetchone()[0]
            if cnt > 0:
                range_issues.append(f"{col}:{cnt}")

        label_issues = conn.execute(
            "SELECT COUNT(*) FROM model_features WHERE delivered_late NOT IN (0,1)"
        ).fetchone()[0]
        if label_issues > 0:
            range_issues.append(f"delivered_late:{label_issues}")

        checks.append(
            QualityCheck(
                name="range_checks",
                status="pass" if not range_issues else "fail",
                detail="ok" if not range_issues else ", ".join(range_issues),
            )
        )

        drift_notes = []
        drift_status = "pass"
        for col in ["distance_km", "weather_severity", "warehouse_load_pct", "order_value_usd"]:
            baseline = conn.execute(
                f"""
                WITH ranked AS (
                    SELECT
                        {col} AS value,
                        ROW_NUMBER() OVER (ORDER BY ship_date DESC, shipment_id DESC) AS rn
                    FROM model_features
                    WHERE {col} IS NOT NULL
                )
                SELECT value FROM ranked
                WHERE rn BETWEEN 15 AND 104
                """
            ).fetchall()
            recent = conn.execute(
                f"""
                WITH ranked AS (
                    SELECT
                        {col} AS value,
                        ROW_NUMBER() OVER (ORDER BY ship_date DESC, shipment_id DESC) AS rn
                    FROM model_features
                    WHERE {col} IS NOT NULL
                )
                SELECT value FROM ranked
                WHERE rn BETWEEN 1 AND 14
                """
            ).fetchall()
            base_arr = np.array([r[0] for r in baseline], dtype=float)
            recent_arr = np.array([r[0] for r in recent], dtype=float)
            psi = _psi(base_arr, recent_arr)
            drift_notes.append(f"{col}:{psi:.3f}")
            if psi >= 0.35:
                drift_status = "fail"
            elif psi >= 0.2 and drift_status != "fail":
                drift_status = "warn"

        checks.append(
            QualityCheck(
                name="distribution_drift_psi",
                status=drift_status,
                detail=", ".join(drift_notes),
            )
        )

    finally:
        conn.close()

    return checks


def run_data_quality_checks(
    raw_dir: Path = RAW_DIR,
    sqlite_path: Path = SQLITE_PATH,
    output_path: Path = QUALITY_REPORT_PATH,
) -> Dict[str, object]:
    checks = []
    checks.extend(_check_raw_schema(raw_dir))

    if sqlite_path.exists():
        checks.extend(_check_sql_integrity(sqlite_path))
    else:
        checks.append(QualityCheck(name="sqlite_presence", status="fail", detail=f"missing {sqlite_path}"))

    status_order = {"pass": 0, "warn": 1, "fail": 2}
    worst = "pass"
    for check in checks:
        if status_order.get(check.status, 0) > status_order.get(worst, 0):
            worst = check.status

    summary = {
        "status": worst,
        "total_checks": len(checks),
        "pass_count": sum(1 for c in checks if c.status == "pass"),
        "warn_count": sum(1 for c in checks if c.status == "warn"),
        "fail_count": sum(1 for c in checks if c.status == "fail"),
        "checks": [c.__dict__ for c in checks],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    return summary
