from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import (
    DAILY_RISK_QUEUE_PATH,
    QUALITY_REPORT_PATH,
    SERVICE_DB_PATH,
    TRAINING_SUMMARY_PATH,
)
from .service_store import verify_audit_chain

STATUS_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _parse_iso_dt(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _worst_status(statuses: List[str]) -> str:
    worst = "pass"
    for status in statuses:
        if STATUS_ORDER.get(status, 0) > STATUS_ORDER.get(worst, 0):
            worst = status
    return worst


def _queue_csv_summary(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {"row_count": 0, "unique_shipments": 0, "duplicate_rows": 0}

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    shipment_ids = [str(row.get("shipment_id", "")).strip() for row in rows if str(row.get("shipment_id", "")).strip()]
    unique_shipments = len(set(shipment_ids))
    return {
        "row_count": len(rows),
        "unique_shipments": unique_shipments,
        "duplicate_rows": max(0, len(rows) - unique_shipments),
    }


def _service_queue_count(path: Path) -> int:
    if not path.exists():
        return 0
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM service_queue").fetchone()[0])
    finally:
        conn.close()


def build_service_health_report(
    *,
    pipeline_status_path: Path,
    quality_report_path: Path = QUALITY_REPORT_PATH,
    training_summary_path: Path = TRAINING_SUMMARY_PATH,
    queue_csv_path: Path = DAILY_RISK_QUEUE_PATH,
    service_db_path: Path = SERVICE_DB_PATH,
    max_pipeline_age_hours: float = 24.0,
    min_model_auc: float = 0.72,
    strict_queue_parity: bool = True,
    now_utc: datetime | None = None,
) -> Dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    checks: List[Dict[str, Any]] = []

    pipeline = _read_json(pipeline_status_path)
    pipeline_state = str(pipeline.get("status", "")).strip().lower()
    if not pipeline:
        checks.append({"id": "pipeline_status", "status": "fail", "detail": "pipeline status file missing"})
        checks.append({"id": "pipeline_freshness", "status": "fail", "detail": "pipeline status unavailable"})
    else:
        if pipeline_state == "ok":
            checks.append({"id": "pipeline_status", "status": "pass", "detail": "pipeline status is ok"})
        else:
            checks.append(
                {
                    "id": "pipeline_status",
                    "status": "fail",
                    "detail": f"pipeline status is {pipeline_state or 'unknown'}",
                }
            )

        finished_dt = _parse_iso_dt(pipeline.get("finished_at_utc"))
        if finished_dt is None:
            checks.append({"id": "pipeline_freshness", "status": "fail", "detail": "finished_at_utc missing/invalid"})
        else:
            age_hours = max(0.0, (now - finished_dt).total_seconds() / 3600.0)
            if age_hours <= float(max_pipeline_age_hours):
                checks.append(
                    {
                        "id": "pipeline_freshness",
                        "status": "pass",
                        "detail": f"pipeline age={age_hours:.2f}h (max={max_pipeline_age_hours:.2f}h)",
                    }
                )
            else:
                checks.append(
                    {
                        "id": "pipeline_freshness",
                        "status": "warn",
                        "detail": f"pipeline age={age_hours:.2f}h exceeds max={max_pipeline_age_hours:.2f}h",
                    }
                )

    quality = _read_json(quality_report_path)
    quality_status = str(quality.get("status", "")).strip().lower()
    if not quality:
        checks.append({"id": "quality_gate", "status": "fail", "detail": "quality report missing"})
    elif quality_status == "fail":
        checks.append(
            {
                "id": "quality_gate",
                "status": "fail",
                "detail": f"quality failed (fails={quality.get('fail_count', 0)}, warns={quality.get('warn_count', 0)})",
            }
        )
    elif quality_status == "warn":
        checks.append(
            {
                "id": "quality_gate",
                "status": "warn",
                "detail": f"quality warn (fails={quality.get('fail_count', 0)}, warns={quality.get('warn_count', 0)})",
            }
        )
    else:
        checks.append(
            {
                "id": "quality_gate",
                "status": "pass",
                "detail": f"quality pass (fails={quality.get('fail_count', 0)}, warns={quality.get('warn_count', 0)})",
            }
        )

    training = _read_json(training_summary_path)
    model_auc = _safe_float((training.get("test_metrics", {}) or {}).get("auc"), default=-1.0)
    if not training or model_auc < 0:
        checks.append({"id": "model_auc", "status": "fail", "detail": "training summary missing/invalid auc"})
    elif model_auc < float(min_model_auc):
        checks.append(
            {
                "id": "model_auc",
                "status": "warn",
                "detail": f"model auc={model_auc:.4f} below target={min_model_auc:.4f}",
            }
        )
    else:
        checks.append(
            {
                "id": "model_auc",
                "status": "pass",
                "detail": f"model auc={model_auc:.4f} meets target={min_model_auc:.4f}",
            }
        )

    queue_csv = _queue_csv_summary(queue_csv_path)
    db_queue_count = _service_queue_count(service_db_path)
    duplicate_rows = int(queue_csv.get("duplicate_rows", 0))
    parity_mismatch = abs(int(queue_csv.get("row_count", 0)) - db_queue_count)
    if not queue_csv_path.exists():
        checks.append({"id": "queue_parity", "status": "fail", "detail": "daily_risk_queue.csv missing"})
    elif not service_db_path.exists():
        checks.append({"id": "queue_parity", "status": "fail", "detail": "service_store.db missing"})
    else:
        if duplicate_rows > 0:
            status = "fail" if strict_queue_parity else "warn"
            checks.append(
                {
                    "id": "queue_duplicates",
                    "status": status,
                    "detail": f"csv duplicates detected: {duplicate_rows}",
                }
            )
        else:
            checks.append({"id": "queue_duplicates", "status": "pass", "detail": "no duplicate shipment rows"})

        if parity_mismatch == 0:
            checks.append(
                {
                    "id": "queue_parity",
                    "status": "pass",
                    "detail": f"csv rows={queue_csv.get('row_count', 0)} matches db rows={db_queue_count}",
                }
            )
        else:
            status = "fail" if strict_queue_parity else "warn"
            checks.append(
                {
                    "id": "queue_parity",
                    "status": status,
                    "detail": f"csv rows={queue_csv.get('row_count', 0)} db rows={db_queue_count} mismatch={parity_mismatch}",
                }
            )

    if not service_db_path.exists():
        checks.append({"id": "audit_chain", "status": "fail", "detail": "service db missing"})
    else:
        audit = verify_audit_chain(path=service_db_path, limit=10000)
        if bool(audit.get("valid")):
            checks.append(
                {
                    "id": "audit_chain",
                    "status": "pass",
                    "detail": f"audit valid (checked={_safe_int(audit.get('checked', 0))})",
                }
            )
        else:
            checks.append(
                {
                    "id": "audit_chain",
                    "status": "fail",
                    "detail": f"audit invalid reason={audit.get('reason', 'unknown')}",
                }
            )

    statuses = [str(item.get("status", "pass")) for item in checks]
    overall_status = _worst_status(statuses)

    return {
        "generated_at_utc": now.isoformat(),
        "overall_status": overall_status,
        "thresholds": {
            "max_pipeline_age_hours": float(max_pipeline_age_hours),
            "min_model_auc": float(min_model_auc),
            "strict_queue_parity": bool(strict_queue_parity),
        },
        "summary": {
            "total_checks": len(checks),
            "pass_count": sum(1 for s in statuses if s == "pass"),
            "warn_count": sum(1 for s in statuses if s == "warn"),
            "fail_count": sum(1 for s in statuses if s == "fail"),
        },
        "checks": checks,
    }


def health_exit_code(report: Dict[str, Any], warn_as_error: bool = False) -> int:
    overall = str(report.get("overall_status", "pass")).strip().lower()
    if overall == "fail":
        return 1
    if overall == "warn" and warn_as_error:
        return 2
    return 0

