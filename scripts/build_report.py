#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import (
    DAILY_RISK_QUEUE_PATH,
    QUALITY_REPORT_PATH,
    SPARQL_RESULTS_PATH,
    SQLITE_PATH,
    TRAINING_SUMMARY_PATH,
)
from control_tower.data_access import fetch_kpi_snapshot, fetch_recent_kpi_series
from control_tower.ops_output import build_ops_html_report, write_monitoring_payload
from control_tower.service_store import fetch_queue_summary


def _read_risk_queue(path: Path) -> list:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_training_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    kpi_snapshot = fetch_kpi_snapshot(SQLITE_PATH)
    recent_series = fetch_recent_kpi_series(SQLITE_PATH)
    model_summary = _read_training_summary(TRAINING_SUMMARY_PATH)
    ranked_queue = _read_risk_queue(DAILY_RISK_QUEUE_PATH)
    quality_report = _read_optional_json(QUALITY_REPORT_PATH)
    sparql_results = _read_optional_json(SPARQL_RESULTS_PATH)
    service_summary = fetch_queue_summary()

    build_ops_html_report(
        kpi_snapshot=kpi_snapshot,
        recent_series=recent_series,
        model_summary=model_summary,
        ranked_queue=ranked_queue,
        quality_report=quality_report,
    )
    write_monitoring_payload(
        kpi_snapshot=kpi_snapshot,
        training_summary=model_summary,
        ranked_queue=ranked_queue,
        quality_report=quality_report,
        sparql_results=sparql_results,
        service_summary=service_summary,
    )

    payload = {
        "latest_date": kpi_snapshot.get("latest_date"),
        "queue_rows": len(ranked_queue),
        "quality_status": quality_report.get("status", "unknown"),
        "sparql_query_count": len(sparql_results.get("queries", [])),
        "service_unresolved": service_summary.get("unresolved", 0),
        "report_path": "data/output/ops_report.html",
        "monitoring_path": "data/output/monitoring_metrics.json",
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
