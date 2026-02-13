#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = ROOT / "data" / "output" / "monitoring_metrics.json"
OUTPUT_FILE = ROOT / "data" / "output" / "datadog_series_payload.json"


def _metric(name: str, value: float, now: int) -> dict:
    return {
        "metric": name,
        "type": 0,
        "points": [[now, float(value)]],
        "tags": ["env:portfolio", "service:semantic-control-tower", "team:data-platform"],
    }


def main() -> None:
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_FILE}")

    with METRICS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    now = int(time.time())

    latest = data.get("kpi_latest", {})
    model = data.get("model_test_metrics", {})
    queue = data.get("risk_queue_metrics", {})
    quality = data.get("quality_gate", {})
    semantic = data.get("semantic_layer", {})
    service_queue = data.get("service_queue", {})

    series = [
        _metric("semantic_control_tower.kpi.on_time_rate", latest.get("on_time_rate", 0.0), now),
        _metric("semantic_control_tower.kpi.sla_breach_count", latest.get("sla_breach_count", 0.0), now),
        _metric("semantic_control_tower.kpi.avg_delay_hours", latest.get("avg_delay_hours", 0.0), now),
        _metric("semantic_control_tower.model.auc", model.get("auc", 0.0), now),
        _metric("semantic_control_tower.model.f1", model.get("f1", 0.0), now),
        _metric("semantic_control_tower.risk_queue.critical_count", queue.get("critical_count", 0.0), now),
        _metric("semantic_control_tower.risk_queue.high_count", queue.get("high_count", 0.0), now),
        _metric("semantic_control_tower.quality.fail_count", quality.get("fail_count", 0.0), now),
        _metric("semantic_control_tower.quality.warn_count", quality.get("warn_count", 0.0), now),
        _metric("semantic_control_tower.semantic.query_count", semantic.get("query_count", 0.0), now),
        _metric("semantic_control_tower.service.unresolved", service_queue.get("unresolved", 0.0), now),
        _metric("semantic_control_tower.service.critical_open", service_queue.get("critical_open", 0.0), now),
        _metric("semantic_control_tower.pipeline.success", 1.0, now),
    ]

    payload = {"series": series}
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    print(json.dumps({"output": str(OUTPUT_FILE), "series_count": len(series)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
