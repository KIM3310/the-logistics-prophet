from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .config import MONITORING_METRICS_PATH, OPS_REPORT_PATH, TRAINING_SUMMARY_PATH

DRIVER_LABELS = {
    "distance_km": "Distance",
    "weather_severity": "Weather",
    "warehouse_load_pct": "Warehouse Load",
    "carrier_reliability_score": "Carrier Risk",
    "promised_days": "Tight Promise",
    "order_value_usd": "High Value",
    "peak_flag": "Peak Day",
    "avg_pick_minutes": "Slow Pick",
    "product_weight_kg": "Heavy Item",
}


def _driver_label(raw: object) -> str:
    value = str(raw or "").strip()
    if not value or value.lower() == "nan":
        return ""
    return DRIVER_LABELS.get(value, value.replace("_", " ").title())


def write_training_summary(
    model_artifact: Dict[str, object],
    output_path: Path = TRAINING_SUMMARY_PATH,
) -> Dict[str, object]:
    selected = str(model_artifact.get("selected_model", ""))
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected,
        "selected_threshold": model_artifact.get("selected_threshold"),
        "selection_reason": model_artifact.get("selection_reason"),
        "train_metrics": model_artifact.get("train_metrics", {}).get(selected, {}),
        "test_metrics": model_artifact.get("test_metrics", {}).get(selected, {}),
        "comparison": {
            "train_metrics": model_artifact.get("train_metrics", {}),
            "test_metrics": model_artifact.get("test_metrics", {}),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    return summary


def write_monitoring_payload(
    kpi_snapshot: Dict[str, object],
    training_summary: Dict[str, object],
    ranked_queue: List[Dict[str, object]],
    quality_report: Dict[str, object] | None = None,
    sparql_results: Dict[str, object] | None = None,
    service_summary: Dict[str, object] | None = None,
    output_path: Path = MONITORING_METRICS_PATH,
) -> Dict[str, object]:
    critical_count = sum(1 for r in ranked_queue if r.get("risk_band") == "Critical")
    high_count = sum(1 for r in ranked_queue if r.get("risk_band") == "High")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "service": "semantic-control-tower",
        "env": "portfolio",
        "kpi_latest": kpi_snapshot.get("latest_kpi", {}),
        "kpi_rolling_7d": kpi_snapshot.get("rolling_7d", {}),
        "top_delay_causes": kpi_snapshot.get("top_delay_causes", []),
        "selected_model": training_summary.get("selected_model"),
        "model_test_metrics": training_summary.get("test_metrics", {}),
        "risk_queue_metrics": {
            "top_n": len(ranked_queue),
            "critical_count": critical_count,
            "high_count": high_count,
            "critical_ratio": round(critical_count / len(ranked_queue), 4) if ranked_queue else 0.0,
        },
        "quality_gate": {
            "status": (quality_report or {}).get("status", "unknown"),
            "fail_count": (quality_report or {}).get("fail_count", 0),
            "warn_count": (quality_report or {}).get("warn_count", 0),
        },
        "semantic_layer": {
            "query_count": len((sparql_results or {}).get("queries", [])),
        },
        "service_queue": {
            "critical_open": (service_summary or {}).get("critical_open", 0),
            "unresolved": (service_summary or {}).get("unresolved", 0),
            "status_breakdown": (service_summary or {}).get("status_breakdown", []),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return payload


def build_ops_html_report(
    kpi_snapshot: Dict[str, object],
    recent_series: List[Dict[str, object]],
    model_summary: Dict[str, object],
    ranked_queue: List[Dict[str, object]],
    quality_report: Dict[str, object] | None = None,
    output_path: Path = OPS_REPORT_PATH,
) -> None:
    latest = kpi_snapshot.get("latest_kpi", {})
    rolling = kpi_snapshot.get("rolling_7d", {})
    causes = kpi_snapshot.get("top_delay_causes", [])
    selected_model = model_summary.get("selected_model", "n/a")

    queue_rows = "\n".join(
        [
            (
                "<tr>"
                f"<td>{r.get('shipment_id','')}</td>"
                f"<td>{r.get('order_id','')}</td>"
                f"<td>{r.get('risk_score','')}</td>"
                f"<td>{r.get('risk_band','')}</td>"
                f"<td>{_driver_label(r.get('key_driver',''))}</td>"
                "</tr>"
            )
            for r in ranked_queue[:25]
        ]
    )

    cause_rows = "\n".join([f"<tr><td>{c.get('event_type','')}</td><td>{c.get('cnt','')}</td></tr>" for c in causes])

    series_rows = "\n".join(
        [
            (
                "<tr>"
                f"<td>{r.get('ship_date','')}</td>"
                f"<td>{r.get('total_shipments','')}</td>"
                f"<td>{r.get('on_time_rate','')}</td>"
                f"<td>{r.get('avg_delay_hours','')}</td>"
                f"<td>{r.get('sla_breach_count','')}</td>"
                "</tr>"
            )
            for r in recent_series[-14:]
        ]
    )

    quality_status = (quality_report or {}).get("status", "unknown")

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>The Logistics Prophet Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 28px; color: #1c1c1c; background:#f7f7f5; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 12px; margin-bottom: 20px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; background: #fff; }}
    .metric {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; margin-bottom: 20px; background: #fff; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
    th {{ background: #f2f2f2; text-align: left; }}
    .small {{ color: #666; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>The Logistics Prophet</h1>
  <p class=\"small\">Generated at {datetime.now(timezone.utc).isoformat()} (UTC)</p>

  <div class=\"grid\">
    <div class=\"card\"><div>Latest On-Time %</div><div class=\"metric\">{latest.get('on_time_rate', 'NA')}</div></div>
    <div class=\"card\"><div>Latest Late Deliveries</div><div class=\"metric\">{latest.get('sla_breach_count', 'NA')}</div></div>
    <div class=\"card\"><div>7-Day Avg On-Time</div><div class=\"metric\">{round(float(rolling.get('on_time_rate_7d', 0.0)), 4)}</div></div>
    <div class=\"card\"><div>Model</div><div class=\"metric\">{selected_model}</div></div>
    <div class=\"card\"><div>Model AUC</div><div class=\"metric\">{model_summary.get('test_metrics', {}).get('auc', 'NA')}</div></div>
    <div class=\"card\"><div>Data Quality</div><div class=\"metric\">{quality_status}</div></div>
  </div>

  <h2>Top Causes</h2>
  <table>
    <thead><tr><th>Cause</th><th>Count</th></tr></thead>
    <tbody>{cause_rows}</tbody>
  </table>

  <h2>Queue (Top 25)</h2>
  <table>
    <thead><tr><th>Shipment</th><th>Order</th><th>Score</th><th>Risk</th><th>Cause</th></tr></thead>
    <tbody>{queue_rows}</tbody>
  </table>

  <h2>Recent KPI (14 Days)</h2>
  <table>
    <thead><tr><th>Date</th><th>Total Shipments</th><th>On-Time %</th><th>Avg Delay Hrs</th><th>Late</th></tr></thead>
    <tbody>{series_rows}</tbody>
  </table>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html.strip() + "\n")
