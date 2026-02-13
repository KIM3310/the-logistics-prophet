#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data" / "output" / "datadog_replay_payload.json"


def _metric(name: str, value: float, now: int) -> dict:
    return {
        "metric": name,
        "type": 0,
        "points": [[now, float(value)]],
        "tags": ["env:portfolio", "service:semantic-control-tower", "scenario:incident-replay"],
    }


def build_replay_payload(now: int) -> dict:
    series = [
        _metric("semantic_control_tower.kpi.on_time_rate", 0.72, now),
        _metric("semantic_control_tower.kpi.sla_breach_count", 39, now),
        _metric("semantic_control_tower.kpi.avg_delay_hours", 16.5, now),
        _metric("semantic_control_tower.model.auc", 0.63, now),
        _metric("semantic_control_tower.risk_queue.critical_count", 24, now),
        _metric("semantic_control_tower.risk_queue.high_count", 37, now),
        _metric("semantic_control_tower.quality.fail_count", 2, now),
        _metric("semantic_control_tower.pipeline.success", 0, now),
    ]
    return {"series": series}


def push_payload(payload: dict, site: str, api_key: str, app_key: str) -> None:
    url = f"https://api.{site}/api/v2/series"
    headers = {
        "DD-API-KEY": api_key,
        "DD-APPLICATION-KEY": app_key,
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
    resp.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a Datadog alert scenario for demo")
    parser.add_argument("--site", default=os.getenv("DD_SITE", "datadoghq.com"))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    now = int(time.time())
    payload = build_replay_payload(now)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    summary = {
        "output": str(OUTPUT_FILE),
        "series_count": len(payload["series"]),
        "pushed": False,
    }

    if args.push:
        api_key = os.getenv("DD_API_KEY", "")
        app_key = os.getenv("DD_APP_KEY", "")
        if not api_key or not app_key:
            raise SystemExit("Set DD_API_KEY and DD_APP_KEY to push replay metrics.")
        push_payload(payload, args.site, api_key, app_key)
        summary["pushed"] = True

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
