#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
SERIES_PAYLOAD = ROOT / "data" / "output" / "datadog_series_payload.json"
DASHBOARD_JSON = ROOT / "monitoring" / "datadog_dashboard.json"
MONITORS_YAML = ROOT / "monitoring" / "monitors.yaml"


class DatadogClient:
    def __init__(self, api_key: str, app_key: str, site: str) -> None:
        self.api_key = api_key
        self.app_key = app_key
        self.base_url = f"https://api.{site}"
        self.headers = {
            "DD-API-KEY": api_key,
            "DD-APPLICATION-KEY": app_key,
            "Content-Type": "application/json",
        }

    def post_series(self, payload: dict) -> dict:
        url = f"{self.base_url}/api/v2/series"
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        return resp.json() if resp.content else {"status": "ok"}

    def create_dashboard(self, payload: dict) -> dict:
        url = f"{self.base_url}/api/v1/dashboard"
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        return resp.json() if resp.content else {"status": "ok"}

    def create_monitor(self, payload: dict) -> dict:
        url = f"{self.base_url}/api/v1/monitor"
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        return resp.json() if resp.content else {"status": "ok"}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_monitors(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    monitors = payload.get("monitors", [])
    if not isinstance(monitors, list):
        raise ValueError("monitor YAML format invalid: expected key `monitors` as list")
    return monitors


def main() -> None:
    parser = argparse.ArgumentParser(description="Push metrics/dashboard/monitors to Datadog")
    parser.add_argument("--site", default=os.getenv("DD_SITE", "datadoghq.com"))
    parser.add_argument("--no-series", action="store_true", help="Skip series upload.")
    parser.add_argument("--apply-dashboard", action="store_true")
    parser.add_argument("--apply-monitors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    push_series = not args.no_series

    api_key = os.getenv("DD_API_KEY", "")
    app_key = os.getenv("DD_APP_KEY", "")
    if not args.dry_run and (not api_key or not app_key):
        raise SystemExit("Set DD_API_KEY and DD_APP_KEY environment variables first.")

    summary = {
        "site": args.site,
        "dry_run": args.dry_run,
        "series_pushed": False,
        "dashboard_created": False,
        "monitors_created": 0,
    }

    if args.dry_run:
        if push_series and SERIES_PAYLOAD.exists():
            payload = _load_json(SERIES_PAYLOAD)
            summary["series_count"] = len(payload.get("series", []))
        if args.apply_dashboard and DASHBOARD_JSON.exists():
            summary["dashboard_preview"] = True
        if args.apply_monitors and MONITORS_YAML.exists():
            summary["monitor_count"] = len(_load_monitors(MONITORS_YAML))
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return

    client = DatadogClient(api_key=api_key, app_key=app_key, site=args.site)

    if push_series:
        payload = _load_json(SERIES_PAYLOAD)
        client.post_series(payload)
        summary["series_pushed"] = True
        summary["series_count"] = len(payload.get("series", []))

    if args.apply_dashboard:
        dashboard = _load_json(DASHBOARD_JSON)
        created = client.create_dashboard(dashboard)
        summary["dashboard_created"] = True
        summary["dashboard_id"] = created.get("id")
        summary["dashboard_url"] = created.get("url")

    if args.apply_monitors:
        monitor_payloads = _load_monitors(MONITORS_YAML)
        created_names = []
        for monitor in monitor_payloads:
            created = client.create_monitor(monitor)
            created_names.append(created.get("name", monitor.get("name", "unknown")))
        summary["monitors_created"] = len(created_names)
        summary["monitor_names"] = created_names

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
