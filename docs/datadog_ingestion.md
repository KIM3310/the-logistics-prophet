# Datadog Integration (Operational)

This repository now supports full Datadog integration workflows:
- metric payload export
- dashboard creation
- monitor creation
- incident replay

## 1) Generate current metrics payload
```bash
python3 scripts/run_pipeline.py
python3 scripts/export_datadog_series.py
```

## 2) Configure credentials
```bash
export DD_API_KEY="<YOUR_API_KEY>"
export DD_APP_KEY="<YOUR_APP_KEY>"
export DD_SITE="datadoghq.com"  # or datadoghq.eu, us3.datadoghq.com, etc.
```

## 3) Push metrics and apply resources
```bash
python3 scripts/push_datadog.py --apply-dashboard --apply-monitors
```

## 4) Dry-run validation (no API calls)
```bash
python3 scripts/push_datadog.py --dry-run --apply-dashboard --apply-monitors
```

## 5) Replay alert scenario
```bash
python3 scripts/replay_alert_scenario.py --push
```

Generated files:
- `data/output/datadog_series_payload.json`
- `data/output/datadog_replay_payload.json`

Configuration sources:
- `monitoring/datadog_dashboard.json`
- `monitoring/monitors.yaml`
