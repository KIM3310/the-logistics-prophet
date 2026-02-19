# The Logistics Prophet Runbook

## Daily startup
1. Run full pipeline:
```bash
python3 scripts/run_pipeline.py
```
   - 동시 실행 방지를 위해 파이프라인 락(`data/processed/.pipeline.lock`)이 자동 적용됩니다.
2. Launch dashboard:
```bash
python3 -m streamlit run app/dashboard.py
```
3. (Optional) Push metrics to Datadog:
```bash
python3 scripts/push_datadog.py --apply-dashboard --apply-monitors
```

## Service console operations
1. Open dashboard and move high-risk shipments to `Check`.
2. Use `Bulk Update` to set `owner/step/eta/note` for many shipments.
3. Add ETA and fix note.
4. Check `Health` for `Past ETA`, `Stale 24h+`, `Critical No Owner`, `Owner Load`.
5. Check `Core Board`, `Next Actions`, and `Time Watch` for step load, top actions, and late items.
6. Use quick actions (`Assign Me`, `Move Next`, `ETA +2h`) for urgent items.
7. Review `Incident Suggestions` and create one-click incident when rule triggers.
8. Open incident if unresolved critical queue grows.
9. Confirm queue state is persisted in `data/processed/service_store.db`.
   - Queue sync는 최신 스코어 대상 기준으로 재동기화되며, 스테일 shipment는 자동 정리됩니다.
   - 동일 추천 규칙(`rule_id`)은 기존 Open/Monitoring 인시던트를 갱신하여 중복 생성을 방지합니다.
   - ETA는 ISO-8601 형식만 허용됩니다(예: `2026-02-13T12:30:00+00:00`).
   - Operator 상태 전이는 정책(Guardrail)에 의해 제한되며 Admin은 override 가능합니다.

## Authentication and roles
- `admin`: full access (queue, incidents, user management)
- `operator`: queue + incidents
- `viewer`: read-only

Default credentials:
- `admin / admin123!`
- `operator / ops123!`
- `viewer / view123!`

Security toggles:
- Disable demo bootstrap users: `LP_BOOTSTRAP_DEMO_USERS=0`
- Override bootstrap passwords:
```bash
export LP_DEMO_ADMIN_PASSWORD="StrongAdminPass!123"
export LP_DEMO_OPERATOR_PASSWORD="StrongOperatorPass!123"
export LP_DEMO_VIEWER_PASSWORD="StrongViewerPass!123"
```
- Hide/show login credential hint in UI: `LP_SHOW_DEMO_CREDENTIALS=1`
- Login lock policy: `LP_AUTH_MAX_FAILED_ATTEMPTS` (default: 5), `LP_AUTH_LOCK_MINUTES` (default: 1)
  - legacy alias: `LP_LOGIN_MAX_ATTEMPTS`, `LP_LOGIN_LOCK_MINUTES`
- Deterministic time-axis replay: `LP_ANCHOR_DATE=YYYY-MM-DD`

Rotate users/passwords with:
```bash
python3 scripts/manage_users.py --username alice --display-name "Alice" --role operator --password "StrongPass!123"
python3 scripts/manage_users.py --list
```

Verify audit chain integrity:
```bash
python3 scripts/verify_audit.py
python3 scripts/service_health_audit.py --warn-as-error
```

Check rule-based incident recommendations:
```bash
python3 scripts/service_core_snapshot.py
python3 scripts/service_core_worklist.py
python3 scripts/workflow_sla_snapshot.py
python3 scripts/recommend_incidents.py
python3 scripts/recommend_incidents.py --apply --owner auto-ops --actor auto-ops --actor-role operator
```

## Quality gate policy
- `scripts/run_data_quality.py` must pass before model training.
- If quality status is `fail`, pipeline stops automatically.
- Review `data/output/data_quality_report.json` for failed checks.

## Incident simulation
- Generate degraded signals:
```bash
python3 scripts/replay_alert_scenario.py
```
- Push degraded signals to Datadog:
```bash
python3 scripts/replay_alert_scenario.py --push
```

## Recovery procedure
1. Re-run pipeline to publish healthy metrics.
2. Validate model AUC in `data/output/training_summary.json`.
3. Validate queue severity distribution in `data/output/daily_risk_queue.csv`.
4. Confirm dashboard panels return to normal ranges.

## Troubleshooting
- Missing Python packages:
```bash
python3 -m pip install --user -r requirements.txt
```
- Reset only service state (keep analytical data):
```bash
rm -f data/processed/service_store.db && python3 scripts/init_service_store.py
```
- Empty dashboard output:
  - Check `data/output/monitoring_metrics.json`
  - Check `data/output/sparql_results.json`
  - Re-run `python3 scripts/run_pipeline.py`
- Pipeline lock timeout:
  - If another run is active, wait and retry.
  - 오래된 락 파일은 자동 회수됩니다(기본 900초). 필요 시 임계치 조정:
```bash
PIPELINE_LOCK_STALE_SEC=300 python3 scripts/run_pipeline.py
```
  - For faster fail in CI/debug:
```bash
PIPELINE_LOCK_TIMEOUT_SEC=5 python3 scripts/run_pipeline.py
```
