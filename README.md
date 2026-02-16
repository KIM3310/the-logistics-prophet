# The Logistics Prophet (Portfolio)

온톨로지 + 데이터 품질 게이트 + 모델 경쟁 실험 + SHAP 설명 + Datadog 운영 연동까지 포함한
실서비스 스타일 내부 관제 프로젝트입니다.

Note: 개인 포트폴리오 프로젝트입니다. 데이터는 synthetic이며 외부 API 없이도 로컬에서 재현 가능합니다.
Datadog 전송은 옵션이며, 키가 없으면 dry-run/파일 출력 형태로 동작합니다.

## Demo video
https://www.youtube.com/watch?v=NDZKmDZ_R-w

## 서비스 정의
배송 지연 리스크를 **사전에 예측**하고,
운영팀이 바로 조치할 수 있도록 **우선순위 큐와 근거(SHAP + SPARQL)** 를 제공하는 서비스.

## 핵심 기능
- API-free synthetic data pipeline
- Data quality gate (schema/null/range/FK/drift)
- Model competition: baseline vs challenger + probability calibration
- Explainability: SHAP global/local artifacts
- Ontology operationalization: RDF instance graph + SPARQL competency queries
- Datadog operational integration (metrics, dashboard, monitors, replay)
- Service operations console (owner assignment, status workflow, incident tracking, run history)
- Evidence pack export (ZIP + manifest with file hashes)
- Postmortem draft export (Markdown + audit timeline)
- Bulk queue action (multi-shipment status/owner/ETA/note update)
- Ops health analytics (overdue ETA, stale queue, critical unassigned, owner backlog)
- Rule-based incident recommendation engine + one-click incident creation
- Recommendation incident deduplication (`rule_id` 기준으로 기존 Open/Monitoring 인시던트 재사용)
- Core Board (Start/Check/Fix/Done load + urgent list + top causes)
- Next Actions (Start/Check/Fix top action list with urgency score)
- Next Actions quick actions (`Assign Me`, `Move Next`, `ETA +2h`)
- Workflow transition guardrails (operator 정책 기반 상태 전이, admin override)
- Status alias support for core terms (`Start/Check/Fix/Done/Skip`)
- Time Watch (step limit check and late item list)
- Strict workflow input validation (ISO-8601 ETA, incident severity/status)
- Role-based access control (admin/operator/viewer) with authenticated login
- Audit trail with hash-chain integrity verification
- Atomic SQLite mart build + stale-aware pipeline lock for safer concurrent execution
- Streamlit ops console UI (core board, queue, health, incidents)
- Docker deployment + GitHub Actions CI

## 프로젝트 구조
```text
the-logistics-prophet/
  app/
    dashboard.py
  docs/
    datadog_ingestion.md
    runbook.md
    postmortem_template.md
  monitoring/
    datadog_dashboard.json
    monitors.yaml
  ontology/
    supply_chain.ttl
    competency_questions.md
  scripts/
    run_pipeline.py
    init_service_store.py
    run_data_quality.py
    build_semantic_layer.py
    train_model.py
    score_daily.py
    build_report.py
    export_datadog_series.py
    push_datadog.py
    replay_alert_scenario.py
    scenario_runner.py
    manage_users.py
    verify_audit.py
    service_core_snapshot.py
    workflow_sla_snapshot.py
    recommend_incidents.py
  src/control_tower/
    synthetic_data.py
    sqlite_pipeline.py
    quality.py
    semantic_layer.py
    service_store.py
    modeling.py
    scoring.py
    ops_output.py
  .github/workflows/ci.yml
  Dockerfile
  docker-compose.yml
```

## 빠른 실행
```bash
cd the-logistics-prophet
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/run_pipeline.py
python3 -m streamlit run app/dashboard.py
```

기본 로그인 계정:
- `admin / admin123!`
- `operator / ops123!`
- `viewer / view123!`

## UI 모드 (Plain / Safe / Cinematic)
브라우저/환경 호환성을 위해 UI는 Streamlit 기본 컴포넌트(컨테이너/컬럼/테이블)로 구성되어 있습니다.

- Plain (troubleshooting): `http://127.0.0.1:8501?ui=plain`
- Safe (default): `http://127.0.0.1:8501` (Streamlit theme 기반)
- Cinematic: `http://127.0.0.1:8501?ui=cinematic` (legacy alias: `?cinematic=1`)
  - 포트폴리오 데모용으로 **은은한 빛(aurora) 분위기 CSS**가 적용됩니다(`app/assets/cinematic.css`).

Blank screen 트러블슈팅:
- `http://127.0.0.1:8501?smoke=1` (최소 렌더링 확인)
- `http://127.0.0.1:8501?ui=plain` (시각 요소 최소화)

## 빠른 네비게이션 (Deep link)
- `?page=control-tower`
- `?page=worklist`
- `?page=queue-update`
- `?page=incidents`
- `?page=insights`
- `?page=governance`

## One-command 옵션
```bash
make demo-local
make demo-local-kill
make demo-local-open
make demo-local-debug
make run
make scenario
make dashboard
```

## Scenario Runner (Report + Evidence Pack)
리뷰어에게 “재현 가능한 결과물”을 공유하기 위해, 엔드투엔드 상태를 점검하고 산출물을 한번에 export 합니다.

```bash
make scenario
# 또는:
python3 scripts/scenario_runner.py --out-dir /tmp/lp-scenario
```

출력:
- `report.md` (Executive snapshot + Worklist top actions + governance)
- `verdict.json` (pipeline/quality/audit 결과)
- `logistics-prophet-evidence-pack-*.zip` (SHA-256 manifest 포함)

## 주요 산출물
- `data/model/model_artifact.json`
- `data/model/selected_model.pkl`
- `data/output/data_quality_report.json`
- `data/output/model_comparison.json`
- `data/output/training_summary.json`
- `data/output/shap_global_importance.csv`
- `data/output/shap_local_explanations.csv`
- `data/output/sparql_results.json`
- `data/output/daily_risk_queue.csv`
- `data/output/monitoring_metrics.json`
- `data/output/datadog_series_payload.json`
- `data/output/ops_report.html`
- `data/processed/service_store.db`

## Datadog 연동
```bash
python3 scripts/export_datadog_series.py
python3 scripts/push_datadog.py --dry-run --apply-dashboard --apply-monitors
```
실제 전송은 `DD_API_KEY`, `DD_APP_KEY` 설정 후 `--dry-run` 제거.

## Alert Replay
```bash
python3 scripts/replay_alert_scenario.py
python3 scripts/replay_alert_scenario.py --push
```

## Service Workflow (Dashboard)
1. `Queue`에서 고위험 건 필터링
2. `Update`에서 단건 또는 벌크로 `step/owner/eta/note` 업데이트
3. `Incident`에서 인시던트 생성/상태관리
4. `Health`에서 past/stale/owner load 운영건전성 확인
5. 모든 변경은 `service_store.db`에 저장되고 Activity Log로 추적

## RBAC / Audit 운영 커맨드
```bash
python3 scripts/manage_users.py --list
python3 scripts/manage_users.py --username alice --display-name "Alice" --role operator --password "StrongPass!123"
python3 scripts/verify_audit.py
python3 scripts/service_core_snapshot.py
python3 scripts/service_core_worklist.py
python3 scripts/workflow_sla_snapshot.py
python3 scripts/recommend_incidents.py
python3 scripts/recommend_incidents.py --apply --owner auto-ops --actor auto-ops --actor-role operator
```

## Docker 실행
```bash
docker compose up --build
```
브라우저: `http://localhost:8501`

## 테스트
```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## 면접 데모 순서 (추천)
1. `python3 scripts/run_pipeline.py` 실행
2. Dashboard에서 KPI/SHAP/Graph + `Core Board`/`Next Actions`/`Queue`/`Update` 시연
3. Datadog dry-run 및 replay 시나리오 시연
4. Runbook 기반 운영 절차 설명
5. Governance 탭에서 Evidence Pack / Postmortem Export 시연

## Glossary (first-time readers)
- PSI: Population Stability Index (drift signal)
- SHAP: SHapley Additive exPlanations (explainability)
- RDF: Resource Description Framework (semantic graph)
- SPARQL: query language for RDF graphs
- RBAC: Role-Based Access Control
