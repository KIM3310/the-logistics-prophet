PYTHON ?= python3

.PHONY: run generate marts service-init users audit quality semantic train score report monitor core-snapshot core-worklist workflow-sla incident-reco incident-reco-apply datadog replay test clean dashboard docker-build docker-up demo-local

run:
	$(PYTHON) scripts/run_pipeline.py

generate:
	$(PYTHON) scripts/generate_data.py

marts:
	$(PYTHON) scripts/build_marts.py

service-init:
	$(PYTHON) scripts/init_service_store.py

users:
	$(PYTHON) scripts/manage_users.py --list

audit:
	$(PYTHON) scripts/verify_audit.py

quality:
	$(PYTHON) scripts/run_data_quality.py

semantic:
	$(PYTHON) scripts/build_semantic_layer.py

train:
	$(PYTHON) scripts/train_model.py

score:
	$(PYTHON) scripts/score_daily.py

report:
	$(PYTHON) scripts/build_report.py

monitor:
	$(PYTHON) scripts/export_datadog_series.py

core-snapshot:
	$(PYTHON) scripts/service_core_snapshot.py

core-worklist:
	$(PYTHON) scripts/service_core_worklist.py

workflow-sla:
	$(PYTHON) scripts/workflow_sla_snapshot.py

incident-reco:
	$(PYTHON) scripts/recommend_incidents.py

incident-reco-apply:
	$(PYTHON) scripts/recommend_incidents.py --apply

datadog:
	$(PYTHON) scripts/push_datadog.py --dry-run --apply-dashboard --apply-monitors

replay:
	$(PYTHON) scripts/replay_alert_scenario.py

dashboard:
	$(PYTHON) -m streamlit run app/dashboard.py

demo-local:
	bash scripts/start_demo_local.sh

docker-build:
	docker build -t semantic-control-tower:latest .

docker-up:
	docker compose up --build

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

clean:
	rm -f data/processed/control_tower.db
	rm -f data/model/model_artifact.json
	rm -f data/output/*.json
	rm -f data/output/*.csv
	rm -f data/output/*.html
