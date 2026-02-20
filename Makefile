VENV_PY := .venv/bin/python
PYTHON ?= python3
ifeq ($(wildcard $(VENV_PY)), $(VENV_PY))
PYTHON := $(VENV_PY)
endif

.PHONY: run generate marts service-init users audit quality semantic train score report monitor health core-snapshot core-worklist workflow-sla incident-reco incident-reco-apply incident-reco-ollama datadog replay scenario test clean dashboard docker-build docker-up demo-local demo-local-kill demo-local-debug demo-local-open

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

health:
	$(PYTHON) scripts/service_health_audit.py --warn-as-error

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

incident-reco-ollama:
	$(PYTHON) scripts/recommend_incidents.py --llm-provider ollama --ollama-healthz

datadog:
	$(PYTHON) scripts/push_datadog.py --dry-run --apply-dashboard --apply-monitors

replay:
	$(PYTHON) scripts/replay_alert_scenario.py

scenario:
	$(PYTHON) scripts/scenario_runner.py

dashboard:
	$(PYTHON) -m streamlit run app/dashboard.py

demo-local:
	bash scripts/start_demo_local.sh

demo-local-kill:
	bash scripts/start_demo_local.sh --kill-port

demo-local-debug:
	bash scripts/start_demo_local.sh --kill-port --debug

demo-local-open:
	bash scripts/start_demo_local.sh --kill-port --open

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
