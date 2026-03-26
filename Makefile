VENV_PY := .venv/bin/python
PYTHON ?= python3
LP_ANCHOR_DATE ?= 2026-03-14
ifeq ($(wildcard $(VENV_PY)), $(VENV_PY))
PYTHON := $(VENV_PY)
endif

.PHONY: setup run generate marts service-init queue-sync users audit quality semantic train score report monitor health compact-summary review-summary review-pack decision-board action-impact core-snapshot core-worklist workflow-sla incident-reco incident-reco-apply incident-reco-ollama datadog replay scenario test smoke-dashboard verify clean dashboard docker-build docker-up demo-local demo-local-kill demo-local-debug demo-local-open

setup:
	python3 -m venv .venv && . .venv/bin/activate && python -m pip install -U pip && python -m pip install -e ".[dev]"

run:
	$(PYTHON) scripts/run_pipeline.py

generate:
	$(PYTHON) scripts/generate_data.py

marts:
	$(PYTHON) scripts/build_marts.py

service-init:
	$(PYTHON) scripts/init_service_store.py

queue-sync:
	$(PYTHON) scripts/sync_service_queue.py

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

health: queue-sync
	$(PYTHON) scripts/service_health_audit.py --warn-as-error

compact-summary:
	$(PYTHON) scripts/service_review_surface.py --section compact_summary

review-summary:
	$(PYTHON) scripts/service_review_surface.py --section review_summary

review-pack:
	$(PYTHON) scripts/service_review_surface.py --section review_pack

decision-board:
	$(PYTHON) scripts/service_review_surface.py --section decision_board

action-impact:
	$(PYTHON) scripts/service_review_surface.py --section action_impact_board

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
	$(PYTHON) -m pytest

smoke-dashboard:
	@set -eu; \
	PORT=8502; \
	LOG=/tmp/the-logistics-prophet-dashboard-smoke.log; \
	LP_ANCHOR_DATE=$(LP_ANCHOR_DATE) $(PYTHON) scripts/run_pipeline.py >/tmp/the-logistics-prophet-pipeline-smoke.log 2>&1; \
	LP_ANCHOR_DATE=$(LP_ANCHOR_DATE) $(PYTHON) -m streamlit run app/dashboard.py --server.port=$$PORT --server.address=127.0.0.1 --server.headless=true >$$LOG 2>&1 & \
	pid=$$!; \
	trap 'kill $$pid >/dev/null 2>&1 || true' EXIT INT TERM; \
	for _ in 1 2 3 4 5 6 7 8 9 10; do \
		if curl -fsS "http://127.0.0.1:$$PORT/_stcore/health" >/dev/null 2>&1; then \
			break; \
		fi; \
		sleep 1; \
	done; \
	curl -fsS "http://127.0.0.1:$$PORT/_stcore/health" >/dev/null; \
	echo "smoke ok: http://127.0.0.1:$$PORT"

verify: run health test smoke-dashboard

clean:
	rm -f data/processed/control_tower.db
	rm -f data/model/model_artifact.json
	rm -f data/output/*.json
	rm -f data/output/*.csv
	rm -f data/output/*.html
