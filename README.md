# The Logistics Prophet

> **Archived / Supporting repo**  
> The active control-tower and review-bundle story now lives primarily in **ops-reliability-workbench** and **lakehouse-contract-lab**.  
> Keep this repo as historical proof for the logistics-specific prediction + operations lane.

[![CI](https://github.com/KIM3310/the-logistics-prophet/actions/workflows/ci.yml/badge.svg)](https://github.com/KIM3310/the-logistics-prophet/actions/workflows/ci.yml)
![Python >=3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

Logistics control tower combining delivery delay prediction, SHAP explanations, ontology queries, and an operational console with incident management.

Personal project. Uses synthetic data and runs locally without external APIs. Datadog integration is optional (dry-run mode when keys are missing).

## Core Features

- Synthetic data pipeline (no external APIs needed)
- Data quality gates (schema/null/range/FK/drift checks)
- Model competition: baseline vs challenger with probability calibration
- SHAP global/local explainability artifacts
- RDF ontology + SPARQL competency queries
- Streamlit ops console (queue, incidents, health, governance)
- Role-based access control (admin/operator/viewer)
- Audit trail with hash-chain integrity
- Evidence pack export (ZIP + SHA-256 manifest)
- Rule-based incident recommendations (optional Ollama enrichment)
- Workflow guardrails with status transitions
- Optional Datadog metrics/dashboard/monitors integration

## Demo video
https://www.youtube.com/watch?v=NDZKmDZ_R-w

## Quickstart

```bash
cd the-logistics-prophet
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[dev]"
python3 scripts/run_pipeline.py
python3 -m streamlit run app/dashboard.py
```

Default login accounts are created at bootstrap. Override passwords with env vars:
- `LP_BOOTSTRAP_DEMO_USERS=0` to skip auto-creation
- `LP_DEMO_ADMIN_PASSWORD`, `LP_DEMO_OPERATOR_PASSWORD`, `LP_DEMO_VIEWER_PASSWORD`
- `LP_SHOW_DEMO_CREDENTIALS=1` to show login hints in UI
- Login lockout: `LP_AUTH_MAX_FAILED_ATTEMPTS` (default 5), `LP_AUTH_LOCK_MINUTES` (default 1)

Reproducibility: `LP_ANCHOR_DATE=YYYY-MM-DD` fixes the synthetic date axis.

### Ollama (optional)
Default is `stub` mode (no LLM needed). Set `LP_LLM_PROVIDER=ollama` for enriched incident briefs.

## Project Structure

```text
the-logistics-prophet/
  app/dashboard.py              # Streamlit control tower
  src/control_tower/            # Core logic (data, modeling, scoring, etc.)
  scripts/                      # Pipeline, training, scoring, admin tools
  ontology/                     # RDF graph + SPARQL queries
  monitoring/                   # Datadog dashboard/monitors
  site/                         # Static site for Cloudflare Pages
  tests/
  docs/
```

## Key Commands

```bash
make demo-local          # Start everything locally
make scenario            # Run scenario + export evidence pack
make health              # Run health audit
make compact-summary     # Print the compact proof summary JSON
make review-summary      # Print the compact reviewer summary JSON
make review-pack         # Print the review pack JSON
make decision-board      # Print the action decision board JSON
make action-impact       # Print KPI deltas for recommended actions
make run                 # Run pipeline
make dashboard           # Start Streamlit
```

## One-command proof path

If you want the shortest end-to-end confidence check, run:

```bash
make verify
```

That single command rebuilds the local pipeline inputs, re-runs the health and
test lanes, and smoke-checks the Streamlit dashboard so the reviewer story
stays anchored to evidence instead of screenshots alone.

## Reviewer Fast Path

Recommended first-pass review order:

1. `make health` — verify pipeline freshness, model floor, queue parity, and audit integrity
2. `make compact-summary` — read the shortest proof compression first
3. `make review-summary` — read the compressed verification snapshot
4. `make review-pack` — inspect review sequence, trust boundary, and proof assets
5. `make decision-board` — confirm the current recommended action path
6. `make action-impact` — check the expected KPI deltas before trusting intervention claims
7. `make smoke-dashboard` — verify the dashboard surface is actually serving

## Dashboard Pages

Navigate with deep links: `?page=control-tower`, `?page=worklist`, `?page=queue-update`, `?page=incidents`, `?page=insights`, `?page=governance`

UI modes: Plain (`?ui=plain`), Safe (default), Cinematic (`?ui=cinematic`)

## Service Workflow

1. Filter high-risk items in Queue
2. Update status/owner/ETA in bulk or per-item
3. Create and manage incidents
4. Check operational health (overdue ETA, stale queue, owner load)
5. All changes tracked in `service_store.db` with activity log

## Health Audit

```bash
make health
# or: python3 scripts/service_health_audit.py --warn-as-error
```

Checks pipeline freshness, quality gates, model performance, queue parity, and audit chain integrity.

## Scenario Runner

```bash
make scenario
# or: python3 scripts/scenario_runner.py --out-dir /tmp/lp-scenario
```

Outputs: `report.md`, `verdict.json`, `logistics-prophet-evidence-pack-*.zip`

## Datadog Integration

```bash
python3 scripts/export_datadog_series.py
python3 scripts/push_datadog.py --dry-run --apply-dashboard --apply-monitors
```

Remove `--dry-run` after setting `DD_API_KEY` and `DD_APP_KEY`.

## RBAC / Audit

```bash
python3 scripts/manage_users.py --list
python3 scripts/verify_audit.py
python3 scripts/recommend_incidents.py
```

## Tests

```bash
python3 -m pytest
```

44 tests covering auth/RBAC, pipeline operations, semantic queries, service health, and service store.

## Docker

```bash
docker compose up --build
```

Browser: `http://localhost:8501`

## Glossary

- PSI: Population Stability Index (drift signal)
- SHAP: SHapley Additive exPlanations
- RDF: Resource Description Framework
- SPARQL: Query language for RDF graphs
- RBAC: Role-Based Access Control

## Cloud + AI Architecture

This repository includes a neutral cloud and AI engineering blueprint that maps the current proof surface to runtime boundaries, data contracts, model-risk controls, deployment posture, and validation hooks.

- [Cloud + AI architecture blueprint](docs/cloud-ai-architecture.md)
- [Machine-readable architecture manifest](architecture/blueprint.json)
- Validation command: `python3 scripts/validate_architecture_blueprint.py`
