# The Logistics Prophet - Flagship Spec (V3)

## 0) Intent
Turn this repo into a flagship-quality, offline-first portfolio project that feels like a real internal ops product:
beautiful UI, deterministic local demo, evidence-backed decisions (quality gate + SHAP + ontology), and production-minded rigor.

This spec is written to be executable:
- clear constraints
- measurable acceptance criteria
- a demo path that works without cloud keys or external services

## 1) COT (Concept of Operations)
### Users
- Ops manager: cares about SLA risk, backlog, MTTR, ownership, auditability.
- Operator: triages the worklist, takes actions, documents notes/ETA, creates incidents.
- Analyst/Engineer: validates data quality, model health, drift, and semantic evidence.

### Primary workflow
1. Run pipeline (or use cached artifacts) to generate "today's" queue and evidence artifacts.
2. Open Control Tower view to see KPI + quality + backlog + top risk.
3. Worklist view: Start with the highest-urgency items, assign owner, move status forward, adjust ETA.
4. Explain view: For a selected shipment, review:
   - the model drivers (local SHAP)
   - raw feature values
   - semantic evidence (RDF/SPARQL results)
   - recent activity/audit log
5. Escalate to Incident when rules or context indicate a broader operational issue.
6. Export artifacts:
   - incident comment draft
   - postmortem draft
   - audit verification summary

## 2) Constraints (Non-Negotiables)
- Offline-first: the app must run without internet access.
- Local-only: no cloud resources required; no API keys required.
- Deterministic: the default demo produces stable output with a fixed seed.
- One-command: a local runner exists to set up + run the demo smoothly.

## 3) UX / Design Goals
- "Single decision surface": fewer scroll-walls, more intentional navigation.
- Crisp, high-contrast typography with a clear visual system.
- Evidence always next to action:
  - work item -> driver -> evidence -> next action
- Error states look intentional (missing artifacts, read-only role, etc.).

## 4) Functional Enhancements (V3)
### UI Navigation
- Convert the single long page into a tabbed product navigation:
  - Control Tower
  - Worklist
  - Queue + Update
  - Incidents
  - Insights (Model + Quality + Semantic)
  - Governance (Audit + RBAC)

### Explainability (local SHAP)
- Add a "Shipment Explain" panel:
  - show local SHAP drivers for the selected shipment
  - show the feature values aligned to those drivers
  - export a one-page incident note (Markdown)

### Ontology evidence (RDF/SPARQL)
- Add a per-shipment semantic evidence view:
  - carrier, warehouse, risk state, delay events
  - derived evidence shown alongside actions
- Cache the RDF graph to keep interactions fast.

### Postmortem export
- Add a postmortem generator in the UI:
  - select incident -> export a prefilled postmortem draft
  - include timeline from audit activity

### Evidence pack export
- Add an offline "evidence pack" exporter:
  - ZIP bundle with key artifacts (quality/model/SHAP/SPARQL/ops report)
  - include `manifest.json` with SHA-256 file hashes for reviewer verification

### Offline hardening
- Remove runtime dependency on externally hosted assets (e.g., Google Fonts imports).

## 5) Demo / Ops Requirements
- Provide `scripts/start_demo_local.sh` to:
  - create venv if missing
  - install dependencies
  - run pipeline (or skip if artifacts are fresh)
  - start Streamlit on port 8501
- Makefile target: `make demo-local`.

## 6) Acceptance Criteria
- `make demo-local` starts the app and the UI loads with no errors.
- No external network requests are required for core UI rendering.
- Pipeline artifacts exist after a fresh run:
  - `data/output/monitoring_metrics.json`
  - `data/output/data_quality_report.json`
  - `data/output/shap_global_importance.csv`
  - `data/output/shap_local_explanations.csv`
  - `data/semantic/instance_graph.ttl`
- Explain panel works for any selected shipment and shows:
  - risk score + band + status + owner + recommendation
  - local SHAP drivers (top 3)
  - semantic evidence rows (carrier/warehouse + delay events)
- Governance tab exports:
  - evidence pack ZIP downloads successfully
  - postmortem draft downloads successfully
- Existing unit tests pass (`make test`) and a quick smoke run is stable.

## 7) Deliverables
- UI upgrades in `app/dashboard.py`
- Semantic query helper in `src/control_tower/*` (testable)
- Local runner script in `scripts/start_demo_local.sh`
- Updated `Makefile` + `README.md`
