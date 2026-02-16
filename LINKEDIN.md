[Personal Project] The Logistics Prophet — Offline-First Logistics Control Tower

I started with a simple goal: predict delivery delays. While building it, I realized the real ops bottleneck isn’t the prediction score; it’s turning scattered signals into a repeatable workflow that produces the next action, with evidence and auditability.

The Logistics Prophet is an end-to-end control tower that runs 100% locally (deterministic synthetic data; no external APIs). It simulates the full loop: data -> quality gate -> model -> explanation -> operator actions -> audit.

What it includes
- Deterministic data pipeline -> SQLite marts
- Data Quality Gate before training/scoring (schema/null/range/FK + drift/PSI)
- Model competition (baseline vs challenger), calibration, threshold selection, daily scoring
- Explainability: SHAP global importance + per-shipment local drivers (with feature values)
- Semantic evidence: RDF instance graph + SPARQL queries (carrier/warehouse/customer/events as queryable facts)
- Ops workflow UI: Start/Check/Fix/Done board, prioritized worklist, bulk updates, and one-click actions (Assign Me / Move Next / ETA +2h)
- Incident tracking + SLA/health signals (overdue ETA, stale items, critical unassigned, owner backlog)
- Governance: RBAC + append-only audit trail with hash-chain verification
- Reviewer exports: Scenario Runner report + Evidence Pack ZIP with SHA-256 manifest + postmortem template
- Optional Datadog integration (dry-run supported)

Engineering decisions I actually had to debug
- Vocabulary alignment: standardized core terms across UI/DB/runbooks and added transition guardrails to prevent invalid moves
- Reliability: pipeline lock to avoid concurrent artifact corruption, caching for fast explain views, and a one-command scenario runner to reproduce results for reviewers

Demo video:
https://www.youtube.com/watch?v=NDZKmDZ_R-w

GitHub:
https://github.com/KIM3310/the-logistics-prophet

Quick start:
make demo-local-open
make scenario

