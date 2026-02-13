# The Logistics Prophet (V2)

## 1) Problem Statement
Operations teams need a single decision surface that is trustworthy, explainable, and monitorable. Most demos stop at charts. This project delivers a full operational flow from semantic modeling to incident replay.

## 2) Target Users
- Operations manager
- Logistics analyst
- Data engineer
- ML engineer

## 3) Goals
- API-free end-to-end system with synthetic data.
- Data quality gate that can stop bad runs.
- Model competition + calibration + explainability.
- Ontology + SPARQL evidence in the operator UI.
- Datadog-ready production-style observability and replay.

## 4) Functional Requirements
- FR-1: Generate synthetic supply-chain events.
- FR-2: Build SQLite marts and feature table.
- FR-3: Enforce quality checks (schema/null/range/FK/drift).
- FR-4: Train baseline vs challenger models and auto-select winner.
- FR-5: Produce SHAP global/local explanations.
- FR-6: Score latest shipments and produce action queue.
- FR-7: Build ontology instance graph and execute competency SPARQL queries.
- FR-8: Export Datadog metric series and apply dashboard/monitor resources.
- FR-9: Support incident replay metrics.

## 5) Success Metrics
- >= 10,000 model feature rows per run.
- Selected model test AUC >= 0.72.
- Quality gate status != fail for default run.
- Risk queue output includes top 50 prioritized shipments.
- SPARQL result file contains >= 3 query result sets.

## 6) Acceptance Criteria
- `python3 scripts/run_pipeline.py` completes successfully.
- Artifacts are generated:
  - `data/model/model_artifact.json`
  - `data/model/selected_model.pkl`
  - `data/output/data_quality_report.json`
  - `data/output/sparql_results.json`
  - `data/output/shap_global_importance.csv`
  - `data/output/daily_risk_queue.csv`
  - `data/output/monitoring_metrics.json`
- Tests pass with `python3 -m unittest discover -s tests -p 'test_*.py'`.
