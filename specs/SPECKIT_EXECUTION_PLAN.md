# SpecKit Execution Plan (V2)

## Phase 1: Data + Quality Foundation
- Generate synthetic data.
- Materialize marts.
- Enforce quality gate and stop on fail.

## Phase 2: Model Lifecycle
- Train baseline logistic and challenger gradient boosting.
- Apply probability calibration.
- Select winner by holdout metrics.
- Export SHAP global/local explanations.

## Phase 3: Semantic Layer
- Build RDF instance graph from operational marts.
- Execute competency SPARQL queries.
- Export query results for UI and monitoring context.

## Phase 4: Operations Outputs
- Score latest batch and generate risk queue.
- Produce HTML report + monitoring payload.
- Export Datadog metric series.

## Phase 5: Production Readiness
- Datadog push and monitor-as-code script.
- Incident replay script.
- Docker + Compose deployment.
- CI workflow and runbook.

## Quality Gates
- QG-1: Quality check status is pass/warn (not fail).
- QG-2: Selected model AUC >= 0.72.
- QG-3: SPARQL result includes >= 3 competency queries.
- QG-4: Risk queue sorted by score and includes actionable fields.
