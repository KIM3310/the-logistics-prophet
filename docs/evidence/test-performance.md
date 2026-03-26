# the-logistics-prophet — Test Performance Report

> Generated: 2026-03-19 | Runner: pytest 8.3.5 | Python 3.11.15

## Test Suite Summary

| Metric | Value |
|--------|-------|
| Total tests collected | 44 |
| Passed | 44 |
| Failed | 0 |
| Execution time | **100.96s** (1m 40s) |

## Test Breakdown by Module

| Test File | Tests | Coverage Area |
|-----------|-------|---------------|
| test_auth_rbac_audit.py | 6 | Authentication, RBAC, audit trail |
| test_frontend_metadata.py | 3 | Frontend metadata contract |
| test_incident_llm.py | 5 | Incident LLM integration |
| test_model_artifact.py | 1 | Model artifact management |
| test_pipeline_lock.py | 4 | Pipeline locking mechanisms |
| test_pipeline_smoke.py | 2 | Pipeline smoke tests |
| test_quality_semantic.py | 2 | Semantic quality checks |
| test_queue_sync.py | 2 | Queue synchronization |
| test_semantic_queries.py | 1 | Semantic query engine |
| test_service_health.py | 2 | Service health endpoints |
| test_service_store.py | 15 | Service store operations |
| test_synthetic_anchor.py | 1 | Synthetic anchor generation |

## Performance Notes

- The 100.96s execution time is dominated by pipeline and ML model operations (sklearn, shap, numpy)
- Service store tests (15 tests) represent the largest functional coverage area
- All tests run without external service dependencies (mocked where needed)
- Pipeline lock and smoke tests validate the core orchestration layer
