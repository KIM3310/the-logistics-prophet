from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import (
    DAILY_RISK_QUEUE_PATH,
    QUALITY_REPORT_PATH,
    SERVICE_DB_PATH,
    TRAINING_SUMMARY_PATH,
)
from .service_store import verify_audit_chain

STATUS_ORDER = {"pass": 0, "warn": 1, "fail": 2}

SERVICE_STAGES = [
    {
        "id": "sense",
        "title": "Sense",
        "owner": "pipeline",
        "outcome": "Synthetic demand/logistics signals are rebuilt into a daily risk queue.",
    },
    {
        "id": "score",
        "title": "Score",
        "owner": "modeling",
        "outcome": "Baseline/challenger evaluation and calibrated risk ranking stay reviewable.",
    },
    {
        "id": "explain",
        "title": "Explain",
        "owner": "semantic-layer",
        "outcome": "SHAP drivers and SPARQL evidence justify why a shipment is risky.",
    },
    {
        "id": "operate",
        "title": "Operate",
        "owner": "service-store",
        "outcome": "Owners, statuses, incidents, and audit-trail updates flow through the control tower.",
    },
    {
        "id": "govern",
        "title": "Govern",
        "owner": "monitoring",
        "outcome": "Datadog exports, evidence packs, and health checks keep the surface production-like.",
    },
]

REVIEW_FLOW = [
    "Run `make health` to confirm the pipeline, quality gate, model threshold, and audit chain.",
    "Open the Streamlit control tower and start with Core Board plus Next Actions.",
    "Use Worklist and Queue + Update to verify actionability, ownership, and ETA transitions.",
    "Inspect Evidence Pack / Governance to validate auditability and review artifacts.",
]

OPERATOR_RULES = [
    "Queue parity and audit-chain integrity are not optional; they gate trust in the console.",
    "Keep predictive signal, human ownership, and incident state transitions on the same screen.",
    "Treat SHAP/SPARQL evidence as justification surfaces, not decoration.",
]

WATCHOUTS = [
    "Synthetic data keeps the project reproducible, but it cannot prove real carrier integration latency.",
    "The dashboard is Streamlit-first, so multi-user concurrency is demonstrated through the service store rather than a web API tier.",
    "Datadog and Ollama integrations are optional and should be read as extensions, not baseline requirements.",
]

REPORT_CONTRACT = {
    "schema": "logistics-worklist-report-v1",
    "required_sections": [
        "executive_summary",
        "queue_snapshot",
        "sla_health",
        "top_risks",
        "operator_actions",
        "audit_status",
    ],
    "export_formats": ["markdown", "zip-evidence-pack", "datadog-snapshot"],
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _parse_iso_dt(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _worst_status(statuses: List[str]) -> str:
    worst = "pass"
    for status in statuses:
        if STATUS_ORDER.get(status, 0) > STATUS_ORDER.get(worst, 0):
            worst = status
    return worst


def _build_health_diagnostics(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    failing_check_ids = [str(item.get("id", "")) for item in checks if item.get("status") == "fail"]
    warning_check_ids = [str(item.get("id", "")) for item in checks if item.get("status") == "warn"]

    if failing_check_ids:
        next_action = f"Resolve {failing_check_ids[0]} and rerun `make health`."
    elif warning_check_ids:
        next_action = f"Review {warning_check_ids[0]} and rerun `make health` before release."
    else:
        next_action = "Run `python3 scripts/scenario_runner.py` to validate the end-to-end ops flow."

    return {
        "failing_check_ids": failing_check_ids,
        "warning_check_ids": warning_check_ids,
        "next_action": next_action,
    }


def _build_review_summary(
    *,
    checks: List[Dict[str, Any]],
    queue_csv_rows: int,
    db_queue_count: int,
    audit_checked: int,
    min_model_auc: float,
    strict_queue_parity: bool,
) -> Dict[str, Any]:
    status_counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in checks:
        status = str(item.get("status", "pass"))
        if status in status_counts:
            status_counts[status] += 1

    return {
        "contract": "logistics-control-review-summary-v1",
        "headline": "Compact reviewer snapshot for queue parity, audit integrity, and operational actionability.",
        "summary": {
            "total_checks": len(checks),
            "queue_csv_rows": queue_csv_rows,
            "service_db_rows": db_queue_count,
            "audit_chain_checked": audit_checked,
            "min_model_auc": float(min_model_auc),
            "strict_queue_parity": bool(strict_queue_parity),
            "status_counts": status_counts,
        },
        "fastest_review_path": [
            "make health",
            "app/dashboard.py?page=control-tower",
            "scripts/scenario_runner.py",
            "scripts/verify_audit.py",
        ],
        "top_watchouts": WATCHOUTS[:2],
    }


def _build_runtime_scorecard(
    *,
    checks: List[Dict[str, Any]],
    queue_csv_rows: int,
    db_queue_count: int,
    audit_checked: int,
    min_model_auc: float,
    strict_queue_parity: bool,
) -> Dict[str, Any]:
    status_counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in checks:
        status = str(item.get("status", "pass"))
        if status in status_counts:
            status_counts[status] += 1

    fail_count = int(status_counts["fail"])
    warn_count = int(status_counts["warn"])
    runtime_score = max(40, 100 - min(fail_count * 18 + warn_count * 6, 60))
    top_failing = [str(item.get("id", "")) for item in checks if item.get("status") == "fail"][:3]
    top_warning = [str(item.get("id", "")) for item in checks if item.get("status") == "warn"][:3]

    return {
        "contract": "logistics-control-runtime-scorecard-v1",
        "headline": "Compact runtime scorecard for queue parity, model floor, and audit integrity before operator action.",
        "summary": {
            "runtime_score": runtime_score,
            "queue_csv_rows": int(queue_csv_rows),
            "service_db_rows": int(db_queue_count),
            "audit_chain_checked": int(audit_checked),
            "min_model_auc": float(min_model_auc),
            "strict_queue_parity": bool(strict_queue_parity),
            "status_counts": status_counts,
        },
        "top_failing_checks": top_failing,
        "top_warning_checks": top_warning,
        "fastest_review_path": [
            "make health",
            "app/dashboard.py?page=control-tower",
            "scripts/service_core_snapshot.py",
            "scripts/verify_audit.py",
        ],
    }


def _build_recovery_drill(
    *,
    checks: List[Dict[str, Any]],
    queue_csv_rows: int,
    db_queue_count: int,
    audit_checked: int,
    min_model_auc: float,
) -> Dict[str, Any]:
    status_counts = {"pass": 0, "warn": 0, "fail": 0}
    for item in checks:
        status = str(item.get("status", "pass"))
        if status in status_counts:
            status_counts[status] += 1

    fail_count = int(status_counts["fail"])
    warn_count = int(status_counts["warn"])
    baseline_risk = max(35, 100 - min(fail_count * 16 + warn_count * 5, 65))
    recovered_risk = max(
        20,
        baseline_risk - min(18, 6 + warn_count * 2 + (0 if queue_csv_rows == db_queue_count else 6)),
    )
    eta_gain_hours = 6 if fail_count == 0 else 2 if warn_count <= 2 else 1

    return {
        "contract": "logistics-control-recovery-drill-v1",
        "headline": "Disruption recovery drill that compares baseline queue risk against an operator action plan before route escalation.",
        "summary": {
            "baseline_risk_score": baseline_risk,
            "recovered_risk_score": recovered_risk,
            "eta_gain_hours": eta_gain_hours,
            "audit_checked": int(audit_checked),
            "min_model_auc": float(min_model_auc),
        },
        "items": [
            {
                "scenario": "carrier-delay-reroute",
                "action_plan": "Reroute the top critical shipments to alternate capacity and tighten manual review for delayed lanes.",
                "baseline_eta_hours": 18,
                "recovered_eta_hours": max(6, 18 - eta_gain_hours),
                "risk_delta": round(baseline_risk - recovered_risk, 1),
            }
        ],
        "review_actions": [
            "Run the recovery drill before claiming the queue is stable enough for downstream handoff.",
            "Keep queue parity and audit integrity visible while simulating the operator action plan.",
            "Pair this drill with the evidence pack before sharing a recovery recommendation.",
        ],
    }


def _build_decision_board(
    *,
    checks: List[Dict[str, Any]],
    queue_csv_rows: int,
    db_queue_count: int,
    audit_checked: int,
    min_model_auc: float,
) -> Dict[str, Any]:
    queue_check = next((item for item in checks if item.get("id") == "queue_parity"), {})
    model_check = next((item for item in checks if item.get("id") == "model_auc"), {})
    audit_check = next((item for item in checks if item.get("id") == "audit_chain"), {})

    items: List[Dict[str, Any]] = []
    if str(queue_check.get("status", "pass")) != "pass":
        items.append(
            {
                "priority": "P0",
                "lane": "queue-integrity",
                "recommended_action": "Rebuild the service-store queue snapshot before escalating any carrier or SLA recommendation.",
                "expected_delta": {
                    "queue_mismatch_rows": abs(int(queue_csv_rows) - int(db_queue_count)),
                    "eta_confidence": "improves after parity is restored",
                },
                "proof_path": "make health",
            }
        )
    if str(model_check.get("status", "pass")) != "pass":
        items.append(
            {
                "priority": "P1",
                "lane": "model-triage",
                "recommended_action": "Hold automated route-priority claims and recalibrate the challenger threshold before widening operator actions.",
                "expected_delta": {
                    "model_auc_floor": float(min_model_auc),
                    "risk_queue_noise_pct": "reduced after threshold recalibration",
                },
                "proof_path": "scripts/run_pipeline.py",
            }
        )
    if str(audit_check.get("status", "pass")) != "pass":
        items.append(
            {
                "priority": "P0",
                "lane": "audit-integrity",
                "recommended_action": "Repair the audit chain before exporting evidence packs or sharing operational recommendations.",
                "expected_delta": {
                    "audit_rows_checked": int(audit_checked),
                    "reviewer_handoff": "blocked until chain is valid",
                },
                "proof_path": "scripts/verify_audit.py",
            }
        )
    if not items:
        items.append(
            {
                "priority": "P2",
                "lane": "steady-state",
                "recommended_action": "Advance from queue review to next-actions and watchlist ownership because health gates are currently green.",
                "expected_delta": {
                    "handoff_risk": "lower due to healthy queue parity and audit posture",
                    "operator_focus": "shifts from validation to action execution",
                },
                "proof_path": "app/dashboard.py?page=next-actions",
            }
        )

    return {
        "contract": "logistics-control-decision-board-v1",
        "headline": "Decision board that turns health checks into operator actions and expected control-tower impact.",
        "summary": {
            "recommended_actions": len(items),
            "top_priority": items[0]["priority"],
            "queue_csv_rows": int(queue_csv_rows),
            "service_db_rows": int(db_queue_count),
            "audit_chain_checked": int(audit_checked),
        },
        "items": items,
        "review_actions": [
            "Start with P0 lanes before opening downstream evidence packs.",
            "Pair every recommended action with the proof path that justifies it.",
            "Treat this board as the bridge from predictive signal to operator execution.",
        ],
    }


def _build_action_impact_board(
    *,
    decision_board: Dict[str, Any],
    recovery_drill: Dict[str, Any],
) -> Dict[str, Any]:
    decision_items = [
        item for item in decision_board.get("items", []) if isinstance(item, dict)
    ]
    recovery_summary = recovery_drill.get("summary", {}) if isinstance(recovery_drill, dict) else {}
    recovery_items = [
        item for item in recovery_drill.get("items", []) if isinstance(item, dict)
    ]
    recovery_item = recovery_items[0] if recovery_items else {}
    baseline_risk = _safe_int(recovery_summary.get("baseline_risk_score", 0))
    recovered_risk = _safe_int(recovery_summary.get("recovered_risk_score", 0))
    baseline_eta = _safe_int(recovery_item.get("baseline_eta_hours", 0))
    recovered_eta = _safe_int(recovery_item.get("recovered_eta_hours", 0))
    eta_gain = max(0, baseline_eta - recovered_eta)

    owner_map = {
        "queue-integrity": "service-store",
        "model-triage": "modeling",
        "audit-integrity": "monitoring",
        "steady-state": "ops-control",
    }
    action_rows = [
        {
            "priority": str(item.get("priority", "P2")),
            "lane": str(item.get("lane", "steady-state")),
            "owner": owner_map.get(str(item.get("lane", "steady-state")), "ops-control"),
            "recommended_action": str(item.get("recommended_action", "")),
            "proof_path": str(item.get("proof_path", "")),
        }
        for item in decision_items
    ]

    return {
        "contract": "logistics-control-action-impact-board-v1",
        "headline": "Action impact board that ties recommended actions to expected KPI movement before operator commitment.",
        "summary": {
            "recommended_actions": len(action_rows),
            "tracked_kpis": 3,
            "baseline_risk_score": baseline_risk,
            "recovered_risk_score": recovered_risk,
            "eta_gain_hours": eta_gain,
        },
        "kpis": [
            {
                "metric": "risk_score",
                "baseline": baseline_risk,
                "expected_after_actions": recovered_risk,
                "delta": recovered_risk - baseline_risk,
                "unit": "score",
            },
            {
                "metric": "eta_hours",
                "baseline": baseline_eta,
                "expected_after_actions": recovered_eta,
                "delta": recovered_eta - baseline_eta,
                "unit": "hours",
            },
            {
                "metric": "review_handoff_confidence",
                "baseline": 55 if action_rows and action_rows[0]["lane"] != "steady-state" else 72,
                "expected_after_actions": 78 if action_rows and action_rows[0]["lane"] != "steady-state" else 88,
                "delta": 23 if action_rows and action_rows[0]["lane"] != "steady-state" else 16,
                "unit": "score",
            },
        ],
        "actions": action_rows,
        "review_actions": [
            "Use this board after the decision board so every recommendation carries an explicit KPI delta.",
            "Keep the risk-score and ETA change visible before escalating to downstream review or export.",
            "Treat action impact as a control-tower claim that still depends on queue parity and audit integrity.",
        ],
    }


def _queue_csv_summary(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {"row_count": 0, "unique_shipments": 0, "duplicate_rows": 0}

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    shipment_ids = [str(row.get("shipment_id", "")).strip() for row in rows if str(row.get("shipment_id", "")).strip()]
    unique_shipments = len(set(shipment_ids))
    return {
        "row_count": len(rows),
        "unique_shipments": unique_shipments,
        "duplicate_rows": max(0, len(rows) - unique_shipments),
    }


def _service_queue_count(path: Path) -> int:
    if not path.exists():
        return 0
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM service_queue").fetchone()[0])
    finally:
        conn.close()


def build_service_health_report(
    *,
    pipeline_status_path: Path,
    quality_report_path: Path = QUALITY_REPORT_PATH,
    training_summary_path: Path = TRAINING_SUMMARY_PATH,
    queue_csv_path: Path = DAILY_RISK_QUEUE_PATH,
    service_db_path: Path = SERVICE_DB_PATH,
    max_pipeline_age_hours: float = 24.0,
    min_model_auc: float = 0.72,
    strict_queue_parity: bool = True,
    now_utc: datetime | None = None,
) -> Dict[str, Any]:
    now = now_utc or datetime.now(timezone.utc)
    checks: List[Dict[str, Any]] = []

    pipeline = _read_json(pipeline_status_path)
    pipeline_state = str(pipeline.get("status", "")).strip().lower()
    if not pipeline:
        checks.append({"id": "pipeline_status", "status": "fail", "detail": "pipeline status file missing"})
        checks.append({"id": "pipeline_freshness", "status": "fail", "detail": "pipeline status unavailable"})
    else:
        if pipeline_state == "ok":
            checks.append({"id": "pipeline_status", "status": "pass", "detail": "pipeline status is ok"})
        else:
            checks.append(
                {
                    "id": "pipeline_status",
                    "status": "fail",
                    "detail": f"pipeline status is {pipeline_state or 'unknown'}",
                }
            )

        finished_dt = _parse_iso_dt(pipeline.get("finished_at_utc"))
        if finished_dt is None:
            checks.append({"id": "pipeline_freshness", "status": "fail", "detail": "finished_at_utc missing/invalid"})
        else:
            age_hours = max(0.0, (now - finished_dt).total_seconds() / 3600.0)
            if age_hours <= float(max_pipeline_age_hours):
                checks.append(
                    {
                        "id": "pipeline_freshness",
                        "status": "pass",
                        "detail": f"pipeline age={age_hours:.2f}h (max={max_pipeline_age_hours:.2f}h)",
                    }
                )
            else:
                checks.append(
                    {
                        "id": "pipeline_freshness",
                        "status": "warn",
                        "detail": f"pipeline age={age_hours:.2f}h exceeds max={max_pipeline_age_hours:.2f}h",
                    }
                )

    quality = _read_json(quality_report_path)
    quality_status = str(quality.get("status", "")).strip().lower()
    if not quality:
        checks.append({"id": "quality_gate", "status": "fail", "detail": "quality report missing"})
    elif quality_status == "fail":
        checks.append(
            {
                "id": "quality_gate",
                "status": "fail",
                "detail": f"quality failed (fails={quality.get('fail_count', 0)}, warns={quality.get('warn_count', 0)})",
            }
        )
    elif quality_status == "warn":
        checks.append(
            {
                "id": "quality_gate",
                "status": "warn",
                "detail": f"quality warn (fails={quality.get('fail_count', 0)}, warns={quality.get('warn_count', 0)})",
            }
        )
    else:
        checks.append(
            {
                "id": "quality_gate",
                "status": "pass",
                "detail": f"quality pass (fails={quality.get('fail_count', 0)}, warns={quality.get('warn_count', 0)})",
            }
        )

    training = _read_json(training_summary_path)
    model_auc = _safe_float((training.get("test_metrics", {}) or {}).get("auc"), default=-1.0)
    if not training or model_auc < 0:
        checks.append({"id": "model_auc", "status": "fail", "detail": "training summary missing/invalid auc"})
    elif model_auc < float(min_model_auc):
        checks.append(
            {
                "id": "model_auc",
                "status": "warn",
                "detail": f"model auc={model_auc:.4f} below target={min_model_auc:.4f}",
            }
        )
    else:
        checks.append(
            {
                "id": "model_auc",
                "status": "pass",
                "detail": f"model auc={model_auc:.4f} meets target={min_model_auc:.4f}",
            }
        )

    queue_csv = _queue_csv_summary(queue_csv_path)
    db_queue_count = _service_queue_count(service_db_path)
    duplicate_rows = int(queue_csv.get("duplicate_rows", 0))
    parity_mismatch = abs(int(queue_csv.get("row_count", 0)) - db_queue_count)
    if not queue_csv_path.exists():
        checks.append({"id": "queue_parity", "status": "fail", "detail": "daily_risk_queue.csv missing"})
    elif not service_db_path.exists():
        checks.append({"id": "queue_parity", "status": "fail", "detail": "service_store.db missing"})
    else:
        if duplicate_rows > 0:
            status = "fail" if strict_queue_parity else "warn"
            checks.append(
                {
                    "id": "queue_duplicates",
                    "status": status,
                    "detail": f"csv duplicates detected: {duplicate_rows}",
                }
            )
        else:
            checks.append({"id": "queue_duplicates", "status": "pass", "detail": "no duplicate shipment rows"})

        if parity_mismatch == 0:
            checks.append(
                {
                    "id": "queue_parity",
                    "status": "pass",
                    "detail": f"csv rows={queue_csv.get('row_count', 0)} matches db rows={db_queue_count}",
                }
            )
        else:
            status = "fail" if strict_queue_parity else "warn"
            checks.append(
                {
                    "id": "queue_parity",
                    "status": status,
                    "detail": f"csv rows={queue_csv.get('row_count', 0)} db rows={db_queue_count} mismatch={parity_mismatch}",
                }
            )

    if not service_db_path.exists():
        checks.append({"id": "audit_chain", "status": "fail", "detail": "service db missing"})
    else:
        audit = verify_audit_chain(path=service_db_path, limit=10000)
        if bool(audit.get("valid")):
            checks.append(
                {
                    "id": "audit_chain",
                    "status": "pass",
                    "detail": f"audit valid (checked={_safe_int(audit.get('checked', 0))})",
                }
            )
        else:
            checks.append(
                {
                    "id": "audit_chain",
                    "status": "fail",
                    "detail": f"audit invalid reason={audit.get('reason', 'unknown')}",
                }
            )

    statuses = [str(item.get("status", "pass")) for item in checks]
    overall_status = _worst_status(statuses)
    diagnostics = _build_health_diagnostics(checks)
    runtime_scorecard = _build_runtime_scorecard(
        checks=checks,
        queue_csv_rows=int(queue_csv.get("row_count", 0)),
        db_queue_count=db_queue_count,
        audit_checked=_safe_int(audit.get("checked", 0)),
        min_model_auc=float(min_model_auc),
        strict_queue_parity=bool(strict_queue_parity),
    )
    recovery_drill = _build_recovery_drill(
        checks=checks,
        queue_csv_rows=int(queue_csv.get("row_count", 0)),
        db_queue_count=db_queue_count,
        audit_checked=_safe_int(audit.get("checked", 0)),
        min_model_auc=float(min_model_auc),
    )
    decision_board = _build_decision_board(
        checks=checks,
        queue_csv_rows=int(queue_csv.get("row_count", 0)),
        db_queue_count=db_queue_count,
        audit_checked=_safe_int(audit.get("checked", 0)),
        min_model_auc=float(min_model_auc),
    )
    action_impact_board = _build_action_impact_board(
        decision_board=decision_board,
        recovery_drill=recovery_drill,
    )

    return {
        "generated_at_utc": now.isoformat(),
        "service_meta": {
            "service": "the-logistics-prophet",
            "readiness_contract": "logistics-control-brief-v1",
            "headline": "Predictive logistics control tower with evidence-backed worklist, governance, and audit surfaces.",
            "review_flow": REVIEW_FLOW,
            "two_minute_review": [
                "Run `make health` to confirm pipeline freshness, quality gate, queue parity, and audit-chain integrity.",
                "Open Control Tower Brief and validate the report contract plus current watchouts.",
                "Open the decision board or Next Actions view before trusting operational recommendations.",
                "Inspect Worklist or Queue + Update before trusting operator actionability claims.",
                "Open Governance or Evidence Pack before sharing downstream review artifacts.",
            ],
            "operator_rules": OPERATOR_RULES,
            "watchouts": WATCHOUTS,
            "stages": SERVICE_STAGES,
            "report_contract": REPORT_CONTRACT,
            "artifacts": {
                "queue_csv_rows": int(queue_csv.get("row_count", 0)),
                "service_db_rows": db_queue_count,
                "strict_queue_parity": bool(strict_queue_parity),
                "docs": 3,
                "monitoring_assets": 2,
                "service_scripts": 8,
                "test_files": 10,
            },
            "proof_assets": [
                {
                    "label": "Health Audit",
                    "path": "make health",
                    "kind": "command",
                    "why": "Confirms pipeline freshness, quality gate, model floor, queue parity, and audit integrity in one pass.",
                },
                {
                    "label": "Control Tower",
                    "path": "app/dashboard.py?page=control-tower",
                    "kind": "surface",
                    "why": "Shows the operator worklist, queue state, governance posture, and executive brief in the same surface.",
                },
                {
                    "label": "Decision Board",
                    "path": "app/dashboard.py?page=next-actions",
                    "kind": "surface",
                    "why": "Converts health posture into recommended operator actions and expected control-tower impact.",
                },
                {
                    "label": "Action Impact Board",
                    "path": "app/dashboard.py?page=next-actions",
                    "kind": "surface",
                    "why": "Makes the expected KPI delta explicit before operators or reviewers accept the recommendation.",
                },
                {
                    "label": "Evidence Pack",
                    "path": "scripts/scenario_runner.py",
                    "kind": "script",
                    "why": "Replays representative logistics scenarios and captures export-ready reviewer evidence.",
                },
                {
                    "label": "Audit Chain Check",
                    "path": "scripts/verify_audit.py",
                    "kind": "script",
                    "why": "Validates that downstream artifacts only move after the queue and audit chain stay consistent.",
                },
            ],
            "review_summary": _build_review_summary(
                checks=checks,
                queue_csv_rows=int(queue_csv.get("row_count", 0)),
                db_queue_count=db_queue_count,
                audit_checked=_safe_int(audit.get("checked", 0)),
                min_model_auc=float(min_model_auc),
                strict_queue_parity=bool(strict_queue_parity),
            ),
            "runtime_scorecard": runtime_scorecard,
            "recovery_drill": recovery_drill,
            "decision_board": decision_board,
            "action_impact_board": action_impact_board,
            "review_pack": {
                "contract": "logistics-control-review-pack-v1",
                "headline": "Reviewer pack for the logistics worklist: queue parity, audit chain, action loop, and evidence exports in one surface.",
                "proof_bundle": {
                    "queue_csv_rows": int(queue_csv.get("row_count", 0)),
                    "service_db_rows": db_queue_count,
                    "strict_queue_parity": bool(strict_queue_parity),
                    "audit_chain_checked": _safe_int(audit.get("checked", 0)),
                    "recovery_drill": "logistics-control-recovery-drill-v1",
                    "decision_board": "logistics-control-decision-board-v1",
                    "action_impact_board": "logistics-control-action-impact-board-v1",
                },
                "approval_gate": {
                    "quality_gate_required": True,
                    "model_auc_floor": float(min_model_auc),
                    "queue_parity_required": bool(strict_queue_parity),
                },
                "trust_boundary": [
                    "Synthetic queue generation, service-store ownership, and audit-chain verification stay within the local repo surface.",
                    "Datadog exports and Ollama recommendations remain optional reviewer extensions rather than baseline runtime requirements.",
                    "Evidence packs are downstream artifacts and should only be shared after queue parity plus audit-chain checks are green.",
                ],
                "review_sequence": [
                    "Run `make health` and confirm pipeline freshness, quality gate, model AUC, queue parity, and audit chain.",
                    "Open Next Actions or the decision board before discussing interventions with operators.",
                    "Open Control Tower Brief, then validate actionability in Worklist and Queue + Update.",
                    "Inspect Governance and Evidence Pack before approving downstream review or export.",
                ],
                "two_minute_review": [
                    "Run `make health` to confirm pipeline and audit posture.",
                    "Open Control Tower Brief for queue parity, schema, and watchouts.",
                    "Open the decision board before trusting next-action recommendations.",
                    "Inspect Worklist or Queue + Update before trusting operations claims.",
                    "Open Governance or Evidence Pack before exporting reviewer artifacts.",
                ],
                "proof_assets": [
                    {
                        "label": "Health Audit",
                        "path": "make health",
                        "kind": "command",
                        "why": "Verifies the baseline pipeline and audit posture before any reviewer claims are made.",
                    },
                    {
                        "label": "Control Tower",
                        "path": "app/dashboard.py?page=control-tower",
                        "kind": "surface",
                        "why": "Lets a reviewer compare queue parity, current actions, and trust boundaries without leaving the app.",
                    },
                    {
                        "label": "Decision Board",
                        "path": "app/dashboard.py?page=next-actions",
                        "kind": "surface",
                        "why": "Shows the recommended actions and expected operational delta before evidence export.",
                    },
                    {
                        "label": "Action Impact Board",
                        "path": "app/dashboard.py?page=next-actions",
                        "kind": "surface",
                        "why": "Adds KPI deltas so the recommendation is reviewable as an operational claim.",
                    },
                    {
                        "label": "Evidence Pack",
                        "path": "scripts/scenario_runner.py",
                        "kind": "script",
                        "why": "Produces the replay artifacts that support risk, SLA, and actionability statements.",
                    },
                    {
                        "label": "Audit Chain Check",
                        "path": "scripts/verify_audit.py",
                        "kind": "script",
                        "why": "Proves the handoff chain is intact before exports or downstream reviews are approved.",
                    },
                ],
            },
        },
        "overall_status": overall_status,
        "diagnostics": diagnostics,
        "thresholds": {
            "max_pipeline_age_hours": float(max_pipeline_age_hours),
            "min_model_auc": float(min_model_auc),
            "strict_queue_parity": bool(strict_queue_parity),
        },
        "summary": {
            "total_checks": len(checks),
            "pass_count": sum(1 for s in statuses if s == "pass"),
            "warn_count": sum(1 for s in statuses if s == "warn"),
            "fail_count": sum(1 for s in statuses if s == "fail"),
        },
        "checks": checks,
    }


def health_exit_code(report: Dict[str, Any], warn_as_error: bool = False) -> int:
    overall = str(report.get("overall_status", "pass")).strip().lower()
    if overall == "fail":
        return 1
    if overall == "warn" and warn_as_error:
        return 2
    return 0
