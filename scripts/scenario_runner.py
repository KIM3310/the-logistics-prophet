#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import (  # noqa: E402
    MODEL_COMPARISON_PATH,
    MONITORING_METRICS_PATH,
    PIPELINE_STATUS_PATH,
    QUALITY_REPORT_PATH,
    RDF_INSTANCE_PATH,
    SHAP_GLOBAL_PATH,
    SHAP_LOCAL_PATH,
    SPARQL_RESULTS_PATH,
)
from control_tower.evidence_pack import build_evidence_pack_bytes  # noqa: E402
from control_tower.service_store import (  # noqa: E402
    SERVICE_DB_PATH,
    fetch_queue_summary,
    fetch_service_core_snapshot,
    fetch_service_core_worklist,
    fetch_workflow_sla_snapshot,
    list_incidents,
    verify_audit_chain,
)


@dataclass
class Verdict:
    pipeline_ready: bool
    quality_status: str
    audit_valid: bool


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_pipeline(force: bool = False) -> bool:
    required = [
        MONITORING_METRICS_PATH,
        QUALITY_REPORT_PATH,
        SPARQL_RESULTS_PATH,
        SHAP_GLOBAL_PATH,
        SHAP_LOCAL_PATH,
        RDF_INSTANCE_PATH,
        SERVICE_DB_PATH,
    ]
    need = force or any(not p.exists() for p in required)
    if not need:
        return True

    result = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py"],
        cwd=ROOT,
        text=True,
    )
    return result.returncode == 0


def _render_md(
    *,
    started_at_utc: str,
    metrics: Dict,
    quality: Dict,
    pipeline_status: Dict,
    queue_summary: Dict,
    core_snapshot: Dict,
    worklist: Dict,
    sla: Dict,
    incidents: List[Dict],
    audit: Dict,
    evidence_zip_name: str,
) -> str:
    latest = metrics.get("kpi_latest", {}) if isinstance(metrics, dict) else {}
    model_metrics = metrics.get("model_test_metrics", {}) if isinstance(metrics, dict) else {}
    quality_gate = metrics.get("quality_gate", {}) if isinstance(metrics, dict) else {}
    service_queue = metrics.get("service_queue", {}) if isinstance(metrics, dict) else {}

    def fmt_pct(x: object) -> str:
        try:
            return f"{float(x) * 100:.1f}%"
        except Exception:
            return "-"

    def fmt_num(x: object, digits: int = 2) -> str:
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return "-"

    q_status = str(quality_gate.get("status", quality.get("status", "unknown")) or "unknown")
    audit_ok = bool(audit.get("valid", False))

    lines: List[str] = []
    lines.append("# The Logistics Prophet - Scenario Runner Report")
    lines.append("")
    lines.append(f"- Started at (UTC): `{started_at_utc}`")
    lines.append(f"- Evidence pack: `{evidence_zip_name}`")
    lines.append("")

    lines.append("## Executive Snapshot")
    lines.append(f"- Latest ship date: `{latest.get('ship_date', '-')}`")
    lines.append(f"- On-time rate: **{fmt_pct(latest.get('on_time_rate'))}**")
    lines.append(f"- Avg delay hours: **{fmt_num(latest.get('avg_delay_hours'))}**")
    lines.append(f"- SLA breaches: **{latest.get('sla_breach_count', '-') }**")
    lines.append(f"- Model AUC: **{fmt_num(model_metrics.get('auc'), 4)}**")
    lines.append(f"- Queue unresolved: **{service_queue.get('unresolved', queue_summary.get('unresolved', '-'))}**")
    lines.append(f"- Critical open: **{service_queue.get('critical_open', queue_summary.get('critical_open', '-'))}**")
    lines.append("")

    lines.append("## Quality Gate")
    lines.append(f"- Status: `{q_status}` (fails={quality_gate.get('fail_count', quality.get('fail_count', 0))}, warns={quality_gate.get('warn_count', quality.get('warn_count', 0))})")
    if isinstance(quality, dict) and quality.get("failed_checks"):
        lines.append("- Failed checks (top):")
        for item in list(quality.get("failed_checks", []))[:8]:
            lines.append(f"  - {item}")
    lines.append("")

    lines.append("## Ops Workflow (Start / Check / Fix / Done)")
    stage_backlog = core_snapshot.get("stage_backlog", []) if isinstance(core_snapshot, dict) else []
    if stage_backlog:
        lines.append("| Step | Count | Share % |")
        lines.append("| --- | ---: | ---: |")
        for row in stage_backlog[:6]:
            lines.append(f"| {row.get('stage','')} | {row.get('count','')} | {row.get('share_pct','')} |")
        lines.append("")

    lines.append("## Worklist (Top Actions)")
    stages = worklist.get("stages", []) if isinstance(worklist, dict) else []
    stage_map = {str(item.get("stage")): item for item in stages if isinstance(item, dict)}
    for step in ["Start", "Check", "Fix"]:
        entry = stage_map.get(step, {})
        items = entry.get("items", []) if isinstance(entry, dict) else []
        lines.append(f"### {step}")
        if not items:
            lines.append("- (no items)")
            continue
        for item in items[:5]:
            lines.append(
                f"- `{item.get('shipment_id','')}` risk=`{item.get('risk_band','')}` urgency=`{item.get('urgency_score','')}` next=`{item.get('next_step','')}` why=`{item.get('why','')}`"
            )
        lines.append("")

    lines.append("## SLA / Health Signals")
    if isinstance(sla, dict):
        lines.append(f"- Past ETA: `{sla.get('past_eta', '-')}`")
        lines.append(f"- Stale 24h+: `{sla.get('stale_24h', '-')}`")
        lines.append(f"- Critical unassigned: `{sla.get('critical_unassigned', '-')}`")
        lines.append(f"- Owner backlog: `{sla.get('owner_backlog', '-')}`")
    lines.append("")

    lines.append("## Governance")
    lines.append(f"- Audit chain valid: **{audit_ok}** (checked={audit.get('checked', '-')})")
    lines.append(f"- Latest hash: `{str(audit.get('latest_hash','') or '')[:18]}...`" if audit.get("latest_hash") else "- Latest hash: `-`")
    open_incidents = [row for row in incidents if str(row.get("status", "")).lower() in {"open", "monitoring"}]
    lines.append(f"- Open/Monitoring incidents: `{len(open_incidents)}` (total={len(incidents)})")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("```bash")
    lines.append("make demo-local-open")
    lines.append("python3 scripts/run_pipeline.py")
    lines.append("python3 scripts/verify_audit.py")
    lines.append("```")
    lines.append("")

    if pipeline_status:
        run_id = str(pipeline_status.get("run_id", "") or "").strip()
        status = str(pipeline_status.get("status", "") or "").strip()
        if run_id or status:
            lines.append("## Pipeline Run")
            lines.append(f"- run_id: `{run_id or '-'}`")
            lines.append(f"- status: `{status or '-'}`")
            lines.append(f"- finished_at_utc: `{pipeline_status.get('finished_at_utc','-')}`")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a repeatable end-to-end validation and export a report + evidence pack.")
    parser.add_argument("--force", action="store_true", help="Force pipeline execution before exporting.")
    parser.add_argument("--out-dir", default="", help="Output directory (default: data/output/scenario_runs/<timestamp>).")
    parser.add_argument("--no-graph", action="store_true", help="Exclude RDF instance graph from evidence pack.")
    parser.add_argument("--no-ops-report", action="store_true", help="Exclude ops HTML report from evidence pack.")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (ROOT / "data" / "output" / "scenario_runs" / started_at.replace(":", ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = _ensure_pipeline(force=bool(args.force))
    metrics = _read_json(MONITORING_METRICS_PATH)
    quality = _read_json(QUALITY_REPORT_PATH)
    pipeline_status = _read_json(PIPELINE_STATUS_PATH)
    _ = _read_json(MODEL_COMPARISON_PATH)

    queue_summary = fetch_queue_summary(SERVICE_DB_PATH)
    core_snapshot = fetch_service_core_snapshot(SERVICE_DB_PATH)
    worklist = fetch_service_core_worklist(SERVICE_DB_PATH, per_stage_limit=6)
    sla = fetch_workflow_sla_snapshot(SERVICE_DB_PATH)
    incidents = list_incidents(SERVICE_DB_PATH, limit=50)
    audit = verify_audit_chain(path=SERVICE_DB_PATH, limit=10000)

    include_graph = not bool(args.no_graph)
    include_ops = not bool(args.no_ops_report)
    evidence_name, evidence_bytes = build_evidence_pack_bytes(
        include_instance_graph=include_graph,
        include_ops_report=include_ops,
    )
    evidence_path = out_dir / evidence_name
    evidence_path.write_bytes(evidence_bytes)

    report_md = _render_md(
        started_at_utc=started_at,
        metrics=metrics,
        quality=quality,
        pipeline_status=pipeline_status,
        queue_summary=queue_summary,
        core_snapshot=core_snapshot,
        worklist=worklist,
        sla=sla,
        incidents=incidents,
        audit=audit,
        evidence_zip_name=evidence_path.name,
    )
    report_path = out_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")

    quality_status = str(metrics.get("quality_gate", {}).get("status", quality.get("status", "unknown")) or "unknown")
    verdict = Verdict(pipeline_ready=ok, quality_status=quality_status, audit_valid=bool(audit.get("valid", False)))
    (out_dir / "verdict.json").write_text(
        json.dumps(
            {
                "pipeline_ready": verdict.pipeline_ready,
                "quality_status": verdict.quality_status,
                "audit_valid": verdict.audit_valid,
                "out_dir": str(out_dir),
                "report": str(report_path),
                "evidence_pack": str(evidence_path),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[scenario] out_dir={out_dir}")
    print(f"[scenario] report={report_path}")
    print(f"[scenario] evidence={evidence_path}")
    print(f"[scenario] pipeline_ready={verdict.pipeline_ready} quality={verdict.quality_status} audit_valid={verdict.audit_valid}")

    # Non-fatal: still export artifacts, but return non-zero for CI/automation.
    if not verdict.pipeline_ready:
        return 2
    if verdict.quality_status == "fail":
        return 3
    if not verdict.audit_valid:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

