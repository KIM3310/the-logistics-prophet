from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import (
    DATADOG_SERIES_PAYLOAD_PATH,
    DAILY_RISK_QUEUE_PATH,
    MODEL_COMPARISON_PATH,
    MONITORING_DIR,
    MONITORING_METRICS_PATH,
    ONTOLOGY_DIR,
    OUTPUT_DIR,
    PIPELINE_STATUS_PATH,
    PROJECT_ROOT,
    QUALITY_REPORT_PATH,
    RDF_INSTANCE_PATH,
    SHAP_GLOBAL_PATH,
    SHAP_LOCAL_PATH,
    SPARQL_RESULTS_PATH,
    TRAINING_SUMMARY_PATH,
)
from .service_store import SERVICE_DB_PATH, verify_audit_chain


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_evidence_pack_bytes(
    *,
    include_instance_graph: bool = True,
    include_ops_report: bool = True,
) -> Tuple[str, bytes]:
    """
    Build a reviewer-friendly ZIP evidence pack with:
    - key artifacts (quality/model/SHAP/SPARQL/ops report)
    - specs + runbook + ontology + monitoring configs
    - a manifest containing SHA-256 hashes and audit chain verification
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    docs_dir = PROJECT_ROOT / "docs"
    specs_dir = PROJECT_ROOT / "specs"

    paths: List[Tuple[str, Path]] = [
        ("README.md", PROJECT_ROOT / "README.md"),
        ("docs/runbook.md", docs_dir / "runbook.md"),
        ("docs/postmortem_template.md", docs_dir / "postmortem_template.md"),
        ("docs/datadog_ingestion.md", docs_dir / "datadog_ingestion.md"),
        ("specs/FLAGSHIP_V3_SPEC.md", specs_dir / "FLAGSHIP_V3_SPEC.md"),
        ("specs/PRODUCT_SPEC.md", specs_dir / "PRODUCT_SPEC.md"),
        ("specs/SPECKIT_EXECUTION_PLAN.md", specs_dir / "SPECKIT_EXECUTION_PLAN.md"),
        ("ontology/supply_chain.ttl", ONTOLOGY_DIR / "supply_chain.ttl"),
        ("ontology/competency_questions.md", ONTOLOGY_DIR / "competency_questions.md"),
        ("monitoring/datadog_dashboard.json", MONITORING_DIR / "datadog_dashboard.json"),
        ("monitoring/monitors.yaml", MONITORING_DIR / "monitors.yaml"),
        ("data/output/monitoring_metrics.json", MONITORING_METRICS_PATH),
        ("data/output/model_comparison.json", MODEL_COMPARISON_PATH),
        ("data/output/training_summary.json", TRAINING_SUMMARY_PATH),
        ("data/output/data_quality_report.json", QUALITY_REPORT_PATH),
        ("data/output/sparql_results.json", SPARQL_RESULTS_PATH),
        ("data/output/shap_global_importance.csv", SHAP_GLOBAL_PATH),
        ("data/output/shap_local_explanations.csv", SHAP_LOCAL_PATH),
        ("data/output/daily_risk_queue.csv", DAILY_RISK_QUEUE_PATH),
        ("data/output/pipeline_status.json", PIPELINE_STATUS_PATH),
        ("data/output/datadog_series_payload.json", DATADOG_SERIES_PAYLOAD_PATH),
    ]

    if include_ops_report:
        paths.append(("data/output/ops_report.html", OUTPUT_DIR / "ops_report.html"))
    if include_instance_graph:
        paths.append(("data/semantic/instance_graph.ttl", RDF_INSTANCE_PATH))

    audit = verify_audit_chain(path=SERVICE_DB_PATH, limit=10000)
    manifest: Dict[str, Any] = {
        "generated_at_utc": now_utc,
        "project": "the-logistics-prophet",
        "audit_verification": audit,
        "files": [],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in paths:
            if not path.exists():
                continue
            data = path.read_bytes()
            zf.writestr(arcname, data)
            manifest["files"].append(
                {
                    "path": arcname,
                    "size_bytes": len(data),
                    "sha256": _sha256_hex(data),
                    "mtime_epoch": int(path.stat().st_mtime),
                }
            )

        manifest_bytes = json.dumps(manifest, ensure_ascii=True, indent=2).encode("utf-8")
        zf.writestr("evidence/manifest.json", manifest_bytes)

    filename = f"logistics-prophet-evidence-pack-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.zip"
    return filename, buf.getvalue()

