#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import PIPELINE_STATUS_PATH  # noqa: E402
from control_tower.service_health import build_service_health_report, health_exit_code  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit service readiness and operational health contracts.")
    parser.add_argument("--max-pipeline-age-hours", type=float, default=24.0)
    parser.add_argument("--min-model-auc", type=float, default=0.72)
    parser.add_argument("--allow-queue-mismatch", action="store_true")
    parser.add_argument("--warn-as-error", action="store_true")
    parser.add_argument("--json-out", default="", help="Optional path to write JSON report.")
    args = parser.parse_args()

    report = build_service_health_report(
        pipeline_status_path=PIPELINE_STATUS_PATH,
        max_pipeline_age_hours=float(args.max_pipeline_age_hours),
        min_model_auc=float(args.min_model_auc),
        strict_queue_parity=not bool(args.allow_queue_mismatch),
    )

    output = json.dumps(report, ensure_ascii=True, indent=2)
    print(output)

    json_out = str(args.json_out or "").strip()
    if json_out:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")

    return health_exit_code(report, warn_as_error=bool(args.warn_as_error))


if __name__ == "__main__":
    raise SystemExit(main())

