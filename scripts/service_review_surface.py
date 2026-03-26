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
from control_tower.service_health import build_service_health_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a reviewer-facing section from the logistics health report."
    )
    parser.add_argument(
        "--section",
        choices=[
            "compact_summary",
            "review_summary",
            "runtime_scorecard",
            "recovery_drill",
            "decision_board",
            "action_impact_board",
            "review_pack",
        ],
        default="review_pack",
    )
    args = parser.parse_args()

    report = build_service_health_report(
        pipeline_status_path=PIPELINE_STATUS_PATH,
        max_pipeline_age_hours=24.0,
        min_model_auc=0.71,
        strict_queue_parity=True,
    )
    section = report["service_meta"][args.section]
    print(json.dumps(section, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
