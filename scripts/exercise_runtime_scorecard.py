from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import PIPELINE_STATUS_PATH
from control_tower.service_health import build_service_health_report


def main() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pipeline failed\n{result.stdout}\n{result.stderr}")

    report = build_service_health_report(
        pipeline_status_path=PIPELINE_STATUS_PATH,
        max_pipeline_age_hours=24.0,
        min_model_auc=0.50,
        strict_queue_parity=True,
    )
    scorecard = report["service_meta"]["runtime_scorecard"]
    print(
        json.dumps(
            {
                "contract": scorecard["contract"],
                "summary": scorecard["summary"],
                "top_failing_checks": scorecard["top_failing_checks"],
                "top_warning_checks": scorecard["top_warning_checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
