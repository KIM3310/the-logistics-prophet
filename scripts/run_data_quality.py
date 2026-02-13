#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.quality import run_data_quality_checks


def main() -> None:
    result = run_data_quality_checks()
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if result.get("status") == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
