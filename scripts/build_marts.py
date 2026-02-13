#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.sqlite_pipeline import build_sqlite_marts


def main() -> None:
    counts = build_sqlite_marts()
    print(json.dumps(counts, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
