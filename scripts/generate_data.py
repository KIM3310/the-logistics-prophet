#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.synthetic_data import generate_synthetic_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic supply-chain data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--orders-per-day", type=int, default=120)
    parser.add_argument("--start-date", default="", help="Optional fixed start date (YYYY-MM-DD) for deterministic runs.")
    args = parser.parse_args()

    start_date = None
    raw_start_date = str(args.start_date or "").strip()
    if raw_start_date:
        try:
            start_date = date.fromisoformat(raw_start_date)
        except ValueError as exc:
            raise SystemExit("Invalid --start-date. Use YYYY-MM-DD.") from exc

    summary = generate_synthetic_data(
        seed=args.seed,
        days=args.days,
        orders_per_day=args.orders_per_day,
        start_date=start_date,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
