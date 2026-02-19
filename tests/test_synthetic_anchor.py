from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestSyntheticAnchor(unittest.TestCase):
    def test_generate_data_with_fixed_start_date(self) -> None:
        root = Path(__file__).resolve().parents[1]
        start_date = "2026-01-01"
        days = 5
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_data.py",
                "--seed",
                "42",
                "--days",
                str(days),
                "--orders-per-day",
                "40",
                "--start-date",
                start_date,
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"generate_data failed:\n{result.stdout}\n{result.stderr}")

        payload = json.loads(result.stdout)
        self.assertEqual(payload.get("start_date"), start_date)
        self.assertEqual(payload.get("end_date"), "2026-01-05")

        orders_path = root / "data" / "raw" / "orders.csv"
        with orders_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertGreater(len(rows), 0)
        first_order_date = str(rows[0].get("order_date", ""))
        self.assertEqual(first_order_date, start_date)


if __name__ == "__main__":
    unittest.main()

