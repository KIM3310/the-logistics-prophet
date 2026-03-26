from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from control_tower.queue_sync import load_queue_rows, sync_queue_from_csv
from control_tower.service_store import fetch_queue, init_service_store


class QueueSyncTests(unittest.TestCase):
    def _write_queue_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "shipment_id",
                    "ship_date",
                    "order_id",
                    "risk_score",
                    "risk_band",
                    "prediction",
                    "key_driver",
                    "driver_2",
                    "driver_3",
                    "recommended_action",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_sync_queue_from_csv_upserts_and_prunes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            queue_csv = tmp_path / "daily_risk_queue.csv"
            service_db = tmp_path / "service_store.db"

            init_service_store(service_db)

            first_rows = [
                {
                    "shipment_id": "S-001",
                    "ship_date": "2026-03-15",
                    "order_id": "O-001",
                    "risk_score": 0.91,
                    "risk_band": "Critical",
                    "prediction": 1,
                    "key_driver": "weather_severity",
                    "driver_2": "distance_km",
                    "driver_3": "carrier_reliability_score",
                    "recommended_action": "Escalate",
                },
                {
                    "shipment_id": "S-002",
                    "ship_date": "2026-03-15",
                    "order_id": "O-002",
                    "risk_score": 0.76,
                    "risk_band": "High",
                    "prediction": 1,
                    "key_driver": "warehouse_load_pct",
                    "driver_2": "distance_km",
                    "driver_3": "peak_flag",
                    "recommended_action": "Monitor",
                },
            ]

            self._write_queue_csv(queue_csv, first_rows)
            synced = sync_queue_from_csv(
                queue_csv_path=queue_csv, service_db_path=service_db
            )
            self.assertEqual(synced, 2)
            self.assertEqual(len(fetch_queue(path=service_db, limit=100)), 2)

            second_rows = [first_rows[0]]
            self._write_queue_csv(queue_csv, second_rows)
            synced = sync_queue_from_csv(
                queue_csv_path=queue_csv, service_db_path=service_db
            )
            self.assertEqual(synced, 1)
            queue = fetch_queue(path=service_db, limit=100)
            self.assertEqual(len(queue), 1)
            self.assertEqual(queue[0]["shipment_id"], "S-001")

    def test_load_queue_rows_requires_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue_csv = Path(tmp) / "daily_risk_queue.csv"
            queue_csv.write_text(
                "shipment_id,ship_date\nS-001,2026-03-15\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_queue_rows(queue_csv)


if __name__ == "__main__":
    unittest.main()
