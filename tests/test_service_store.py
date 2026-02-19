from __future__ import annotations

import sqlite3
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import SERVICE_DB_PATH
from control_tower.service_store import (
    allowed_next_statuses,
    bulk_update_queue_actions,
    fetch_activity,
    fetch_ops_health,
    fetch_queue,
    fetch_service_core_snapshot,
    fetch_service_core_worklist,
    fetch_workflow_sla_snapshot,
    incident_recommendations_from_metrics,
    list_incidents,
    list_pipeline_runs,
    list_recent_activity,
    upsert_incident,
    upsert_incident_from_recommendation,
    upsert_queue_rows,
    update_queue_action,
)


class TestServiceStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SERVICE_DB_PATH.exists():
            result = subprocess.run(
                [sys.executable, "scripts/run_pipeline.py"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Pipeline failed:\n{result.stdout}\n{result.stderr}")

    def test_queue_materialized(self) -> None:
        rows = fetch_queue(path=SERVICE_DB_PATH, limit=10)
        self.assertGreater(len(rows), 0)

    def test_action_update_writes_activity(self) -> None:
        rows = fetch_queue(path=SERVICE_DB_PATH, limit=1)
        self.assertTrue(rows)
        shipment_id = str(rows[0]["shipment_id"])

        update_queue_action(
            shipment_id=shipment_id,
            status="Investigating",
            owner="ops-user",
            note="service-store-test",
            eta_action_at="",
            actor="test-suite",
            actor_role="operator",
            path=SERVICE_DB_PATH,
        )

        activity = fetch_activity(shipment_id=shipment_id, path=SERVICE_DB_PATH, limit=5)
        self.assertTrue(activity)
        self.assertIn("test-suite", [str(row.get("actor", "")) for row in activity])

    def test_status_alias_update_and_transition_order(self) -> None:
        rows = fetch_queue(path=SERVICE_DB_PATH, limit=1)
        self.assertTrue(rows)
        shipment_id = str(rows[0]["shipment_id"])

        update_queue_action(
            shipment_id=shipment_id,
            status="New",
            owner="ops-admin",
            note="normalize for alias test",
            eta_action_at="",
            actor="test-suite",
            actor_role="admin",
            path=SERVICE_DB_PATH,
        )

        update_queue_action(
            shipment_id=shipment_id,
            status="check",
            owner="ops-user",
            note="alias check should map to Investigating",
            eta_action_at="",
            actor="test-suite",
            actor_role="operator",
            path=SERVICE_DB_PATH,
        )

        refreshed = fetch_queue(path=SERVICE_DB_PATH, limit=500)
        status_map = {str(row["shipment_id"]): str(row["status"]) for row in refreshed}
        self.assertEqual(status_map.get(shipment_id), "Investigating")

        self.assertEqual(
            allowed_next_statuses("New", actor_role="operator"),
            ["New", "Investigating", "Dismissed"],
        )
        self.assertEqual(
            allowed_next_statuses("start", actor_role="operator"),
            ["New", "Investigating", "Dismissed"],
        )
        self.assertEqual(
            allowed_next_statuses("check", actor_role="admin"),
            ["New", "Investigating", "Mitigating", "Resolved", "Dismissed"],
        )

    def test_eta_validation_rejects_invalid_timestamp(self) -> None:
        rows = fetch_queue(path=SERVICE_DB_PATH, limit=1)
        self.assertTrue(rows)
        shipment_id = str(rows[0]["shipment_id"])
        with self.assertRaises(ValueError):
            update_queue_action(
                shipment_id=shipment_id,
                status="Investigating",
                owner="ops-user",
                note="invalid eta test",
                eta_action_at="not-a-timestamp",
                actor="test-suite",
                actor_role="operator",
                path=SERVICE_DB_PATH,
            )

    def test_bulk_update_and_ops_health(self) -> None:
        rows = fetch_queue(path=SERVICE_DB_PATH, limit=3)
        self.assertGreaterEqual(len(rows), 3)
        shipment_ids = [str(row["shipment_id"]) for row in rows]
        past_eta = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()

        for shipment_id in shipment_ids:
            update_queue_action(
                shipment_id=shipment_id,
                status="New",
                owner="ops-admin",
                note="normalize status for bulk test",
                eta_action_at="",
                actor="test-suite",
                actor_role="admin",
                path=SERVICE_DB_PATH,
            )

        result = bulk_update_queue_actions(
            shipment_ids=shipment_ids,
            status="Investigating",
            owner="ops-bulk",
            note="bulk-update-test",
            eta_action_at=past_eta,
            actor="test-suite",
            actor_role="operator",
            path=SERVICE_DB_PATH,
        )
        self.assertEqual(int(result.get("updated", 0)), len(shipment_ids))
        self.assertEqual(result.get("missing"), [])
        self.assertEqual(result.get("invalid_transitions"), [])

        refreshed = fetch_queue(path=SERVICE_DB_PATH, limit=100)
        refreshed_map = {str(row["shipment_id"]): row for row in refreshed}
        for shipment_id in shipment_ids:
            self.assertIn(shipment_id, refreshed_map)
            self.assertEqual(str(refreshed_map[shipment_id]["status"]), "Investigating")
            self.assertEqual(str(refreshed_map[shipment_id]["owner"]), "ops-bulk")

        health = fetch_ops_health(path=SERVICE_DB_PATH)
        self.assertGreaterEqual(int(health.get("overdue_eta", 0)), 1)
        backlog = health.get("owner_backlog", [])
        owners = {str(item.get("owner")): int(item.get("count", 0)) for item in backlog if isinstance(item, dict)}
        self.assertGreaterEqual(owners.get("ops-bulk", 0), len(shipment_ids))

    def test_transition_guardrail_and_admin_override(self) -> None:
        rows = fetch_queue(path=SERVICE_DB_PATH, limit=1)
        self.assertTrue(rows)
        shipment_id = str(rows[0]["shipment_id"])

        update_queue_action(
            shipment_id=shipment_id,
            status="Resolved",
            owner="ops-admin",
            note="force resolve for guardrail test",
            eta_action_at="",
            actor="test-suite",
            actor_role="admin",
            path=SERVICE_DB_PATH,
        )

        with self.assertRaises(ValueError):
            update_queue_action(
                shipment_id=shipment_id,
                status="New",
                owner="ops-user",
                note="operator should not reopen",
                eta_action_at="",
                actor="test-suite",
                actor_role="operator",
                path=SERVICE_DB_PATH,
            )

        update_queue_action(
            shipment_id=shipment_id,
            status="New",
            owner="ops-admin",
            note="admin override reopen",
            eta_action_at="",
            actor="test-suite",
            actor_role="admin",
            path=SERVICE_DB_PATH,
        )

    def test_bulk_transition_partial_with_invalid_rows(self) -> None:
        rows = fetch_queue(path=SERVICE_DB_PATH, limit=2)
        self.assertGreaterEqual(len(rows), 2)
        shipment_a = str(rows[0]["shipment_id"])
        shipment_b = str(rows[1]["shipment_id"])

        update_queue_action(
            shipment_id=shipment_a,
            status="Resolved",
            owner="ops-admin",
            note="setup resolved",
            eta_action_at="",
            actor="test-suite",
            actor_role="admin",
            path=SERVICE_DB_PATH,
        )
        update_queue_action(
            shipment_id=shipment_b,
            status="New",
            owner="ops-admin",
            note="setup new",
            eta_action_at="",
            actor="test-suite",
            actor_role="admin",
            path=SERVICE_DB_PATH,
        )

        result = bulk_update_queue_actions(
            shipment_ids=[shipment_a, shipment_b],
            status="Investigating",
            owner="ops-bulk",
            note="bulk guardrail",
            eta_action_at="",
            actor="test-suite",
            actor_role="operator",
            path=SERVICE_DB_PATH,
        )

        self.assertEqual(int(result.get("updated", 0)), 1)
        invalid = result.get("invalid_transitions", [])
        self.assertTrue(invalid)
        invalid_ids = {str(item.get("shipment_id", "")) for item in invalid if isinstance(item, dict)}
        self.assertIn(shipment_a, invalid_ids)

    def test_incident_recommendation_rules(self) -> None:
        queue_summary = {
            "critical_open": 2,
            "unresolved": 44,
        }
        ops_health = {
            "overdue_eta": 7,
            "stale_24h": 9,
            "critical_unassigned": 1,
            "avg_unresolved_age_hours": 12.2,
        }
        recommendations = incident_recommendations_from_metrics(queue_summary=queue_summary, ops_health=ops_health, max_items=5)
        self.assertGreaterEqual(len(recommendations), 3)
        self.assertIn("SEV-1", [str(r.get("severity")) for r in recommendations])

    def test_incident_recommendation_deduplication(self) -> None:
        unique_rule = f"test-rule-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        recommendation = {
            "rule_id": unique_rule,
            "severity": "SEV-2",
            "title": "Test recommendation dedup",
            "description": "Ensure same rule updates existing incident instead of creating duplicates.",
        }

        first = upsert_incident_from_recommendation(
            recommendation=recommendation,
            owner="ops-reco",
            actor="test-suite",
            actor_role="operator",
            path=SERVICE_DB_PATH,
        )
        second = upsert_incident_from_recommendation(
            recommendation=recommendation,
            owner="ops-reco",
            actor="test-suite",
            actor_role="operator",
            path=SERVICE_DB_PATH,
        )

        self.assertFalse(bool(first.get("deduplicated")))
        self.assertTrue(bool(second.get("deduplicated")))
        self.assertEqual(str(first.get("incident_id")), str(second.get("incident_id")))

        conn = sqlite3.connect(SERVICE_DB_PATH)
        try:
            cnt = conn.execute("SELECT COUNT(*) FROM service_incidents WHERE source_rule = ?", (unique_rule,)).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(int(cnt), 1)

    def test_upsert_incident_validation(self) -> None:
        with self.assertRaises(ValueError):
            upsert_incident(
                title="Validation test",
                severity="SEV-X",
                description="invalid severity should fail",
                status="Open",
                owner="ops-user",
                actor="test-suite",
                actor_role="operator",
                path=SERVICE_DB_PATH,
            )

        with self.assertRaises(ValueError):
            upsert_incident(
                title="Validation test",
                severity="SEV-2",
                description="invalid status should fail",
                status="Invalid",
                owner="ops-user",
                actor="test-suite",
                actor_role="operator",
                path=SERVICE_DB_PATH,
            )

    def test_service_core_snapshot_structure(self) -> None:
        snapshot = fetch_service_core_snapshot(path=SERVICE_DB_PATH, candidate_limit=8)
        self.assertIn("stage_backlog", snapshot)
        self.assertIn("escalation_candidates", snapshot)
        self.assertIn("driver_hotspots", snapshot)
        self.assertTrue(isinstance(snapshot.get("stage_backlog"), list))
        if snapshot.get("stage_backlog"):
            first = snapshot["stage_backlog"][0]
            self.assertIn("stage", first)
            self.assertIn("label", first)
            self.assertIn("count", first)

    def test_service_core_worklist_structure(self) -> None:
        snapshot = fetch_service_core_worklist(path=SERVICE_DB_PATH, per_stage_limit=3)
        self.assertIn("open_total", snapshot)
        self.assertIn("stages", snapshot)
        self.assertIn("top_items", snapshot)
        self.assertTrue(isinstance(snapshot.get("stages"), list))
        stages = snapshot.get("stages", [])
        self.assertEqual(len(stages), 3)
        for stage in stages:
            self.assertIn("stage", stage)
            self.assertIn("count", stage)
            self.assertIn("items", stage)
            items = stage.get("items", [])
            self.assertLessEqual(len(items), 3)
            for item in items:
                self.assertIn("shipment_id", item)
                self.assertIn("urgency_score", item)
                self.assertIn("next_step", item)

    def test_workflow_sla_snapshot_structure(self) -> None:
        snapshot = fetch_workflow_sla_snapshot(path=SERVICE_DB_PATH, candidate_limit=8)
        self.assertIn("unresolved_total", snapshot)
        self.assertIn("breached_total", snapshot)
        self.assertIn("breach_rate_pct", snapshot)
        self.assertIn("age_buckets", snapshot)
        self.assertIn("stage_sla", snapshot)
        self.assertIn("breached_candidates", snapshot)

    def test_query_limits_are_safely_bounded(self) -> None:
        queue_rows = fetch_queue(path=SERVICE_DB_PATH, limit=-10)
        self.assertGreater(len(queue_rows), 0)

        recent_activity = list_recent_activity(path=SERVICE_DB_PATH, limit=999999)
        self.assertLessEqual(len(recent_activity), 1000)

        incidents = list_incidents(path=SERVICE_DB_PATH, limit=0)
        self.assertLessEqual(len(incidents), 500)

        runs = list_pipeline_runs(path=SERVICE_DB_PATH, limit=-1)
        self.assertLessEqual(len(runs), 500)

    def test_zz_queue_sync_prunes_stale_rows(self) -> None:
        queue = fetch_queue(path=SERVICE_DB_PATH, limit=1000)
        self.assertGreaterEqual(len(queue), 3)

        keep = queue[:2]
        payload = []
        for row in keep:
            payload.append(
                {
                    "shipment_id": str(row.get("shipment_id", "")),
                    "ship_date": str(row.get("ship_date", "")),
                    "order_id": str(row.get("order_id", "")),
                    "risk_score": float(row.get("risk_score", 0.0) or 0.0),
                    "risk_band": str(row.get("risk_band", "Low")),
                    "prediction": int(float(row.get("prediction", 0) or 0)),
                    "key_driver": str(row.get("key_driver", "")),
                    "driver_2": str(row.get("driver_2", "")),
                    "driver_3": str(row.get("driver_3", "")),
                    "recommended_action": str(row.get("recommended_action", "")),
                }
            )

        synced = upsert_queue_rows(payload, path=SERVICE_DB_PATH)
        self.assertEqual(synced, 2)

        refreshed = fetch_queue(path=SERVICE_DB_PATH, limit=1000)
        shipment_ids = {str(row.get("shipment_id", "")) for row in refreshed}
        expected_ids = {str(row.get("shipment_id", "")) for row in keep}
        self.assertEqual(shipment_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
