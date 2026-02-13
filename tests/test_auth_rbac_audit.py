from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import SERVICE_DB_PATH
from control_tower.service_store import (
    authenticate_user,
    fetch_queue,
    update_queue_action,
    verify_audit_chain,
)


class TestAuthRbacAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_pipeline.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Pipeline failed:\n{result.stdout}\n{result.stderr}")

    def test_default_users_can_authenticate(self) -> None:
        admin = authenticate_user("admin", "admin123!", path=SERVICE_DB_PATH)
        operator = authenticate_user("operator", "ops123!", path=SERVICE_DB_PATH)
        viewer = authenticate_user("viewer", "view123!", path=SERVICE_DB_PATH)

        self.assertIsNotNone(admin)
        self.assertIsNotNone(operator)
        self.assertIsNotNone(viewer)
        self.assertEqual(admin["role"], "admin")
        self.assertEqual(operator["role"], "operator")
        self.assertEqual(viewer["role"], "viewer")

    def test_viewer_cannot_update_queue(self) -> None:
        rows = fetch_queue(path=SERVICE_DB_PATH, limit=1)
        self.assertTrue(rows)
        shipment_id = str(rows[0]["shipment_id"])

        with self.assertRaises(PermissionError):
            update_queue_action(
                shipment_id=shipment_id,
                status="Investigating",
                owner="viewer-user",
                note="viewer should fail",
                eta_action_at="",
                actor="viewer",
                actor_role="viewer",
                path=SERVICE_DB_PATH,
            )

    def test_audit_chain_valid(self) -> None:
        result = verify_audit_chain(path=SERVICE_DB_PATH)
        self.assertTrue(result.get("valid"), f"Audit chain invalid: {result}")
        self.assertGreaterEqual(int(result.get("checked", 0)), 1)


if __name__ == "__main__":
    unittest.main()
