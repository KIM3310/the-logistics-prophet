from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import SERVICE_DB_PATH
from control_tower.service_store import (
    authenticate_user,
    authenticate_user_with_status,
    fetch_queue,
    init_service_store,
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

    def test_audit_chain_reports_invalid_json_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="logistics_audit_chain_") as tmp:
            db_path = Path(tmp) / "service.db"
            init_service_store(path=db_path)
            admin = authenticate_user("admin", "admin123!", path=db_path)
            self.assertIsNotNone(admin)

            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    UPDATE service_activity_log
                    SET payload_json = ?
                    WHERE id = (SELECT MAX(id) FROM service_activity_log)
                    """,
                    ("{bad-json",),
                )
                conn.commit()
            finally:
                conn.close()

            result = verify_audit_chain(path=db_path)
            self.assertFalse(result.get("valid"))
            self.assertEqual(result.get("reason"), "invalid_payload_json")

    def test_audit_chain_reports_unexpected_genesis_reset(self) -> None:
        with tempfile.TemporaryDirectory(prefix="logistics_audit_chain_reset_") as tmp:
            db_path = Path(tmp) / "service.db"
            init_service_store(path=db_path)

            self.assertIsNotNone(authenticate_user("admin", "admin123!", path=db_path))
            self.assertIsNotNone(authenticate_user("operator", "ops123!", path=db_path))

            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    UPDATE service_activity_log
                    SET prev_hash = 'GENESIS'
                    WHERE id = (SELECT MAX(id) FROM service_activity_log)
                    """
                )
                conn.commit()
            finally:
                conn.close()

            result = verify_audit_chain(path=db_path)
            self.assertFalse(result.get("valid"))
            self.assertEqual(result.get("reason"), "unexpected_genesis_reset")

    def test_auth_lockout_policy_enforced(self) -> None:
        with tempfile.TemporaryDirectory(prefix="logistics_auth_lock_") as tmp:
            db_path = Path(tmp) / "service.db"
            old_attempt = os.environ.get("LP_AUTH_MAX_FAILED_ATTEMPTS")
            old_lock = os.environ.get("LP_AUTH_LOCK_MINUTES")
            try:
                os.environ["LP_AUTH_MAX_FAILED_ATTEMPTS"] = "2"
                os.environ["LP_AUTH_LOCK_MINUTES"] = "1"
                init_service_store(path=db_path)

                first = authenticate_user_with_status("admin", "wrong-password", path=db_path)
                self.assertFalse(bool(first.get("ok")))
                self.assertEqual(first.get("reason"), "invalid_credentials")
                self.assertEqual(int(first.get("attempts_remaining", -1)), 1)

                second = authenticate_user_with_status("admin", "wrong-password", path=db_path)
                self.assertFalse(bool(second.get("ok")))
                self.assertEqual(second.get("reason"), "account_locked")
                self.assertEqual(int(second.get("attempts_remaining", -1)), 0)
                self.assertTrue(str(second.get("locked_until", "")).strip())

                blocked = authenticate_user_with_status("admin", "admin123!", path=db_path)
                self.assertFalse(bool(blocked.get("ok")))
                self.assertEqual(blocked.get("reason"), "account_locked")
            finally:
                if old_attempt is None:
                    os.environ.pop("LP_AUTH_MAX_FAILED_ATTEMPTS", None)
                else:
                    os.environ["LP_AUTH_MAX_FAILED_ATTEMPTS"] = old_attempt
                if old_lock is None:
                    os.environ.pop("LP_AUTH_LOCK_MINUTES", None)
                else:
                    os.environ["LP_AUTH_LOCK_MINUTES"] = old_lock


if __name__ == "__main__":
    unittest.main()
