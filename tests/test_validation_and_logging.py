"""Tests for input validation, structured logging, and error handling improvements."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import setup_logging
from control_tower.data_access import fetch_shipment_feature_row
from control_tower.service_store import (
    INCIDENT_SEVERITIES,
    INCIDENT_STATUSES,
    normalize_status_input,
    upsert_incident,
    init_service_store,
)
from control_tower.queue_sync import load_queue_rows


class TestSetupLogging(unittest.TestCase):
    """Verify that setup_logging configures the control_tower logger."""

    def test_setup_logging_creates_handler(self) -> None:
        setup_logging(level=logging.DEBUG)
        logger = logging.getLogger("control_tower")
        self.assertTrue(
            logger.handlers, "Expected at least one handler on control_tower logger"
        )
        self.assertEqual(logger.level, logging.DEBUG)

    def test_setup_logging_idempotent(self) -> None:
        setup_logging(level=logging.INFO)
        count_before = len(logging.getLogger("control_tower").handlers)
        setup_logging(level=logging.INFO)
        count_after = len(logging.getLogger("control_tower").handlers)
        self.assertEqual(
            count_before, count_after, "setup_logging should not add duplicate handlers"
        )


class TestStatusValidation(unittest.TestCase):
    """Verify normalize_status_input handles valid/invalid inputs."""

    def test_canonical_statuses_accepted(self) -> None:
        for status in ["New", "Investigating", "Mitigating", "Resolved", "Dismissed"]:
            self.assertEqual(normalize_status_input(status), status)

    def test_aliases_resolved(self) -> None:
        self.assertEqual(normalize_status_input("check"), "Investigating")
        self.assertEqual(normalize_status_input("fix"), "Mitigating")
        self.assertEqual(normalize_status_input("done"), "Resolved")
        self.assertEqual(normalize_status_input("skip"), "Dismissed")
        self.assertEqual(normalize_status_input("start"), "New")

    def test_case_insensitive(self) -> None:
        self.assertEqual(normalize_status_input("new"), "New")
        self.assertEqual(normalize_status_input("NEW"), "New")
        self.assertEqual(normalize_status_input("Investigating"), "Investigating")

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_status_input("")

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_status_input("InvalidStatus")


class TestIncidentSeverityValidation(unittest.TestCase):
    """Verify incident severity and status inputs are validated."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_incidents.db"
        init_service_store(self.db_path)

    def test_valid_severities_accepted(self) -> None:
        for sev in INCIDENT_SEVERITIES:
            incident_id = upsert_incident(
                title=f"Test {sev}",
                severity=sev,
                description="Test incident",
                status="Open",
                actor="test",
                actor_role="operator",
                path=self.db_path,
            )
            self.assertTrue(incident_id.startswith("INC-"))

    def test_invalid_severity_rejected(self) -> None:
        for bad_sev in ["SEV-0", "SEV-4", "critical", "", "HIGH"]:
            with self.assertRaises(ValueError, msg=f"Should reject severity={bad_sev}"):
                upsert_incident(
                    title="Test bad severity",
                    severity=bad_sev,
                    description="Should fail",
                    status="Open",
                    actor="test",
                    actor_role="operator",
                    path=self.db_path,
                )

    def test_invalid_incident_status_rejected(self) -> None:
        for bad_status in ["Pending", "Active", "", "new"]:
            with self.assertRaises(
                ValueError, msg=f"Should reject status={bad_status}"
            ):
                upsert_incident(
                    title="Test bad status",
                    severity="SEV-2",
                    description="Should fail",
                    status=bad_status,
                    actor="test",
                    actor_role="operator",
                    path=self.db_path,
                )

    def test_valid_statuses_accepted(self) -> None:
        for status in INCIDENT_STATUSES:
            incident_id = upsert_incident(
                title=f"Test {status}",
                severity="SEV-3",
                description="Valid status test",
                status=status,
                actor="test",
                actor_role="operator",
                path=self.db_path,
            )
            self.assertTrue(incident_id)

    def test_empty_title_rejected(self) -> None:
        with self.assertRaises(ValueError):
            upsert_incident(
                title="",
                severity="SEV-2",
                description="Empty title",
                status="Open",
                actor="test",
                actor_role="operator",
                path=self.db_path,
            )

    def test_permission_denied_for_viewer(self) -> None:
        with self.assertRaises(PermissionError):
            upsert_incident(
                title="Viewer test",
                severity="SEV-3",
                description="Should fail",
                status="Open",
                actor="viewer",
                actor_role="viewer",
                path=self.db_path,
            )


class TestDataAccessValidation(unittest.TestCase):
    """Verify data access input validation."""

    def test_empty_shipment_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            fetch_shipment_feature_row(Path("/nonexistent.db"), "")

    def test_none_shipment_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            fetch_shipment_feature_row(Path("/nonexistent.db"), None)  # type: ignore[arg-type]


class TestQueueSyncValidation(unittest.TestCase):
    """Verify queue sync validates input files."""

    def test_missing_csv_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_queue_rows(Path("/nonexistent/queue.csv"))


class TestModuleLoggers(unittest.TestCase):
    """Verify each module creates a logger under the control_tower namespace."""

    def test_all_modules_have_loggers(self) -> None:
        expected_loggers = [
            "control_tower.data_access",
            "control_tower.evidence_pack",
            "control_tower.incident_llm",
            "control_tower.modeling",
            "control_tower.ops_output",
            "control_tower.quality",
            "control_tower.queue_sync",
            "control_tower.scoring",
            "control_tower.semantic_layer",
            "control_tower.semantic_queries",
            "control_tower.service_health",
            "control_tower.service_store",
            "control_tower.sqlite_pipeline",
            "control_tower.synthetic_data",
        ]
        # Import all modules to ensure loggers are registered
        import control_tower.data_access  # noqa: F401
        import control_tower.evidence_pack  # noqa: F401
        import control_tower.incident_llm  # noqa: F401
        import control_tower.modeling  # noqa: F401
        import control_tower.ops_output  # noqa: F401
        import control_tower.quality  # noqa: F401
        import control_tower.queue_sync  # noqa: F401
        import control_tower.scoring  # noqa: F401
        import control_tower.semantic_layer  # noqa: F401
        import control_tower.semantic_queries  # noqa: F401
        import control_tower.service_health  # noqa: F401
        import control_tower.service_store  # noqa: F401
        import control_tower.sqlite_pipeline  # noqa: F401
        import control_tower.synthetic_data  # noqa: F401

        manager = logging.Logger.manager
        for name in expected_loggers:
            self.assertIn(
                name,
                manager.loggerDict,
                f"Logger '{name}' not found. Each module should call logging.getLogger().",
            )


if __name__ == "__main__":
    unittest.main()
