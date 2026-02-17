from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


class TestPipelineLock(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        script_path = root / "scripts" / "run_pipeline.py"
        spec = importlib.util.spec_from_file_location("run_pipeline_test_module", script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import script module: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.pipeline = module
        cls.original_lock = module.PIPELINE_LOCK

    @classmethod
    def tearDownClass(cls) -> None:
        cls.pipeline.PIPELINE_LOCK = cls.original_lock

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.lock_path = Path(self.tmpdir.name) / ".pipeline.lock"
        self.pipeline.PIPELINE_LOCK = self.lock_path

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_release_requires_owner_token(self) -> None:
        token = self.pipeline._acquire_lock(timeout_sec=1, poll_sec=0.05, stale_sec=10)
        self.assertTrue(self.lock_path.exists())

        self.pipeline._release_lock("wrong-token")
        self.assertTrue(self.lock_path.exists())

        self.pipeline._release_lock(token)
        self.assertFalse(self.lock_path.exists())

    def test_reclaims_stale_lock(self) -> None:
        stale_payload = {
            "token": "old-token",
            "pid": 0,
            "acquired_at_utc": "2000-01-01T00:00:00+00:00",
        }
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w", encoding="utf-8") as f:
            json.dump(stale_payload, f, ensure_ascii=True)

        token = self.pipeline._acquire_lock(timeout_sec=1, poll_sec=0.05, stale_sec=0.1)
        self.assertTrue(self.lock_path.exists())
        with self.lock_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload.get("token"), token)
        self.pipeline._release_lock(token)
        self.assertFalse(self.lock_path.exists())

    def test_active_lock_times_out(self) -> None:
        token = self.pipeline._acquire_lock(timeout_sec=1, poll_sec=0.05, stale_sec=10)
        with self.assertRaises(RuntimeError):
            self.pipeline._acquire_lock(timeout_sec=0.2, poll_sec=0.05, stale_sec=0.1)
        self.pipeline._release_lock(token)

    def test_env_float_parsing_is_resilient(self) -> None:
        old_timeout = os.environ.get("PIPELINE_LOCK_TIMEOUT_SEC")
        old_stale = os.environ.get("PIPELINE_LOCK_STALE_SEC")
        try:
            os.environ["PIPELINE_LOCK_TIMEOUT_SEC"] = "not-a-number"
            os.environ["PIPELINE_LOCK_STALE_SEC"] = "-999"
            token = self.pipeline._acquire_lock(timeout_sec=None, poll_sec=0.05, stale_sec=None)
            self.assertTrue(self.lock_path.exists())
            self.pipeline._release_lock(token)
            self.assertFalse(self.lock_path.exists())
        finally:
            if old_timeout is None:
                os.environ.pop("PIPELINE_LOCK_TIMEOUT_SEC", None)
            else:
                os.environ["PIPELINE_LOCK_TIMEOUT_SEC"] = old_timeout
            if old_stale is None:
                os.environ.pop("PIPELINE_LOCK_STALE_SEC", None)
            else:
                os.environ["PIPELINE_LOCK_STALE_SEC"] = old_stale


if __name__ == "__main__":
    unittest.main()
