#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_STATUS = ROOT / "data" / "output" / "pipeline_status.json"
PIPELINE_LOCK = ROOT / "data" / "processed" / ".pipeline.lock"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.service_store import log_pipeline_run


def _run(cmd: list[str]) -> dict:
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    elapsed = round(time.time() - start, 3)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return {
        "command": " ".join(cmd),
        "elapsed_sec": elapsed,
        "stdout": result.stdout.strip(),
    }


def _write_status(payload: dict) -> None:
    PIPELINE_STATUS.parent.mkdir(parents=True, exist_ok=True)
    with PIPELINE_STATUS.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _parse_iso_dt(value: str) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _read_lock_payload() -> dict[str, Any]:
    if not PIPELINE_LOCK.exists():
        return {}
    try:
        with PIPELINE_LOCK.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _should_reclaim_lock(payload: dict[str, Any], stale_sec: float) -> bool:
    pid_raw = payload.get("pid", 0)
    try:
        pid = int(pid_raw)
    except (TypeError, ValueError):
        pid = 0

    if pid > 0 and not _pid_alive(pid):
        return True

    acquired_at = _parse_iso_dt(str(payload.get("acquired_at_utc", "")))
    if acquired_at is None:
        try:
            age = time.time() - PIPELINE_LOCK.stat().st_mtime
        except OSError:
            return False
        return age >= stale_sec

    age = (datetime.now(timezone.utc) - acquired_at).total_seconds()
    return age >= stale_sec and (pid <= 0 or not _pid_alive(pid))


def _acquire_lock(timeout_sec: float | None = None, poll_sec: float = 0.4, stale_sec: float | None = None) -> str:
    if timeout_sec is None:
        timeout_sec = _read_env_float("PIPELINE_LOCK_TIMEOUT_SEC", default=60.0, minimum=0.1)
    if stale_sec is None:
        stale_sec = _read_env_float("PIPELINE_LOCK_STALE_SEC", default=900.0, minimum=1.0)
    PIPELINE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    lock_token = f"{os.getpid()}-{int(time.time() * 1000)}"
    lock_payload = {
        "token": lock_token,
        "pid": os.getpid(),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    while True:
        try:
            fd = os.open(str(PIPELINE_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(lock_payload, f, ensure_ascii=True, indent=2)
            return lock_token
        except FileExistsError:
            payload = _read_lock_payload()
            if _should_reclaim_lock(payload, stale_sec=stale_sec):
                try:
                    PIPELINE_LOCK.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.time() - started > timeout_sec:
                raise RuntimeError(f"pipeline lock timeout: {PIPELINE_LOCK}")
            time.sleep(poll_sec)


def _release_lock(lock_token: str) -> None:
    if PIPELINE_LOCK.exists():
        payload = _read_lock_payload()
        if payload.get("token") == lock_token:
            try:
                PIPELINE_LOCK.unlink()
            except FileNotFoundError:
                pass


def _read_env_float(name: str, *, default: float, minimum: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return max(minimum, default)
    try:
        value = float(raw)
    except ValueError:
        return max(minimum, default)
    if not math.isfinite(value):
        return max(minimum, default)
    return max(minimum, value)


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    run_id = f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    timer = time.time()
    generate_step = [
        sys.executable,
        "scripts/generate_data.py",
        "--seed",
        "42",
        "--days",
        "120",
        "--orders-per-day",
        "120",
    ]
    anchored_start_date = str(os.getenv("LP_ANCHOR_DATE", "")).strip()
    if anchored_start_date:
        generate_step.extend(["--start-date", anchored_start_date])

    steps = [
        generate_step,
        [sys.executable, "scripts/build_marts.py"],
        [sys.executable, "scripts/init_service_store.py"],
        [sys.executable, "scripts/run_data_quality.py"],
        [sys.executable, "scripts/train_model.py"],
        [sys.executable, "scripts/score_daily.py"],
        [sys.executable, "scripts/build_semantic_layer.py"],
        [sys.executable, "scripts/build_report.py"],
        [sys.executable, "scripts/export_datadog_series.py"],
    ]

    logs = []
    lock_acquired = False
    lock_token = ""
    try:
        lock_token = _acquire_lock()
        lock_acquired = True
        for step in steps:
            logs.append(_run(step))

        payload = {
            "status": "ok",
            "run_id": run_id,
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "steps": logs,
        }
        _write_status(payload)
        log_pipeline_run(
            run_id=run_id,
            started_at=payload["started_at_utc"],
            finished_at=payload["finished_at_utc"],
            status="ok",
            step_count=len(logs),
            duration_sec=round(time.time() - timer, 3),
        )
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "failed",
            "run_id": run_id,
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "steps": logs,
            "error": str(exc),
        }
        _write_status(payload)
        log_pipeline_run(
            run_id=run_id,
            started_at=payload["started_at_utc"],
            finished_at=payload["finished_at_utc"],
            status="failed",
            step_count=len(logs),
            duration_sec=round(time.time() - timer, 3),
            error_text=str(exc),
        )
        raise
    finally:
        if lock_acquired:
            _release_lock(lock_token)


if __name__ == "__main__":
    main()
