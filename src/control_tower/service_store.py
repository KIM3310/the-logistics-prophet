from __future__ import annotations

from collections import Counter
import hashlib
import json
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from .config import SERVICE_DB_PATH

PERMISSIONS = {
    "admin": {"queue_update", "incident_manage", "user_manage", "view"},
    "operator": {"queue_update", "incident_manage", "view"},
    "viewer": {"view"},
}

STATUS_ORDER = ["New", "Investigating", "Mitigating", "Resolved", "Dismissed"]
QUEUE_STATUSES = set(STATUS_ORDER)
UNRESOLVED_STATUSES = {"New", "Investigating", "Mitigating"}
INCIDENT_SEVERITIES = {"SEV-1", "SEV-2", "SEV-3"}
INCIDENT_STATUSES = {"Open", "Monitoring", "Closed"}

STATUS_ALIASES = {
    "new": "New",
    "start": "New",
    "investigating": "Investigating",
    "investigate": "Investigating",
    "check": "Investigating",
    "mitigating": "Mitigating",
    "mitigate": "Mitigating",
    "fix": "Mitigating",
    "resolved": "Resolved",
    "resolve": "Resolved",
    "done": "Resolved",
    "closed": "Resolved",
    "close": "Resolved",
    "dismissed": "Dismissed",
    "dismiss": "Dismissed",
    "skip": "Dismissed",
}

SERVICE_CORE_STATUS_TERMS: Dict[str, Dict[str, object]] = {
    "New": {"stage": "Start", "core_label": "Start", "stage_order": 1},
    "Investigating": {"stage": "Check", "core_label": "Check", "stage_order": 2},
    "Mitigating": {"stage": "Fix", "core_label": "Fix", "stage_order": 3},
    "Resolved": {"stage": "Done", "core_label": "Done", "stage_order": 4},
    "Dismissed": {"stage": "Done", "core_label": "Skip", "stage_order": 4},
}

SERVICE_CORE_STAGE_LABELS = {
    "Start": "Start",
    "Check": "Check",
    "Fix": "Fix",
    "Done": "Done",
}

QUEUE_TRANSITIONS: Dict[str, set[str]] = {
    "New": {"Investigating", "Dismissed"},
    "Investigating": {"Mitigating", "Resolved", "Dismissed"},
    "Mitigating": {"Investigating", "Resolved", "Dismissed"},
    "Resolved": set(),
    "Dismissed": set(),
}


def _ensure_columns(conn: sqlite3.Connection, table: str, required: Dict[str, str]) -> None:
    existing_rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(r[1]) for r in existing_rows}
    for col, col_def in required.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_eta_action_at(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    dt = _parse_iso_datetime(raw)
    if dt is None:
        raise ValueError("eta_action_at must be ISO-8601 timestamp or empty")
    return dt.isoformat()


def service_core_term_for_status(status: str) -> Dict[str, object]:
    normalized = str(status or "").strip()
    base = SERVICE_CORE_STATUS_TERMS.get(normalized)
    if base:
        return {"status": normalized, **base}
    return {"status": normalized or "Unknown", "stage": "Unknown", "core_label": "Unknown", "stage_order": 999}


def normalize_status_input(status: str) -> str:
    raw = str(status or "").strip()
    if not raw:
        raise ValueError("status is required")

    if raw in QUEUE_STATUSES:
        return raw

    canonical_by_lower = {item.casefold(): item for item in STATUS_ORDER}
    lowered = raw.casefold()
    if lowered in canonical_by_lower:
        return canonical_by_lower[lowered]

    alias = STATUS_ALIASES.get(lowered)
    if alias:
        return alias

    supported = ", ".join(STATUS_ORDER + ["Start", "Check", "Fix", "Done", "Skip"])
    raise ValueError(f"invalid status: {raw}. supported values: {supported}")


def list_service_core_terms() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for status in STATUS_ORDER:
        details = SERVICE_CORE_STATUS_TERMS.get(status, {})
        rows.append({"status": status, **details})
    return rows


def allowed_next_statuses(current_status: str, actor_role: str = "operator") -> List[str]:
    normalized = normalize_status_input(current_status)
    if actor_role.lower() == "admin":
        return STATUS_ORDER.copy()

    allowed = set(QUEUE_TRANSITIONS.get(normalized, set()))
    if normalized in QUEUE_STATUSES:
        allowed.add(normalized)
    if not allowed:
        return STATUS_ORDER.copy()
    return [status for status in STATUS_ORDER if status in allowed]


def _validate_status_transition(current_status: str, next_status: str, actor_role: str) -> None:
    current = normalize_status_input(current_status)
    nxt = normalize_status_input(next_status)
    if current == nxt:
        return
    if actor_role.lower() == "admin":
        return

    allowed = QUEUE_TRANSITIONS.get(current, set())
    if nxt not in allowed:
        allowed_labels = ", ".join([status for status in STATUS_ORDER if status in allowed]) if allowed else "(none)"
        raise ValueError(f"invalid transition {current} -> {nxt}; allowed: {allowed_labels}")


def _dict_factory(row: sqlite3.Row) -> Dict[str, object]:
    return dict(row)


def _hash_payload(payload: Dict[str, object], prev_hash: str) -> str:
    msg = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(prev_hash.encode("utf-8") + b"|" + msg).hexdigest()
    return digest


def _hash_password(password: str, salt_hex: str | None = None) -> Tuple[str, str]:
    if salt_hex is None:
        salt_hex = secrets.token_hex(16)
    salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 160_000).hex()
    return digest, salt_hex


def _verify_password(password: str, digest_hex: str, salt_hex: str) -> bool:
    computed, _ = _hash_password(password, salt_hex=salt_hex)
    return secrets.compare_digest(computed, digest_hex)


def has_permission(role: str, permission: str) -> bool:
    perms = PERMISSIONS.get(role.lower(), set())
    return permission in perms


def init_service_store(path: Path = SERVICE_DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS service_queue (
                shipment_id TEXT PRIMARY KEY,
                ship_date TEXT,
                order_id TEXT,
                risk_score REAL,
                risk_band TEXT,
                prediction INTEGER,
                key_driver TEXT,
                driver_2 TEXT,
                driver_3 TEXT,
                recommended_action TEXT,
                status TEXT DEFAULT 'New',
                owner TEXT DEFAULT '',
                note TEXT DEFAULT '',
                eta_action_at TEXT DEFAULT '',
                created_at TEXT,
                updated_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_service_queue_band ON service_queue(risk_band);
            CREATE INDEX IF NOT EXISTS idx_service_queue_status ON service_queue(status);
            CREATE INDEX IF NOT EXISTS idx_service_queue_score ON service_queue(risk_score);

            CREATE TABLE IF NOT EXISTS service_activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor TEXT,
                actor_role TEXT,
                action TEXT,
                entity_type TEXT,
                entity_id TEXT,
                payload_json TEXT,
                previous_state_json TEXT,
                new_state_json TEXT,
                reason TEXT,
                request_id TEXT,
                prev_hash TEXT,
                event_hash TEXT,
                created_at TEXT
            );

            -- service_activity_log index is created after migration

            CREATE TABLE IF NOT EXISTS service_incidents (
                incident_id TEXT PRIMARY KEY,
                title TEXT,
                severity TEXT,
                description TEXT,
                status TEXT,
                owner TEXT DEFAULT '',
                source_rule TEXT DEFAULT '',
                opened_at TEXT,
                updated_at TEXT,
                closed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_service_incidents_status ON service_incidents(status);

            CREATE TABLE IF NOT EXISTS service_run_history (
                run_id TEXT PRIMARY KEY,
                started_at TEXT,
                finished_at TEXT,
                status TEXT,
                step_count INTEGER,
                duration_sec REAL,
                error_text TEXT
            );

            CREATE TABLE IF NOT EXISTS service_users (
                username TEXT PRIMARY KEY,
                display_name TEXT,
                role TEXT,
                password_hash TEXT,
                password_salt TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT,
                updated_at TEXT,
                last_login_at TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_service_users_role ON service_users(role);
            """
        )
        _ensure_columns(
            conn,
            "service_queue",
            {
                "status": "TEXT DEFAULT 'New'",
                "owner": "TEXT DEFAULT ''",
                "note": "TEXT DEFAULT ''",
                "eta_action_at": "TEXT DEFAULT ''",
                "updated_at": "TEXT DEFAULT ''",
            },
        )
        _ensure_columns(
            conn,
            "service_activity_log",
            {
                "actor_role": "TEXT DEFAULT ''",
                "entity_type": "TEXT DEFAULT ''",
                "entity_id": "TEXT DEFAULT ''",
                "previous_state_json": "TEXT DEFAULT '{}'",
                "new_state_json": "TEXT DEFAULT '{}'",
                "reason": "TEXT DEFAULT ''",
                "request_id": "TEXT DEFAULT ''",
                "prev_hash": "TEXT DEFAULT 'GENESIS'",
                "event_hash": "TEXT DEFAULT ''",
            },
        )
        _ensure_columns(
            conn,
            "service_incidents",
            {
                "owner": "TEXT DEFAULT ''",
                "source_rule": "TEXT DEFAULT ''",
            },
        )
        _ensure_columns(
            conn,
            "service_users",
            {
                "display_name": "TEXT DEFAULT ''",
                "role": "TEXT DEFAULT 'viewer'",
                "password_hash": "TEXT DEFAULT ''",
                "password_salt": "TEXT DEFAULT ''",
                "is_active": "INTEGER DEFAULT 1",
                "created_at": "TEXT DEFAULT ''",
                "updated_at": "TEXT DEFAULT ''",
                "last_login_at": "TEXT DEFAULT ''",
            },
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_service_queue_owner ON service_queue(owner)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_service_queue_updated_at ON service_queue(updated_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_service_queue_eta_action_at ON service_queue(eta_action_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_service_incidents_source_rule ON service_incidents(source_rule)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_service_activity_entity ON service_activity_log(entity_type, entity_id)"
        )
        conn.commit()

        _ensure_default_users(conn)
        conn.commit()
    finally:
        conn.close()


def _ensure_default_users(conn: sqlite3.Connection) -> None:
    cnt = conn.execute("SELECT COUNT(*) FROM service_users").fetchone()[0]
    if cnt > 0:
        return

    now = utc_now_iso()
    defaults = [
        ("admin", "Admin", "admin", "admin123!"),
        ("operator", "Operator", "operator", "ops123!"),
        ("viewer", "Viewer", "viewer", "view123!"),
    ]
    rows = []
    for username, display_name, role, password in defaults:
        password_hash, password_salt = _hash_password(password)
        rows.append((username, display_name, role, password_hash, password_salt, 1, now, now, ""))

    conn.executemany(
        """
        INSERT INTO service_users (
            username, display_name, role, password_hash, password_salt, is_active, created_at, updated_at, last_login_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def list_users(path: Path = SERVICE_DB_PATH, include_inactive: bool = False) -> List[Dict[str, object]]:
    init_service_store(path)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        if include_inactive:
            rows = conn.execute(
                "SELECT username, display_name, role, is_active, created_at, updated_at, last_login_at FROM service_users ORDER BY username"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT username, display_name, role, is_active, created_at, updated_at, last_login_at FROM service_users WHERE is_active = 1 ORDER BY username"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def create_or_update_user(
    username: str,
    display_name: str,
    role: str,
    password: str,
    actor: str,
    actor_role: str,
    is_active: bool = True,
    path: Path = SERVICE_DB_PATH,
) -> None:
    init_service_store(path)
    if not has_permission(actor_role, "user_manage"):
        raise PermissionError("actor does not have permission user_manage")

    role = role.lower().strip()
    if role not in PERMISSIONS:
        raise ValueError("invalid role")

    now = utc_now_iso()
    password_hash, password_salt = _hash_password(password)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        prev = conn.execute(
            "SELECT username, display_name, role, is_active FROM service_users WHERE username = ?",
            (username,),
        ).fetchone()

        conn.execute(
            """
            INSERT INTO service_users (username, display_name, role, password_hash, password_salt, is_active, created_at, updated_at, last_login_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                display_name=excluded.display_name,
                role=excluded.role,
                password_hash=excluded.password_hash,
                password_salt=excluded.password_salt,
                is_active=excluded.is_active,
                updated_at=excluded.updated_at
            """,
            (username, display_name, role, password_hash, password_salt, 1 if is_active else 0, now, now, ""),
        )

        curr = conn.execute(
            "SELECT username, display_name, role, is_active FROM service_users WHERE username = ?",
            (username,),
        ).fetchone()

        _append_activity(
            conn=conn,
            actor=actor,
            actor_role=actor_role,
            action="user_upsert",
            entity_type="user",
            entity_id=username,
            payload={"username": username, "role": role, "active": bool(is_active)},
            previous_state=dict(prev) if prev else {},
            new_state=dict(curr) if curr else {},
            reason="user administration",
        )
        conn.commit()
    finally:
        conn.close()


def authenticate_user(username: str, password: str, path: Path = SERVICE_DB_PATH) -> Dict[str, object] | None:
    init_service_store(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT username, display_name, role, password_hash, password_salt, is_active
            FROM service_users WHERE username = ?
            """,
            (username,),
        ).fetchone()
        if not row:
            return None

        if int(row["is_active"]) != 1:
            return None

        if not _verify_password(password, str(row["password_hash"]), str(row["password_salt"])):
            return None

        now = utc_now_iso()
        conn.execute("UPDATE service_users SET last_login_at = ?, updated_at = ? WHERE username = ?", (now, now, username))

        _append_activity(
            conn=conn,
            actor=username,
            actor_role=str(row["role"]),
            action="login_success",
            entity_type="auth",
            entity_id=username,
            payload={"username": username},
            previous_state={},
            new_state={"last_login_at": now},
            reason="interactive login",
        )
        conn.commit()

        return {
            "username": str(row["username"]),
            "display_name": str(row["display_name"]),
            "role": str(row["role"]),
        }
    finally:
        conn.close()


def _append_activity(
    conn: sqlite3.Connection,
    actor: str,
    actor_role: str,
    action: str,
    entity_type: str,
    entity_id: str,
    payload: Dict[str, object],
    previous_state: Dict[str, object],
    new_state: Dict[str, object],
    reason: str,
    request_id: str | None = None,
) -> None:
    created_at = utc_now_iso()
    if request_id is None:
        request_id = str(uuid.uuid4())

    prev_hash_row = conn.execute("SELECT event_hash FROM service_activity_log ORDER BY id DESC LIMIT 1").fetchone()
    prev_hash_candidate = str(prev_hash_row[0]) if prev_hash_row and prev_hash_row[0] else ""
    prev_hash = prev_hash_candidate if len(prev_hash_candidate) == 64 else "GENESIS"

    canonical_payload = {
        "actor": actor,
        "actor_role": actor_role,
        "action": action,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "payload": payload,
        "previous_state": previous_state,
        "new_state": new_state,
        "reason": reason,
        "request_id": request_id,
        "created_at": created_at,
    }
    event_hash = _hash_payload(canonical_payload, prev_hash)

    conn.execute(
        """
        INSERT INTO service_activity_log (
            actor, actor_role, action, entity_type, entity_id,
            payload_json, previous_state_json, new_state_json,
            reason, request_id, prev_hash, event_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            actor,
            actor_role,
            action,
            entity_type,
            entity_id,
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            json.dumps(previous_state, ensure_ascii=True, sort_keys=True),
            json.dumps(new_state, ensure_ascii=True, sort_keys=True),
            reason,
            request_id,
            prev_hash,
            event_hash,
            created_at,
        ),
    )


def verify_audit_chain(path: Path = SERVICE_DB_PATH, limit: int = 5000) -> Dict[str, object]:
    init_service_store(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, actor, actor_role, action, entity_type, entity_id,
                   payload_json, previous_state_json, new_state_json, reason,
                   request_id, prev_hash, event_hash, created_at
            FROM service_activity_log
            ORDER BY id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    prev_hash = "GENESIS"
    checked = 0
    skipped_legacy = 0
    latest_checked_id = 0
    for row in rows:
        row_prev_hash = str(row["prev_hash"] or "")
        row_event_hash = str(row["event_hash"] or "")

        # Legacy rows from older schema/version may not carry verifiable chain fields.
        if len(row_event_hash) != 64 or (row_prev_hash not in ("GENESIS", "") and len(row_prev_hash) != 64):
            skipped_legacy += 1
            continue

        payload = {
            "actor": row["actor"],
            "actor_role": row["actor_role"],
            "action": row["action"],
            "entity_type": row["entity_type"],
            "entity_id": row["entity_id"],
            "payload": json.loads(row["payload_json"] or "{}"),
            "previous_state": json.loads(row["previous_state_json"] or "{}"),
            "new_state": json.loads(row["new_state_json"] or "{}"),
            "reason": row["reason"],
            "request_id": row["request_id"],
            "created_at": row["created_at"],
        }

        if row_prev_hash == "" and row_event_hash:
            row_prev_hash = "GENESIS"

        if row_prev_hash == "GENESIS":
            prev_hash = "GENESIS"

        if row_prev_hash != prev_hash:
            return {
                "valid": False,
                "checked": checked,
                "skipped_legacy": skipped_legacy,
                "failed_id": int(row["id"]),
                "reason": "prev_hash_mismatch",
                "latest_hash": prev_hash,
            }

        expected_hash = _hash_payload(payload, prev_hash)
        if row_event_hash != expected_hash:
            return {
                "valid": False,
                "checked": checked,
                "skipped_legacy": skipped_legacy,
                "failed_id": int(row["id"]),
                "reason": "event_hash_mismatch",
                "latest_hash": prev_hash,
            }

        prev_hash = row_event_hash
        checked += 1
        latest_checked_id = int(row["id"])

    return {
        "valid": True,
        "checked": checked,
        "skipped_legacy": skipped_legacy,
        "latest_hash": prev_hash,
        "latest_checked_id": latest_checked_id,
    }


def upsert_queue_rows(rows: List[Dict[str, object]], path: Path = SERVICE_DB_PATH) -> int:
    init_service_store(path)
    now = utc_now_iso()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        sql = (
            "INSERT INTO service_queue ("
            "shipment_id, ship_date, order_id, risk_score, risk_band, prediction, key_driver, driver_2, driver_3, "
            "recommended_action, status, owner, note, eta_action_at, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(shipment_id) DO UPDATE SET "
            "ship_date=excluded.ship_date, "
            "order_id=excluded.order_id, "
            "risk_score=excluded.risk_score, "
            "risk_band=excluded.risk_band, "
            "prediction=excluded.prediction, "
            "key_driver=excluded.key_driver, "
            "driver_2=excluded.driver_2, "
            "driver_3=excluded.driver_3, "
            "recommended_action=excluded.recommended_action, "
            "updated_at=excluded.updated_at"
        )

        payload = []
        for row in rows:
            payload.append(
                (
                    str(row.get("shipment_id", "")),
                    str(row.get("ship_date", "")),
                    str(row.get("order_id", "")),
                    float(row.get("risk_score", 0.0)),
                    str(row.get("risk_band", "Low")),
                    int(float(row.get("prediction", 0))),
                    str(row.get("key_driver", "")),
                    str(row.get("driver_2", "")),
                    str(row.get("driver_3", "")),
                    str(row.get("recommended_action", "")),
                    "New",
                    "",
                    "",
                    "",
                    now,
                    now,
                )
            )

        conn.executemany(sql, payload)
        _append_activity(
            conn=conn,
            actor="pipeline",
            actor_role="system",
            action="queue_sync",
            entity_type="queue",
            entity_id="bulk",
            payload={"rows": len(payload)},
            previous_state={},
            new_state={"rows": len(payload)},
            reason="daily scoring sync",
        )
        conn.commit()
        return len(payload)
    finally:
        conn.close()


def fetch_queue(
    path: Path = SERVICE_DB_PATH,
    band: str | None = None,
    status: str | None = None,
    owner: str | None = None,
    limit: int = 500,
) -> List[Dict[str, object]]:
    init_service_store(path)

    where = []
    params: List[object] = []
    if band and band != "All":
        where.append("risk_band = ?")
        params.append(band)
    if status and status != "All":
        where.append("status = ?")
        params.append(status)
    if owner and owner != "All":
        where.append("owner = ?")
        params.append(owner)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    sql = (
        "SELECT shipment_id, ship_date, order_id, risk_score, risk_band, prediction, key_driver, driver_2, driver_3, "
        "recommended_action, status, owner, note, eta_action_at, created_at, updated_at "
        f"FROM service_queue {where_sql} ORDER BY risk_score DESC LIMIT ?"
    )
    params.append(limit)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_queue_summary(path: Path = SERVICE_DB_PATH) -> Dict[str, object]:
    init_service_store(path)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM service_queue GROUP BY status ORDER BY cnt DESC"
        ).fetchall()
        risk_rows = conn.execute(
            "SELECT risk_band, COUNT(*) AS cnt FROM service_queue GROUP BY risk_band ORDER BY cnt DESC"
        ).fetchall()
        critical_open = conn.execute(
            "SELECT COUNT(*) FROM service_queue WHERE risk_band='Critical' AND status IN ('New','Investigating')"
        ).fetchone()[0]
        unresolved = conn.execute(
            "SELECT COUNT(*) FROM service_queue WHERE status IN ('New','Investigating','Mitigating')"
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "status_breakdown": [dict(r) for r in status_rows],
        "risk_breakdown": [dict(r) for r in risk_rows],
        "critical_open": int(critical_open),
        "unresolved": int(unresolved),
    }


def fetch_ops_health(path: Path = SERVICE_DB_PATH, owner_limit: int = 8) -> Dict[str, object]:
    rows = fetch_queue(path=path, limit=10_000)
    now = datetime.now(timezone.utc)

    overdue_eta = 0
    stale_24h = 0
    critical_unassigned = 0
    unresolved_age_hours: List[float] = []
    owner_backlog: Counter[str] = Counter()

    for row in rows:
        status = str(row.get("status", "New"))
        risk_band = str(row.get("risk_band", "Low"))
        owner = str(row.get("owner", "")).strip()
        eta_dt = _parse_iso_datetime(str(row.get("eta_action_at", "")))
        updated_dt = _parse_iso_datetime(str(row.get("updated_at", "")))

        if status in UNRESOLVED_STATUSES:
            owner_key = owner if owner else "Unassigned"
            owner_backlog[owner_key] += 1

            if eta_dt and eta_dt < now:
                overdue_eta += 1

            if updated_dt:
                age_hours = (now - updated_dt).total_seconds() / 3600.0
                unresolved_age_hours.append(age_hours)
                if age_hours >= 24.0:
                    stale_24h += 1

            if risk_band == "Critical" and not owner:
                critical_unassigned += 1

    avg_age = round(sum(unresolved_age_hours) / len(unresolved_age_hours), 2) if unresolved_age_hours else 0.0

    return {
        "generated_at_utc": now.isoformat(),
        "total_queue": len(rows),
        "unresolved": int(sum(owner_backlog.values())),
        "overdue_eta": int(overdue_eta),
        "stale_24h": int(stale_24h),
        "critical_unassigned": int(critical_unassigned),
        "avg_unresolved_age_hours": avg_age,
        "owner_backlog": [{"owner": owner, "count": int(cnt)} for owner, cnt in owner_backlog.most_common(owner_limit)],
    }


def fetch_service_core_snapshot(path: Path = SERVICE_DB_PATH, candidate_limit: int = 12) -> Dict[str, object]:
    rows = fetch_queue(path=path, limit=10_000)
    now = datetime.now(timezone.utc)

    stage_counts: Counter[str] = Counter()
    driver_counts: Counter[str] = Counter()
    escalation_candidates: List[Dict[str, object]] = []
    unresolved = 0
    risk_rank = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}

    for row in rows:
        status = str(row.get("status", "New"))
        term = service_core_term_for_status(status)
        stage = str(term.get("stage", "Unknown"))
        stage_counts[stage] += 1

        if status in UNRESOLVED_STATUSES:
            unresolved += 1
            key_driver = str(row.get("key_driver", "")).strip()
            if key_driver:
                driver_counts[key_driver] += 1

            risk_band = str(row.get("risk_band", "Low"))
            risk_score = float(row.get("risk_score", 0.0) or 0.0)
            owner = str(row.get("owner", "")).strip()
            eta_raw = str(row.get("eta_action_at", ""))
            updated_raw = str(row.get("updated_at", ""))
            eta_dt = _parse_iso_datetime(eta_raw)
            updated_dt = _parse_iso_datetime(updated_raw)
            stale_hours = round(((now - updated_dt).total_seconds() / 3600.0), 2) if updated_dt else 0.0

            reasons = []
            if risk_band == "Critical" and not owner:
                reasons.append("critical_unassigned")
            if eta_dt and eta_dt < now:
                reasons.append("eta_breached")
            if stale_hours >= 24.0:
                reasons.append("stale_24h")
            if status == "New" and risk_band in {"Critical", "High"}:
                reasons.append("priority_backlog")

            if reasons:
                escalation_candidates.append(
                    {
                        "shipment_id": str(row.get("shipment_id", "")),
                        "order_id": str(row.get("order_id", "")),
                        "risk_band": risk_band,
                        "risk_score": round(risk_score, 4),
                        "status": status,
                        "core_stage": str(term.get("core_label", "Unknown")),
                        "owner": owner,
                        "eta_action_at": eta_raw,
                        "stale_hours": stale_hours,
                        "reasons": ",".join(reasons),
                    }
                )

    escalation_candidates.sort(
        key=lambda row: (
            -risk_rank.get(str(row.get("risk_band", "Low")), 0),
            -float(row.get("risk_score", 0.0) or 0.0),
            -float(row.get("stale_hours", 0.0) or 0.0),
        )
    )

    queue_size = len(rows)
    stage_backlog: List[Dict[str, object]] = []
    for stage, count in stage_counts.items():
        stage_backlog.append(
            {
                "stage": stage,
                "label": SERVICE_CORE_STAGE_LABELS.get(stage, stage),
                "count": int(count),
                "share_pct": round((float(count) / float(queue_size)) * 100.0, 1) if queue_size else 0.0,
                "stage_order": min(
                    [
                        int(service_core_term_for_status(status).get("stage_order", 999))
                        for status, details in SERVICE_CORE_STATUS_TERMS.items()
                        if str(details.get("stage")) == stage
                    ]
                    or [999]
                ),
            }
        )
    stage_backlog.sort(key=lambda row: int(row.get("stage_order", 999)))

    driver_hotspots = [{"driver": key, "count": int(cnt)} for key, cnt in driver_counts.most_common(8)]

    return {
        "generated_at_utc": now.isoformat(),
        "queue_size": queue_size,
        "unresolved": unresolved,
        "stage_backlog": stage_backlog,
        "escalation_candidates": escalation_candidates[:candidate_limit],
        "driver_hotspots": driver_hotspots,
    }


def fetch_workflow_sla_snapshot(path: Path = SERVICE_DB_PATH, candidate_limit: int = 12) -> Dict[str, object]:
    rows = fetch_queue(path=path, limit=10_000)
    now = datetime.now(timezone.utc)
    thresholds = {
        "New": 4.0,
        "Investigating": 12.0,
        "Mitigating": 24.0,
    }

    age_buckets = {"0-4h": 0, "4-12h": 0, "12-24h": 0, "24h+": 0}
    stage_counters: Dict[str, Dict[str, object]] = {
        status: {"status": status, "core_stage": service_core_term_for_status(status).get("core_label", "Unknown"), "threshold_hours": threshold, "in_stage": 0, "breached": 0}
        for status, threshold in thresholds.items()
    }
    breached_candidates: List[Dict[str, object]] = []

    unresolved_total = 0
    breached_total = 0

    for row in rows:
        status = str(row.get("status", "New"))
        if status not in UNRESOLVED_STATUSES:
            continue

        unresolved_total += 1
        updated_dt = _parse_iso_datetime(str(row.get("updated_at", "")))
        age_hours = round(((now - updated_dt).total_seconds() / 3600.0), 2) if updated_dt else 0.0

        if age_hours < 4.0:
            age_buckets["0-4h"] += 1
        elif age_hours < 12.0:
            age_buckets["4-12h"] += 1
        elif age_hours < 24.0:
            age_buckets["12-24h"] += 1
        else:
            age_buckets["24h+"] += 1

        threshold = thresholds.get(status)
        if threshold is not None:
            stage_counters[status]["in_stage"] = int(stage_counters[status]["in_stage"]) + 1
            if age_hours >= threshold:
                stage_counters[status]["breached"] = int(stage_counters[status]["breached"]) + 1
                breached_total += 1
                risk_score = float(row.get("risk_score", 0.0) or 0.0)
                risk_band = str(row.get("risk_band", "Low"))
                core_stage = str(service_core_term_for_status(status).get("core_label", "Unknown"))
                over_by = round(age_hours - threshold, 2)
                breached_candidates.append(
                    {
                        "shipment_id": str(row.get("shipment_id", "")),
                        "order_id": str(row.get("order_id", "")),
                        "status": status,
                        "core_stage": core_stage,
                        "risk_band": risk_band,
                        "risk_score": round(risk_score, 4),
                        "owner": str(row.get("owner", "")),
                        "age_hours": age_hours,
                        "threshold_hours": threshold,
                        "over_by_hours": over_by,
                    }
                )

    breached_candidates.sort(
        key=lambda row: (
            -float(row.get("over_by_hours", 0.0) or 0.0),
            -float(row.get("risk_score", 0.0) or 0.0),
        )
    )

    breach_rate = round((float(breached_total) / float(unresolved_total)) * 100.0, 1) if unresolved_total else 0.0

    return {
        "generated_at_utc": now.isoformat(),
        "unresolved_total": unresolved_total,
        "breached_total": breached_total,
        "breach_rate_pct": breach_rate,
        "age_buckets": [{"bucket": bucket, "count": int(cnt)} for bucket, cnt in age_buckets.items()],
        "stage_sla": [
            {
                **item,
                "in_stage": int(item.get("in_stage", 0)),
                "breached": int(item.get("breached", 0)),
            }
            for item in stage_counters.values()
        ],
        "breached_candidates": breached_candidates[:candidate_limit],
    }


def _next_step_hint(status: str, no_owner: bool, overdue_eta: bool) -> str:
    if no_owner:
        return "Set owner"
    if overdue_eta:
        return "Move now"
    if status == "New":
        return "Start check"
    if status == "Investigating":
        return "Start fix"
    if status == "Mitigating":
        return "Close or set new ETA"
    return "Review"


def fetch_service_core_worklist(path: Path = SERVICE_DB_PATH, per_stage_limit: int = 6) -> Dict[str, object]:
    rows = fetch_queue(path=path, limit=10_000)
    now = datetime.now(timezone.utc)

    stage_order = ["Start", "Check", "Fix"]
    stage_items: Dict[str, List[Dict[str, object]]] = {stage: [] for stage in stage_order}
    risk_rank = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    stage_bonus = {"Start": 10.0, "Check": 6.0, "Fix": 4.0}

    for row in rows:
        try:
            status = normalize_status_input(str(row.get("status", "New")))
        except ValueError:
            continue
        if status not in UNRESOLVED_STATUSES:
            continue

        term = service_core_term_for_status(status)
        stage = str(term.get("stage", "Start"))
        if stage not in stage_items:
            continue

        risk_band = str(row.get("risk_band", "Low"))
        risk_score = float(row.get("risk_score", 0.0) or 0.0)
        owner = str(row.get("owner", "")).strip()
        updated_dt = _parse_iso_datetime(str(row.get("updated_at", "")))
        eta_dt = _parse_iso_datetime(str(row.get("eta_action_at", "")))
        age_hours = round(((now - updated_dt).total_seconds() / 3600.0), 2) if updated_dt else 0.0
        overdue_eta = bool(eta_dt and eta_dt < now)
        no_owner = not bool(owner)

        urgency_score = round(
            (risk_score * 100.0)
            + min(age_hours, 72.0) * 0.8
            + (30.0 if overdue_eta else 0.0)
            + (16.0 if no_owner else 0.0)
            + (12.0 if risk_band == "Critical" else 0.0)
            + stage_bonus.get(stage, 0.0),
            2,
        )

        why: List[str] = []
        if overdue_eta:
            why.append("Past ETA")
        if no_owner:
            why.append("No owner")
        if risk_band == "Critical":
            why.append("Critical risk")
        if age_hours >= 12.0:
            why.append("Old item")
        if not why:
            why.append("High risk")

        stage_items[stage].append(
            {
                "shipment_id": str(row.get("shipment_id", "")),
                "order_id": str(row.get("order_id", "")),
                "status": status,
                "step": str(term.get("core_label", stage)),
                "risk_band": risk_band,
                "risk_score": round(risk_score, 4),
                "owner": owner,
                "age_hours": age_hours,
                "eta_action_at": str(row.get("eta_action_at", "")),
                "overdue_eta": overdue_eta,
                "no_owner": no_owner,
                "urgency_score": urgency_score,
                "next_step": _next_step_hint(status=status, no_owner=no_owner, overdue_eta=overdue_eta),
                "why": ", ".join(why),
            }
        )

    total_open = sum(len(stage_items[stage]) for stage in stage_order)
    stages: List[Dict[str, object]] = []
    for stage in stage_order:
        items = sorted(
            stage_items[stage],
            key=lambda item: (
                -float(item.get("urgency_score", 0.0) or 0.0),
                -risk_rank.get(str(item.get("risk_band", "Low")), 0),
                -float(item.get("risk_score", 0.0) or 0.0),
                -float(item.get("age_hours", 0.0) or 0.0),
            ),
        )
        stages.append(
            {
                "stage": stage,
                "label": SERVICE_CORE_STAGE_LABELS.get(stage, stage),
                "count": len(items),
                "share_pct": round((float(len(items)) / float(total_open)) * 100.0, 1) if total_open else 0.0,
                "items": items[: max(1, int(per_stage_limit))],
            }
        )

    top_items: List[Dict[str, object]] = []
    for stage in stages:
        for item in stage.get("items", []):
            top_items.append(item)
    top_items.sort(key=lambda item: -float(item.get("urgency_score", 0.0) or 0.0))

    return {
        "generated_at_utc": now.isoformat(),
        "open_total": total_open,
        "stages": stages,
        "top_items": top_items[:12],
    }


def incident_recommendations_from_metrics(
    queue_summary: Dict[str, object], ops_health: Dict[str, object], max_items: int = 5
) -> List[Dict[str, object]]:
    recommendations: List[Dict[str, object]] = []
    unresolved = int(queue_summary.get("unresolved", 0))
    critical_open = int(queue_summary.get("critical_open", 0))
    overdue_eta = int(ops_health.get("overdue_eta", 0))
    stale_24h = int(ops_health.get("stale_24h", 0))
    critical_unassigned = int(ops_health.get("critical_unassigned", 0))
    avg_age = float(ops_health.get("avg_unresolved_age_hours", 0.0))

    if critical_open >= 1 and critical_unassigned >= 1:
        recommendations.append(
            {
                "rule_id": "critical_unassigned",
                "severity": "SEV-1",
                "title": "Critical items have no owner",
                "description": (
                    f"{critical_open} critical items are open, and {critical_unassigned} have no owner. "
                    "Assign an owner now and start a fast response."
                ),
                "suggested_owner": "control-tower-lead",
            }
        )

    if overdue_eta >= 5:
        recommendations.append(
            {
                "rule_id": "overdue_eta_spike",
                "severity": "SEV-2",
                "title": "Many items are past ETA",
                "description": (
                    f"{overdue_eta} open items are past ETA. "
                    "Add more help and reassign owners now."
                ),
                "suggested_owner": "ops-manager",
            }
        )

    if stale_24h >= 8:
        recommendations.append(
            {
                "rule_id": "stale_queue_24h",
                "severity": "SEV-2",
                "title": "Many items had no update for 24h+",
                "description": (
                    f"{stale_24h} open items had no update for 24h+. "
                    "Start a shared call to remove blockers."
                ),
                "suggested_owner": "sre-ops",
            }
        )

    if unresolved >= 40:
        recommendations.append(
            {
                "rule_id": "queue_pressure_high",
                "severity": "SEV-3",
                "title": "Too many open items",
                "description": (
                    f"Open items={unresolved}, average age={avg_age:.1f}h. "
                    "Run a focused cleanup plan before delays grow."
                ),
                "suggested_owner": "operations-planner",
            }
        )

    return recommendations[:max_items]


def derive_incident_recommendations(path: Path = SERVICE_DB_PATH, max_items: int = 5) -> List[Dict[str, object]]:
    queue_summary = fetch_queue_summary(path=path)
    ops_health = fetch_ops_health(path=path)
    return incident_recommendations_from_metrics(queue_summary=queue_summary, ops_health=ops_health, max_items=max_items)


def update_queue_action(
    shipment_id: str,
    status: str,
    owner: str,
    note: str,
    eta_action_at: str,
    actor: str,
    actor_role: str = "operator",
    path: Path = SERVICE_DB_PATH,
) -> None:
    init_service_store(path)
    if not has_permission(actor_role, "queue_update"):
        raise PermissionError("actor does not have permission queue_update")
    normalized_status = normalize_status_input(status)
    normalized_owner = str(owner).strip()
    normalized_note = str(note).strip()
    normalized_eta = _normalize_eta_action_at(eta_action_at)

    now = utc_now_iso()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        prev = conn.execute(
            "SELECT shipment_id, status, owner, note, eta_action_at, updated_at FROM service_queue WHERE shipment_id = ?",
            (shipment_id,),
        ).fetchone()
        if not prev:
            raise ValueError(f"shipment_id not found: {shipment_id}")
        _validate_status_transition(str(prev["status"]), normalized_status, actor_role=actor_role)

        conn.execute(
            """
            UPDATE service_queue
            SET status = ?, owner = ?, note = ?, eta_action_at = ?, updated_at = ?
            WHERE shipment_id = ?
            """,
            (normalized_status, normalized_owner, normalized_note, normalized_eta, now, shipment_id),
        )

        curr = conn.execute(
            "SELECT shipment_id, status, owner, note, eta_action_at, updated_at FROM service_queue WHERE shipment_id = ?",
            (shipment_id,),
        ).fetchone()

        payload = {
            "status": normalized_status,
            "owner": normalized_owner,
            "note": normalized_note,
            "eta_action_at": normalized_eta,
        }
        _append_activity(
            conn=conn,
            actor=actor,
            actor_role=actor_role,
            action="queue_update",
            entity_type="shipment",
            entity_id=shipment_id,
            payload=payload,
            previous_state=dict(prev),
            new_state=dict(curr) if curr else {},
            reason="operator action",
        )
        conn.commit()
    finally:
        conn.close()


def bulk_update_queue_actions(
    shipment_ids: List[str],
    status: str,
    owner: str,
    note: str,
    eta_action_at: str,
    actor: str,
    actor_role: str = "operator",
    path: Path = SERVICE_DB_PATH,
) -> Dict[str, object]:
    init_service_store(path)
    if not has_permission(actor_role, "queue_update"):
        raise PermissionError("actor does not have permission queue_update")
    normalized_status = normalize_status_input(status)
    normalized_owner = str(owner).strip()
    normalized_note = str(note).strip()
    normalized_eta = _normalize_eta_action_at(eta_action_at)

    normalized_ids = []
    seen = set()
    for shipment_id in shipment_ids:
        candidate = str(shipment_id).strip()
        if candidate and candidate not in seen:
            normalized_ids.append(candidate)
            seen.add(candidate)

    if not normalized_ids:
        return {"updated": 0, "missing": [], "invalid_transitions": []}

    now = utc_now_iso()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in normalized_ids)
        prev_rows = conn.execute(
            f"""
            SELECT shipment_id, status, owner, note, eta_action_at, updated_at
            FROM service_queue
            WHERE shipment_id IN ({placeholders})
            """,
            tuple(normalized_ids),
        ).fetchall()
        prev_map = {str(row["shipment_id"]): dict(row) for row in prev_rows}

        invalid_transitions: List[Dict[str, object]] = []
        update_payload = []
        for shipment_id in normalized_ids:
            if shipment_id not in prev_map:
                continue
            current_status = str(prev_map[shipment_id].get("status", ""))
            try:
                _validate_status_transition(current_status, normalized_status, actor_role=actor_role)
            except ValueError as exc:
                invalid_transitions.append(
                    {
                        "shipment_id": shipment_id,
                        "current_status": current_status,
                        "requested_status": normalized_status,
                        "reason": str(exc),
                    }
                )
                continue
            update_payload.append((normalized_status, normalized_owner, normalized_note, normalized_eta, now, shipment_id))
        if update_payload:
            conn.executemany(
                """
                UPDATE service_queue
                SET status = ?, owner = ?, note = ?, eta_action_at = ?, updated_at = ?
                WHERE shipment_id = ?
                """,
                update_payload,
            )

        curr_rows = conn.execute(
            f"""
            SELECT shipment_id, status, owner, note, eta_action_at, updated_at
            FROM service_queue
            WHERE shipment_id IN ({placeholders})
            """,
            tuple(normalized_ids),
        ).fetchall()
        curr_map = {str(row["shipment_id"]): dict(row) for row in curr_rows}

        missing = [shipment_id for shipment_id in normalized_ids if shipment_id not in curr_map]
        invalid_ids = {str(item.get("shipment_id", "")) for item in invalid_transitions}
        updated_ids = [shipment_id for shipment_id in normalized_ids if shipment_id in curr_map and shipment_id not in invalid_ids]

        previous_state = {shipment_id: prev_map.get(shipment_id, {}) for shipment_id in updated_ids}
        new_state = {shipment_id: curr_map.get(shipment_id, {}) for shipment_id in updated_ids}
        _append_activity(
            conn=conn,
            actor=actor,
            actor_role=actor_role,
            action="queue_bulk_update",
            entity_type="shipment",
            entity_id="bulk",
            payload={
                "shipment_ids": updated_ids,
                "status": normalized_status,
                "owner": normalized_owner,
                "note": normalized_note,
                "eta_action_at": normalized_eta,
                "updated_count": len(updated_ids),
                "missing_count": len(missing),
                "invalid_transition_count": len(invalid_transitions),
            },
            previous_state=previous_state,
            new_state=new_state,
            reason="bulk queue operation",
        )
        conn.commit()
        return {"updated": len(updated_ids), "missing": missing, "invalid_transitions": invalid_transitions}
    finally:
        conn.close()


def fetch_activity(shipment_id: str, path: Path = SERVICE_DB_PATH, limit: int = 30) -> List[Dict[str, object]]:
    init_service_store(path)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, actor, actor_role, action, entity_type, entity_id,
                   payload_json, previous_state_json, new_state_json,
                   reason, request_id, prev_hash, event_hash, created_at
            FROM service_activity_log
            WHERE entity_id = ? OR entity_id = 'bulk'
            ORDER BY id DESC
            LIMIT ?
            """,
            (shipment_id, limit),
        ).fetchall()
    finally:
        conn.close()

    parsed = []
    for row in rows:
        item = dict(row)
        for key in ["payload_json", "previous_state_json", "new_state_json"]:
            try:
                item[key.replace("_json", "")] = json.loads(item.get(key, "{}") or "{}")
            except json.JSONDecodeError:
                item[key.replace("_json", "")] = {"raw": item.get(key, "")}
        parsed.append(item)
    return parsed


def list_recent_activity(path: Path = SERVICE_DB_PATH, limit: int = 100) -> List[Dict[str, object]]:
    init_service_store(path)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, actor, actor_role, action, entity_type, entity_id,
                   payload_json, previous_state_json, new_state_json,
                   reason, request_id, prev_hash, event_hash, created_at
            FROM service_activity_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    parsed = []
    for row in rows:
        item = dict(row)
        for key in ["payload_json", "previous_state_json", "new_state_json"]:
            try:
                item[key.replace("_json", "")] = json.loads(item.get(key, "{}") or "{}")
            except json.JSONDecodeError:
                item[key.replace("_json", "")] = {"raw": item.get(key, "")}
        parsed.append(item)
    return parsed


def upsert_incident(
    title: str,
    severity: str,
    description: str,
    status: str = "Open",
    incident_id: str | None = None,
    owner: str = "",
    source_rule: str = "",
    actor: str = "operator",
    actor_role: str = "operator",
    path: Path = SERVICE_DB_PATH,
) -> str:
    init_service_store(path)
    if not has_permission(actor_role, "incident_manage"):
        raise PermissionError("actor does not have permission incident_manage")
    title = str(title).strip()
    description = str(description).strip()
    owner = str(owner).strip()
    source_rule = str(source_rule).strip()
    if not title:
        raise ValueError("incident title is required")
    if severity not in INCIDENT_SEVERITIES:
        raise ValueError(f"invalid severity: {severity}")
    if status not in INCIDENT_STATUSES:
        raise ValueError(f"invalid status: {status}")

    now = utc_now_iso()
    if not incident_id:
        incident_id = f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        prev = conn.execute(
            "SELECT incident_id, title, severity, description, status, owner, source_rule, updated_at FROM service_incidents WHERE incident_id = ?",
            (incident_id,),
        ).fetchone()

        conn.execute(
            """
            INSERT INTO service_incidents (incident_id, title, severity, description, status, owner, source_rule, opened_at, updated_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(incident_id) DO UPDATE SET
                title=excluded.title,
                severity=excluded.severity,
                description=excluded.description,
                status=excluded.status,
                owner=excluded.owner,
                source_rule=excluded.source_rule,
                updated_at=excluded.updated_at,
                closed_at=excluded.closed_at
            """,
            (
                incident_id,
                title,
                severity,
                description,
                status,
                owner,
                source_rule,
                now,
                now,
                now if status == "Closed" else "",
            ),
        )

        curr = conn.execute(
            "SELECT incident_id, title, severity, description, status, owner, source_rule, updated_at FROM service_incidents WHERE incident_id = ?",
            (incident_id,),
        ).fetchone()

        _append_activity(
            conn=conn,
            actor=actor,
            actor_role=actor_role,
            action="incident_upsert",
            entity_type="incident",
            entity_id=incident_id,
            payload={"title": title, "severity": severity, "status": status, "source_rule": source_rule},
            previous_state=dict(prev) if prev else {},
            new_state=dict(curr) if curr else {},
            reason="incident operation",
        )
        conn.commit()
    finally:
        conn.close()

    return incident_id


def upsert_incident_from_recommendation(
    recommendation: Dict[str, object],
    actor: str,
    actor_role: str = "operator",
    owner: str = "",
    path: Path = SERVICE_DB_PATH,
) -> Dict[str, object]:
    init_service_store(path)
    rule_id = str(recommendation.get("rule_id", "")).strip()
    title = str(recommendation.get("title", "Control tower recommendation")).strip()
    severity = str(recommendation.get("severity", "SEV-3")).strip() or "SEV-3"
    if severity not in INCIDENT_SEVERITIES:
        severity = "SEV-3"
    description = str(recommendation.get("description", "")).strip()
    if not description:
        description = f"Rule-based recommendation generated from {rule_id or 'unspecified rule'}."

    existing_incident_id = ""
    existing_status = "Open"
    if rule_id:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT incident_id, status
                FROM service_incidents
                WHERE source_rule = ? AND status IN ('Open', 'Monitoring')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (rule_id,),
            ).fetchone()
            if row:
                existing_incident_id = str(row["incident_id"])
                existing_status = str(row["status"] or "Open")
        finally:
            conn.close()

    incident_id = upsert_incident(
        title=title,
        severity=severity,
        description=description,
        status=existing_status if existing_incident_id else "Open",
        incident_id=existing_incident_id or None,
        owner=owner,
        source_rule=rule_id,
        actor=actor,
        actor_role=actor_role,
        path=path,
    )
    return {
        "incident_id": incident_id,
        "deduplicated": bool(existing_incident_id),
        "source_rule": rule_id,
    }


def list_incidents(path: Path = SERVICE_DB_PATH, limit: int = 50) -> List[Dict[str, object]]:
    init_service_store(path)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT incident_id, title, severity, description, status, owner, source_rule, opened_at, updated_at, closed_at
            FROM service_incidents
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def log_pipeline_run(
    run_id: str,
    started_at: str,
    finished_at: str,
    status: str,
    step_count: int,
    duration_sec: float,
    error_text: str = "",
    path: Path = SERVICE_DB_PATH,
) -> None:
    init_service_store(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        prev = conn.execute(
            "SELECT run_id, status, step_count, duration_sec, error_text FROM service_run_history WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        conn.execute(
            """
            INSERT INTO service_run_history (run_id, started_at, finished_at, status, step_count, duration_sec, error_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                started_at=excluded.started_at,
                finished_at=excluded.finished_at,
                status=excluded.status,
                step_count=excluded.step_count,
                duration_sec=excluded.duration_sec,
                error_text=excluded.error_text
            """,
            (run_id, started_at, finished_at, status, step_count, duration_sec, error_text),
        )

        curr = conn.execute(
            "SELECT run_id, status, step_count, duration_sec, error_text FROM service_run_history WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        _append_activity(
            conn=conn,
            actor="pipeline",
            actor_role="system",
            action="pipeline_run",
            entity_type="pipeline",
            entity_id=run_id,
            payload={"status": status, "step_count": step_count, "duration_sec": duration_sec},
            previous_state=dict(prev) if prev else {},
            new_state=dict(curr) if curr else {},
            reason="pipeline execution",
        )

        conn.commit()
    finally:
        conn.close()


def list_pipeline_runs(path: Path = SERVICE_DB_PATH, limit: int = 30) -> List[Dict[str, object]]:
    init_service_store(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT run_id, started_at, finished_at, status, step_count, duration_sec, error_text
            FROM service_run_history
            ORDER BY finished_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
