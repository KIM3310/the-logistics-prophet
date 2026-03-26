#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import SERVICE_DB_PATH
from control_tower.service_store import create_or_update_user, list_users


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage service users")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--username", default="")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--role", default="viewer")
    parser.add_argument("--password", default="")
    parser.add_argument("--active", choices=["yes", "no"], default="yes")
    parser.add_argument("--actor", default="admin")
    parser.add_argument("--actor-role", default="admin")
    args = parser.parse_args()

    if args.list:
        users = list_users(path=SERVICE_DB_PATH, include_inactive=True)
        print(json.dumps({"users": users}, ensure_ascii=True, indent=2))
        return

    if not args.username or not args.display_name or not args.password:
        raise SystemExit("For upsert: provide --username --display-name --password")

    create_or_update_user(
        username=args.username,
        display_name=args.display_name,
        role=args.role,
        password=args.password,
        actor=args.actor,
        actor_role=args.actor_role,
        is_active=(args.active == "yes"),
        path=SERVICE_DB_PATH,
    )

    users = list_users(path=SERVICE_DB_PATH, include_inactive=True)
    print(json.dumps({"status": "ok", "users": users}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
