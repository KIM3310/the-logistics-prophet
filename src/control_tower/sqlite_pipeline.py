from __future__ import annotations

import csv
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List

from .config import RAW_DIR, SQLITE_PATH


TABLE_COLUMNS: Dict[str, List[str]] = {
    "customers": ["customer_id", "segment", "region", "join_date"],
    "products": ["product_id", "category", "weight_kg", "price_usd"],
    "warehouses": ["warehouse_id", "region", "capacity_tier", "avg_pick_minutes"],
    "carriers": ["carrier_id", "name", "mode", "reliability_score"],
    "orders": [
        "order_id",
        "customer_id",
        "product_id",
        "warehouse_id",
        "order_date",
        "promised_days",
        "order_value_usd",
        "peak_flag",
        "quantity",
    ],
    "shipments": [
        "shipment_id",
        "order_id",
        "carrier_id",
        "ship_date",
        "delivery_date",
        "distance_km",
        "weather_severity",
        "warehouse_load_pct",
        "transit_days_actual",
        "delivered_on_time",
        "delay_hours",
    ],
    "delay_events": ["event_id", "shipment_id", "event_type", "severity", "event_time"],
}


def _read_csv_rows(path: Path, columns: List[str]) -> Iterable[tuple]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield tuple(row[c] for c in columns)


def build_sqlite_marts(sqlite_path: Path = SQLITE_PATH, raw_dir: Path = RAW_DIR) -> dict:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = sqlite_path.with_name(f"{sqlite_path.name}.tmp-{secrets.token_hex(8)}")
    if temp_path.exists():
        temp_path.unlink()

    conn = sqlite3.connect(temp_path)
    try:
        _create_tables(conn)
        _load_raw_tables(conn, raw_dir)
        _create_indexes(conn)
        _create_feature_mart(conn)
        _create_kpi_tables(conn)
        conn.commit()

        counts = {}
        for table in ["orders", "shipments", "delay_events", "model_features", "kpi_daily"]:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        conn.close()

    os.replace(temp_path, sqlite_path)
    return counts


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            segment TEXT,
            region TEXT,
            join_date TEXT
        );

        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            category TEXT,
            weight_kg REAL,
            price_usd REAL
        );

        CREATE TABLE warehouses (
            warehouse_id TEXT PRIMARY KEY,
            region TEXT,
            capacity_tier TEXT,
            avg_pick_minutes REAL
        );

        CREATE TABLE carriers (
            carrier_id TEXT PRIMARY KEY,
            name TEXT,
            mode TEXT,
            reliability_score REAL
        );

        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT,
            product_id TEXT,
            warehouse_id TEXT,
            order_date TEXT,
            promised_days REAL,
            order_value_usd REAL,
            peak_flag INTEGER,
            quantity INTEGER
        );

        CREATE TABLE shipments (
            shipment_id TEXT PRIMARY KEY,
            order_id TEXT,
            carrier_id TEXT,
            ship_date TEXT,
            delivery_date TEXT,
            distance_km REAL,
            weather_severity REAL,
            warehouse_load_pct REAL,
            transit_days_actual REAL,
            delivered_on_time INTEGER,
            delay_hours REAL
        );

        CREATE TABLE delay_events (
            event_id TEXT PRIMARY KEY,
            shipment_id TEXT,
            event_type TEXT,
            severity TEXT,
            event_time TEXT
        );
        """
    )


def _load_raw_tables(conn: sqlite3.Connection, raw_dir: Path) -> None:
    for table, columns in TABLE_COLUMNS.items():
        csv_path = raw_dir / f"{table}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input CSV: {csv_path}")

        placeholders = ",".join("?" for _ in columns)
        sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
        conn.executemany(sql, _read_csv_rows(csv_path, columns))


def _create_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_orders_product ON orders(product_id);
        CREATE INDEX idx_orders_warehouse ON orders(warehouse_id);
        CREATE INDEX idx_shipments_order ON shipments(order_id);
        CREATE INDEX idx_shipments_ship_date ON shipments(ship_date);
        CREATE INDEX idx_shipments_carrier ON shipments(carrier_id);
        CREATE INDEX idx_delay_events_shipment ON delay_events(shipment_id);
        """
    )


def _create_feature_mart(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS model_features;
        CREATE TABLE model_features AS
        WITH delay_event_counts AS (
            SELECT
                shipment_id,
                COUNT(*) AS delay_event_count
            FROM delay_events
            GROUP BY shipment_id
        )
        SELECT
            s.shipment_id,
            o.order_id,
            s.ship_date,
            o.order_date,
            o.customer_id,
            o.product_id,
            o.warehouse_id,
            s.carrier_id,
            CAST(s.distance_km AS REAL) AS distance_km,
            CAST(s.weather_severity AS REAL) AS weather_severity,
            CAST(s.warehouse_load_pct AS REAL) AS warehouse_load_pct,
            CAST(c.reliability_score AS REAL) AS carrier_reliability_score,
            CAST(o.promised_days AS REAL) AS promised_days,
            CAST(o.order_value_usd AS REAL) AS order_value_usd,
            CAST(o.peak_flag AS INTEGER) AS peak_flag,
            CAST(w.avg_pick_minutes AS REAL) AS avg_pick_minutes,
            CAST(p.weight_kg AS REAL) AS product_weight_kg,
            CAST(COALESCE(dec.delay_event_count, 0) AS INTEGER) AS delay_event_count,
            CAST(1 - s.delivered_on_time AS INTEGER) AS delivered_late,
            CAST(s.delay_hours AS REAL) AS delay_hours
        FROM shipments s
        JOIN orders o ON s.order_id = o.order_id
        JOIN carriers c ON s.carrier_id = c.carrier_id
        JOIN warehouses w ON o.warehouse_id = w.warehouse_id
        JOIN products p ON o.product_id = p.product_id
        LEFT JOIN delay_event_counts dec ON s.shipment_id = dec.shipment_id;

        CREATE INDEX idx_model_features_ship_date ON model_features(ship_date);
        """
    )


def _create_kpi_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS kpi_daily;
        CREATE TABLE kpi_daily AS
        SELECT
            ship_date,
            COUNT(*) AS total_shipments,
            ROUND(AVG(CASE WHEN delivered_late = 0 THEN 1.0 ELSE 0.0 END), 4) AS on_time_rate,
            ROUND(AVG(delay_hours), 2) AS avg_delay_hours,
            SUM(CASE WHEN delivered_late = 1 THEN 1 ELSE 0 END) AS sla_breach_count
        FROM model_features
        GROUP BY ship_date
        ORDER BY ship_date;

        DROP TABLE IF EXISTS delay_root_cause;
        CREATE TABLE delay_root_cause AS
        SELECT
            de.event_type,
            COUNT(*) AS cnt
        FROM delay_events de
        GROUP BY de.event_type
        ORDER BY cnt DESC;
        """
    )
