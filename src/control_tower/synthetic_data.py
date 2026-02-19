from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import DEFAULT_DAYS, DEFAULT_ORDERS_PER_DAY, DEFAULT_SEED, RAW_DIR


@dataclass
class GenerationSummary:
    start_date: str
    end_date: str
    customers: int
    products: int
    warehouses: int
    carriers: int
    orders: int
    shipments: int
    delay_events: int


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_start_date(value: str) -> date:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("start_date must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError("start_date must be YYYY-MM-DD") from exc
    return parsed


def _build_customers(rng: np.random.Generator, n_customers: int, start_date: date) -> List[dict]:
    segments = ["B2C", "B2B", "Enterprise"]
    regions = ["NE", "SE", "MW", "SW", "W"]
    rows = []
    for i in range(1, n_customers + 1):
        join_offset = int(rng.integers(30, 1000))
        rows.append(
            {
                "customer_id": f"C{i:05d}",
                "segment": str(rng.choice(segments, p=[0.62, 0.28, 0.10])),
                "region": str(rng.choice(regions)),
                "join_date": (start_date - timedelta(days=join_offset)).isoformat(),
            }
        )
    return rows


def _build_products(rng: np.random.Generator, n_products: int) -> List[dict]:
    categories = ["Electronics", "Home", "Beauty", "Sports", "Office"]
    rows = []
    for i in range(1, n_products + 1):
        category = str(rng.choice(categories))
        weight = float(np.round(rng.uniform(0.2, 12.0), 2))
        base_price = {
            "Electronics": (40, 1400),
            "Home": (15, 500),
            "Beauty": (6, 120),
            "Sports": (18, 700),
            "Office": (8, 320),
        }[category]
        price = float(np.round(rng.uniform(*base_price), 2))
        rows.append(
            {
                "product_id": f"P{i:04d}",
                "category": category,
                "weight_kg": weight,
                "price_usd": price,
            }
        )
    return rows


def _build_warehouses(rng: np.random.Generator) -> List[dict]:
    regions = ["NE", "SE", "MW", "SW", "W", "NE", "SE", "MW"]
    tiers = ["L", "L", "M", "M", "L", "S", "S", "M"]
    rows = []
    for i, (region, tier) in enumerate(zip(regions, tiers), start=1):
        if tier == "L":
            avg_pick = int(rng.integers(11, 18))
        elif tier == "M":
            avg_pick = int(rng.integers(16, 24))
        else:
            avg_pick = int(rng.integers(21, 30))
        rows.append(
            {
                "warehouse_id": f"W{i:03d}",
                "region": region,
                "capacity_tier": tier,
                "avg_pick_minutes": avg_pick,
            }
        )
    return rows


def _build_carriers() -> List[dict]:
    return [
        {"carrier_id": "K1", "name": "SkyShip", "mode": "Air", "reliability_score": 0.92},
        {"carrier_id": "K2", "name": "RoadSprint", "mode": "Ground", "reliability_score": 0.84},
        {"carrier_id": "K3", "name": "BlueCargo", "mode": "Sea", "reliability_score": 0.79},
        {"carrier_id": "K4", "name": "RapidRail", "mode": "Rail", "reliability_score": 0.87},
        {"carrier_id": "K5", "name": "PrimeRoute", "mode": "Ground", "reliability_score": 0.90},
        {"carrier_id": "K6", "name": "EconomyLine", "mode": "Ground", "reliability_score": 0.75},
    ]


def generate_synthetic_data(
    seed: int = DEFAULT_SEED,
    days: int = DEFAULT_DAYS,
    orders_per_day: int = DEFAULT_ORDERS_PER_DAY,
    start_date: date | None = None,
) -> GenerationSummary:
    rng = np.random.default_rng(seed)

    if start_date is None:
        anchored = str(os.getenv("LP_ANCHOR_DATE", "")).strip()
        if anchored:
            start_date = _parse_start_date(anchored)
        else:
            start_date = date.today() - timedelta(days=days)

    date_axis = [start_date + timedelta(days=i) for i in range(days)]
    end_date = date_axis[-1]

    n_customers = max(1200, (days * orders_per_day) // 5)
    n_products = 180

    customers = _build_customers(rng, n_customers=n_customers, start_date=start_date)
    products = _build_products(rng, n_products=n_products)
    warehouses = _build_warehouses(rng)
    carriers = _build_carriers()

    customer_ids = [r["customer_id"] for r in customers]
    product_ids = [r["product_id"] for r in products]
    warehouse_ids = [r["warehouse_id"] for r in warehouses]
    carrier_ids = [r["carrier_id"] for r in carriers]

    product_map: Dict[str, dict] = {r["product_id"]: r for r in products}
    warehouse_map: Dict[str, dict] = {r["warehouse_id"]: r for r in warehouses}
    carrier_map: Dict[str, dict] = {r["carrier_id"]: r for r in carriers}

    orders: List[dict] = []
    shipments: List[dict] = []
    delay_events: List[dict] = []

    order_seq = 1
    shipment_seq = 1
    event_seq = 1

    for d in date_axis:
        weekday = d.weekday()
        seasonal_peak = 1 if d.month in (11, 12) else 0
        weekly_peak = 1 if weekday in (0, 1) else 0

        day_multiplier = 1.0 + 0.12 * seasonal_peak + 0.06 * weekly_peak
        daily_orders = int(max(20, rng.normal(orders_per_day * day_multiplier, 10)))

        for _ in range(daily_orders):
            order_id = f"O{order_seq:07d}"
            shipment_id = f"S{shipment_seq:07d}"

            customer_id = str(rng.choice(customer_ids))
            product_id = str(rng.choice(product_ids))
            warehouse_id = str(rng.choice(warehouse_ids))
            carrier_id = str(
                rng.choice(
                    carrier_ids,
                    p=np.array([0.18, 0.20, 0.10, 0.12, 0.24, 0.16]),
                )
            )

            product = product_map[product_id]
            warehouse = warehouse_map[warehouse_id]
            carrier = carrier_map[carrier_id]

            distance_km = float(np.round(np.clip(rng.normal(620, 300), 35, 2200), 1))
            weather_severity = float(np.round(np.clip(rng.beta(2.3, 5.2), 0, 1), 3))

            base_load = {"L": 0.64, "M": 0.72, "S": 0.78}[warehouse["capacity_tier"]]
            load_noise = float(rng.normal(0.0, 0.09))
            warehouse_load_pct = float(np.round(np.clip(base_load + load_noise, 0.35, 0.98), 3))

            peak_flag = 1 if seasonal_peak or weekly_peak else 0
            quantity = int(rng.integers(1, 5))
            order_value = float(np.round(quantity * product["price_usd"] * rng.uniform(0.9, 1.08), 2))

            promised_days = int(np.clip(np.round(distance_km / 430 + rng.normal(1.2, 0.5)), 1, 8))

            risk_logit = (
                -4.7
                + 3.5 * weather_severity
                + 3.3 * warehouse_load_pct
                + 3.2 * (1.0 - carrier["reliability_score"])
                + 0.9 * peak_flag
                + 0.0008 * distance_km
                - 0.3 * promised_days
                + 0.00035 * order_value
                + 0.04 * warehouse["avg_pick_minutes"]
                + 0.05 * float(product["weight_kg"])
                + rng.normal(0.0, 0.07)
            )
            delay_probability = _sigmoid(risk_logit)
            delayed = int(rng.random() < delay_probability)

            if delayed:
                delay_days = int(rng.integers(1, 4))
                transit_days_actual = promised_days + delay_days
                delay_hours = int(delay_days * 24 + rng.integers(1, 12))
            else:
                transit_days_actual = max(1, promised_days - int(rng.integers(0, 2)))
                delay_hours = 0

            ship_date = d.isoformat()
            delivery_date = (d + timedelta(days=transit_days_actual)).isoformat()

            orders.append(
                {
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "product_id": product_id,
                    "warehouse_id": warehouse_id,
                    "order_date": d.isoformat(),
                    "promised_days": promised_days,
                    "order_value_usd": order_value,
                    "peak_flag": peak_flag,
                    "quantity": quantity,
                }
            )

            shipments.append(
                {
                    "shipment_id": shipment_id,
                    "order_id": order_id,
                    "carrier_id": carrier_id,
                    "ship_date": ship_date,
                    "delivery_date": delivery_date,
                    "distance_km": distance_km,
                    "weather_severity": weather_severity,
                    "warehouse_load_pct": warehouse_load_pct,
                    "transit_days_actual": transit_days_actual,
                    "delivered_on_time": 0 if delayed else 1,
                    "delay_hours": delay_hours,
                }
            )

            if delayed:
                primary_event = str(rng.choice(["Weather", "Capacity", "Carrier", "Routing", "Customs"]))
                severity = str(rng.choice(["Medium", "High", "Critical"], p=[0.40, 0.44, 0.16]))
                delay_events.append(
                    {
                        "event_id": f"E{event_seq:08d}",
                        "shipment_id": shipment_id,
                        "event_type": primary_event,
                        "severity": severity,
                        "event_time": (d + timedelta(days=1)).isoformat(),
                    }
                )
                event_seq += 1

                if rng.random() < 0.28:
                    secondary_event = str(rng.choice(["Capacity", "Carrier", "Routing"]))
                    delay_events.append(
                        {
                            "event_id": f"E{event_seq:08d}",
                            "shipment_id": shipment_id,
                            "event_type": secondary_event,
                            "severity": "Medium",
                            "event_time": (d + timedelta(days=1)).isoformat(),
                        }
                    )
                    event_seq += 1

            order_seq += 1
            shipment_seq += 1

    _write_csv(
        RAW_DIR / "customers.csv",
        ["customer_id", "segment", "region", "join_date"],
        customers,
    )
    _write_csv(
        RAW_DIR / "products.csv",
        ["product_id", "category", "weight_kg", "price_usd"],
        products,
    )
    _write_csv(
        RAW_DIR / "warehouses.csv",
        ["warehouse_id", "region", "capacity_tier", "avg_pick_minutes"],
        warehouses,
    )
    _write_csv(
        RAW_DIR / "carriers.csv",
        ["carrier_id", "name", "mode", "reliability_score"],
        carriers,
    )
    _write_csv(
        RAW_DIR / "orders.csv",
        [
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
        orders,
    )
    _write_csv(
        RAW_DIR / "shipments.csv",
        [
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
        shipments,
    )
    _write_csv(
        RAW_DIR / "delay_events.csv",
        ["event_id", "shipment_id", "event_type", "severity", "event_time"],
        delay_events,
    )

    return GenerationSummary(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        customers=len(customers),
        products=len(products),
        warehouses=len(warehouses),
        carriers=len(carriers),
        orders=len(orders),
        shipments=len(shipments),
        delay_events=len(delay_events),
    )
