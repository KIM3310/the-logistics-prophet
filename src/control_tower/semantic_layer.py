from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Dict, List

from rdflib import Graph, Literal, Namespace, RDF, XSD

from .config import DAILY_RISK_QUEUE_PATH, ONTOLOGY_DIR, RDF_INSTANCE_PATH, SPARQL_RESULTS_PATH, SQLITE_PATH

EX = Namespace("http://example.com/supply-chain#")


def _shipment_uri(shipment_id: str):
    return EX[f"shipment/{shipment_id}"]


def _order_uri(order_id: str):
    return EX[f"order/{order_id}"]


def _carrier_uri(carrier_id: str):
    return EX[f"carrier/{carrier_id}"]


def _warehouse_uri(warehouse_id: str):
    return EX[f"warehouse/{warehouse_id}"]


def _customer_uri(customer_id: str):
    return EX[f"customer/{customer_id}"]


def _product_uri(product_id: str):
    return EX[f"product/{product_id}"]


def _event_uri(event_id: str):
    return EX[f"event/{event_id}"]


def _risk_uri(shipment_id: str):
    return EX[f"risk/{shipment_id}"]


def build_instance_graph(
    sqlite_path: Path = SQLITE_PATH,
    ontology_path: Path = ONTOLOGY_DIR / "supply_chain.ttl",
    output_path: Path = RDF_INSTANCE_PATH,
    shipment_limit: int = 4000,
    risk_queue_path: Path = DAILY_RISK_QUEUE_PATH,
) -> Dict[str, object]:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    g = Graph()
    g.bind("ex", EX)

    try:
        carriers = conn.execute("SELECT carrier_id, reliability_score FROM carriers").fetchall()
        for row in carriers:
            carrier = _carrier_uri(str(row["carrier_id"]))
            g.add((carrier, RDF.type, EX.Carrier))
            g.add((carrier, EX.reliabilityScore, Literal(float(row["reliability_score"]), datatype=XSD.decimal)))

        warehouses = conn.execute("SELECT warehouse_id FROM warehouses").fetchall()
        for row in warehouses:
            warehouse = _warehouse_uri(str(row["warehouse_id"]))
            g.add((warehouse, RDF.type, EX.Warehouse))

        customers = conn.execute("SELECT customer_id FROM customers LIMIT 5000").fetchall()
        for row in customers:
            customer = _customer_uri(str(row["customer_id"]))
            g.add((customer, RDF.type, EX.Customer))

        products = conn.execute("SELECT product_id FROM products").fetchall()
        for row in products:
            product = _product_uri(str(row["product_id"]))
            g.add((product, RDF.type, EX.Product))

        shipments = conn.execute(
            """
            SELECT
                mf.shipment_id,
                mf.order_id,
                mf.customer_id,
                mf.product_id,
                mf.warehouse_id,
                mf.carrier_id,
                mf.order_date,
                mf.promised_days,
                mf.order_value_usd,
                mf.distance_km,
                mf.weather_severity,
                mf.warehouse_load_pct,
                mf.delivered_late,
                mf.delay_hours
            FROM model_features mf
            ORDER BY mf.ship_date DESC
            LIMIT ?
            """,
            (shipment_limit,),
        ).fetchall()

        for row in shipments:
            shipment_id = str(row["shipment_id"])
            order_id = str(row["order_id"])
            shipment = _shipment_uri(shipment_id)
            order = _order_uri(order_id)

            g.add((order, RDF.type, EX.Order))
            g.add((shipment, RDF.type, EX.Shipment))

            g.add((order, EX.hasShipment, shipment))
            g.add((order, EX.placedBy, _customer_uri(str(row["customer_id"]))))
            g.add((order, EX.containsProduct, _product_uri(str(row["product_id"]))))
            g.add((order, EX.fulfilledFrom, _warehouse_uri(str(row["warehouse_id"]))))

            g.add((shipment, EX.handledBy, _carrier_uri(str(row["carrier_id"]))))
            g.add((order, EX.orderDate, Literal(str(row["order_date"]), datatype=XSD.date)))
            g.add((order, EX.promisedDays, Literal(int(float(row["promised_days"])), datatype=XSD.integer)))
            g.add((order, EX.orderValueUsd, Literal(float(row["order_value_usd"]), datatype=XSD.decimal)))
            g.add((shipment, EX.distanceKm, Literal(float(row["distance_km"]), datatype=XSD.decimal)))
            g.add((shipment, EX.weatherSeverity, Literal(float(row["weather_severity"]), datatype=XSD.decimal)))
            g.add((shipment, EX.warehouseLoadPct, Literal(float(row["warehouse_load_pct"]), datatype=XSD.decimal)))
            g.add((shipment, EX.deliveredOnTime, Literal(bool(1 - int(float(row["delivered_late"]))), datatype=XSD.boolean)))
            g.add((shipment, EX.delayHours, Literal(float(row["delay_hours"]), datatype=XSD.decimal)))

            if int(float(row["delivered_late"])) == 1:
                g.add((shipment, EX.violates, EX.SLA_Standard))

        delay_events = conn.execute(
            """
            SELECT de.event_id, de.shipment_id, de.event_type, de.severity
            FROM delay_events de
            JOIN shipments s ON de.shipment_id = s.shipment_id
            ORDER BY de.event_time DESC
            LIMIT 6000
            """
        ).fetchall()

        for row in delay_events:
            event = _event_uri(str(row["event_id"]))
            shipment = _shipment_uri(str(row["shipment_id"]))
            g.add((event, RDF.type, EX.DelayEvent))
            g.add((event, EX.delayCause, Literal(str(row["event_type"]), datatype=XSD.string)))
            g.add((event, EX.severity, Literal(str(row["severity"]), datatype=XSD.string)))
            g.add((shipment, EX.hasDelayEvent, event))

        if risk_queue_path.exists():
            with risk_queue_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    shipment_id = str(row.get("shipment_id", ""))
                    if not shipment_id:
                        continue
                    risk = _risk_uri(shipment_id)
                    shipment = _shipment_uri(shipment_id)
                    g.add((risk, RDF.type, EX.RiskState))
                    g.add((shipment, EX.hasRiskState, risk))
                    try:
                        score = float(row.get("risk_score", 0.0))
                    except (TypeError, ValueError):
                        score = 0.0
                    band = str(row.get("risk_band", ""))
                    g.add((risk, EX.riskScore, Literal(score, datatype=XSD.decimal)))
                    g.add((risk, EX.riskBand, Literal(band, datatype=XSD.string)))

    finally:
        conn.close()

    if ontology_path.exists():
        g.parse(ontology_path, format="turtle")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(output_path), format="turtle")

    return {
        "triples": len(g),
        "output_path": str(output_path),
        "shipments_materialized": len(shipments),
    }


def _query_to_dicts(graph: Graph, query: str, key_order: List[str]) -> List[Dict[str, object]]:
    rows = []
    for record in graph.query(query):
        row = {}
        for i, key in enumerate(key_order):
            value = record[i]
            if value is None:
                row[key] = None
            else:
                row[key] = str(value)
        rows.append(row)
    return rows


def run_competency_queries(
    graph_path: Path = RDF_INSTANCE_PATH,
    output_path: Path = SPARQL_RESULTS_PATH,
) -> Dict[str, object]:
    graph = Graph()
    graph.parse(graph_path, format="turtle")

    queries = [
        {
            "id": "CQ1_high_delay_shipments",
            "description": "Shipments with delayHours >= 24",
            "query": """
            PREFIX ex: <http://example.com/supply-chain#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            SELECT ?shipment ?delayHours
            WHERE {
              ?shipment a ex:Shipment ; ex:delayHours ?delayHours .
              FILTER (?delayHours >= "24"^^xsd:decimal)
            }
            ORDER BY DESC(?delayHours)
            LIMIT 15
            """,
            "columns": ["shipment", "delayHours"],
        },
        {
            "id": "CQ2_carrier_sla_violations",
            "description": "Carrier-level SLA violation counts",
            "query": """
            PREFIX ex: <http://example.com/supply-chain#>
            SELECT ?carrier (COUNT(?shipment) AS ?violations)
            WHERE {
              ?shipment a ex:Shipment ; ex:handledBy ?carrier ; ex:violates ex:SLA_Standard .
            }
            GROUP BY ?carrier
            ORDER BY DESC(?violations)
            LIMIT 10
            """,
            "columns": ["carrier", "violations"],
        },
        {
            "id": "CQ3_warehouse_delay_hotspots",
            "description": "Warehouses with most delayed shipments",
            "query": """
            PREFIX ex: <http://example.com/supply-chain#>
            SELECT ?warehouse (COUNT(?shipment) AS ?delayed)
            WHERE {
              ?order a ex:Order ; ex:fulfilledFrom ?warehouse ; ex:hasShipment ?shipment .
              ?shipment ex:deliveredOnTime ?onTime .
              FILTER (?onTime = false)
            }
            GROUP BY ?warehouse
            ORDER BY DESC(?delayed)
            LIMIT 10
            """,
            "columns": ["warehouse", "delayed"],
        },
        {
            "id": "CQ4_top_predicted_risk_shipments",
            "description": "Shipments with highest predicted risk scores",
            "query": """
            PREFIX ex: <http://example.com/supply-chain#>
            SELECT ?shipment ?score ?band
            WHERE {
              ?shipment a ex:Shipment ; ex:hasRiskState ?risk .
              ?risk ex:riskScore ?score ; ex:riskBand ?band .
            }
            ORDER BY DESC(?score)
            LIMIT 15
            """,
            "columns": ["shipment", "score", "band"],
        },
    ]

    result_queries: List[Dict[str, object]] = []
    for q in queries:
        rows = _query_to_dicts(graph, q["query"], q["columns"])
        result_queries.append(
            {
                "id": q["id"],
                "description": q["description"],
                "columns": q["columns"],
                "row_count": len(rows),
                "rows": rows,
            }
        )

    payload = {
        "generated_at_utc": datetime_utc_iso(),
        "graph_path": str(graph_path),
        "triple_count": len(graph),
        "queries": result_queries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    return payload


def datetime_utc_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def load_sparql_results(path: Path = SPARQL_RESULTS_PATH) -> Dict[str, object]:
    if not path.exists():
        return {"queries": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
