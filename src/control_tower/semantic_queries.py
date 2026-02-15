from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rdflib import Graph, Namespace

EX = Namespace("http://example.com/supply-chain#")


def shipment_iri(shipment_id: str) -> str:
    shipment_id = str(shipment_id or "").strip()
    if not shipment_id:
        raise ValueError("shipment_id is required")
    # The instance graph uses IRIs shaped like:
    #   http://example.com/supply-chain#shipment/S0012345
    return f"{str(EX)}shipment/{shipment_id}"


def _iri(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    return f"<{raw}>" if raw else ""


@dataclass
class ShipmentEvidence:
    shipment: str
    carrier: str
    warehouse: str
    customer: str
    product: str
    risk_score: str
    risk_band: str
    distance_km: str
    weather_severity: str
    warehouse_load_pct: str
    delay_hours: str
    delivered_on_time: str
    delay_events: List[Dict[str, str]]

    def as_dict(self) -> Dict[str, object]:
        return {
            "shipment": self.shipment,
            "carrier": self.carrier,
            "warehouse": self.warehouse,
            "customer": self.customer,
            "product": self.product,
            "risk_score": self.risk_score,
            "risk_band": self.risk_band,
            "distance_km": self.distance_km,
            "weather_severity": self.weather_severity,
            "warehouse_load_pct": self.warehouse_load_pct,
            "delay_hours": self.delay_hours,
            "delivered_on_time": self.delivered_on_time,
            "delay_events": self.delay_events,
        }


def load_instance_graph(graph_path: Path) -> Graph:
    graph = Graph()
    graph.parse(str(graph_path), format="turtle")
    return graph


def query_shipment_evidence(graph: Graph, shipment_id: str, event_limit: int = 8) -> ShipmentEvidence:
    shipment_uri = shipment_iri(shipment_id)

    context_query = f"""
    PREFIX ex: <{str(EX)}>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT ?carrier ?warehouse ?customer ?product ?riskScore ?riskBand ?distanceKm ?weatherSeverity ?warehouseLoadPct ?delayHours ?onTime
    WHERE {{
      BIND({_iri(shipment_uri)} AS ?shipment)
      OPTIONAL {{ ?shipment ex:handledBy ?carrier . }}
      OPTIONAL {{
        ?order ex:hasShipment ?shipment ;
              ex:fulfilledFrom ?warehouse ;
              ex:placedBy ?customer ;
              ex:containsProduct ?product .
      }}
      OPTIONAL {{ ?shipment ex:distanceKm ?distanceKm . }}
      OPTIONAL {{ ?shipment ex:weatherSeverity ?weatherSeverity . }}
      OPTIONAL {{ ?shipment ex:warehouseLoadPct ?warehouseLoadPct . }}
      OPTIONAL {{ ?shipment ex:delayHours ?delayHours . }}
      OPTIONAL {{ ?shipment ex:deliveredOnTime ?onTime . }}
      OPTIONAL {{
        ?shipment ex:hasRiskState ?risk .
        ?risk ex:riskScore ?riskScore .
        ?risk ex:riskBand ?riskBand .
      }}
    }}
    LIMIT 1
    """

    carrier = ""
    warehouse = ""
    customer = ""
    product = ""
    risk_score = ""
    risk_band = ""
    distance_km = ""
    weather_severity = ""
    warehouse_load_pct = ""
    delay_hours = ""
    delivered_on_time = ""

    for row in graph.query(context_query):
        carrier = str(row[0] or "")
        warehouse = str(row[1] or "")
        customer = str(row[2] or "")
        product = str(row[3] or "")
        risk_score = str(row[4] or "")
        risk_band = str(row[5] or "")
        distance_km = str(row[6] or "")
        weather_severity = str(row[7] or "")
        warehouse_load_pct = str(row[8] or "")
        delay_hours = str(row[9] or "")
        delivered_on_time = str(row[10] or "")
        break

    events_query = f"""
    PREFIX ex: <{str(EX)}>
    SELECT ?event ?cause ?severity
    WHERE {{
      BIND({_iri(shipment_uri)} AS ?shipment)
      ?shipment ex:hasDelayEvent ?event .
      OPTIONAL {{ ?event ex:delayCause ?cause . }}
      OPTIONAL {{ ?event ex:severity ?severity . }}
    }}
    LIMIT {int(max(1, event_limit))}
    """

    delay_events: List[Dict[str, str]] = []
    for row in graph.query(events_query):
        delay_events.append(
            {
                "event": str(row[0] or ""),
                "cause": str(row[1] or ""),
                "severity": str(row[2] or ""),
            }
        )

    return ShipmentEvidence(
        shipment=shipment_uri,
        carrier=carrier,
        warehouse=warehouse,
        customer=customer,
        product=product,
        risk_score=risk_score,
        risk_band=risk_band,
        distance_km=distance_km,
        weather_severity=weather_severity,
        warehouse_load_pct=warehouse_load_pct,
        delay_hours=delay_hours,
        delivered_on_time=delivered_on_time,
        delay_events=delay_events,
    )

