from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "model"
OUTPUT_DIR = DATA_DIR / "output"
SEMANTIC_DIR = DATA_DIR / "semantic"
ONTOLOGY_DIR = PROJECT_ROOT / "ontology"
MONITORING_DIR = PROJECT_ROOT / "monitoring"

SQLITE_PATH = PROCESSED_DIR / "control_tower.db"
SERVICE_DB_PATH = PROCESSED_DIR / "service_store.db"
MODEL_PATH = MODEL_DIR / "model_artifact.json"
MODEL_BINARY_PATH = MODEL_DIR / "selected_model.pkl"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison.json"
SHAP_GLOBAL_PATH = OUTPUT_DIR / "shap_global_importance.csv"
SHAP_LOCAL_PATH = OUTPUT_DIR / "shap_local_explanations.csv"
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "training_summary.json"
DAILY_RISK_QUEUE_PATH = OUTPUT_DIR / "daily_risk_queue.csv"
MONITORING_METRICS_PATH = OUTPUT_DIR / "monitoring_metrics.json"
OPS_REPORT_PATH = OUTPUT_DIR / "ops_report.html"
QUALITY_REPORT_PATH = OUTPUT_DIR / "data_quality_report.json"
SPARQL_RESULTS_PATH = OUTPUT_DIR / "sparql_results.json"
RDF_INSTANCE_PATH = SEMANTIC_DIR / "instance_graph.ttl"
PIPELINE_STATUS_PATH = OUTPUT_DIR / "pipeline_status.json"
DATADOG_SERIES_PAYLOAD_PATH = OUTPUT_DIR / "datadog_series_payload.json"
DATADOG_REPLAY_PAYLOAD_PATH = OUTPUT_DIR / "datadog_replay_payload.json"

FEATURE_COLUMNS = [
    "distance_km",
    "weather_severity",
    "warehouse_load_pct",
    "carrier_reliability_score",
    "promised_days",
    "order_value_usd",
    "peak_flag",
    "avg_pick_minutes",
    "product_weight_kg",
]
TARGET_COLUMN = "delivered_late"

DEFAULT_SEED = 42
DEFAULT_DAYS = 120
DEFAULT_ORDERS_PER_DAY = 120

SCHEMA_VERSION = "1.0.0"
PIPELINE_SERVICE = "semantic-control-tower"
PIPELINE_ENV = "portfolio"
