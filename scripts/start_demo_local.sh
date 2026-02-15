#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-8501}"

VENV_DIR="${VENV_DIR:-.venv}"
VENV_PY="$VENV_DIR/bin/python"

FORCE_PIPELINE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE_PIPELINE=1
fi

echo "[demo] root: $ROOT_DIR"
echo "[demo] python: $PYTHON_BIN"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[demo] creating venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "[demo] installing dependencies"
"$VENV_PY" -m pip install -r requirements.txt >/dev/null

need_pipeline=0
required_files=(
  "data/output/monitoring_metrics.json"
  "data/output/data_quality_report.json"
  "data/output/sparql_results.json"
  "data/output/shap_global_importance.csv"
  "data/output/shap_local_explanations.csv"
  "data/semantic/instance_graph.ttl"
  "data/processed/service_store.db"
)

for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    need_pipeline=1
    break
  fi
done

if [[ "$FORCE_PIPELINE" -eq 1 || "$need_pipeline" -eq 1 ]]; then
  echo "[demo] running pipeline"
  "$VENV_PY" scripts/run_pipeline.py >/dev/null
else
  echo "[demo] pipeline artifacts found (skip)"
fi

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "[demo] starting dashboard: http://127.0.0.1:${PORT}"
exec "$VENV_PY" -m streamlit run app/dashboard.py \
  --server.address 127.0.0.1 \
  --server.port "$PORT"

