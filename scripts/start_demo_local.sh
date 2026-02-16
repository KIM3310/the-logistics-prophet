#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-8501}"
KILL_PORT="${KILL_PORT:-0}"
LOG_LEVEL="${LOG_LEVEL:-info}"
AUTO_PORT="${AUTO_PORT:-1}"
OPEN_BROWSER="${OPEN_BROWSER:-0}"

VENV_DIR="${VENV_DIR:-.venv}"
VENV_PY="$VENV_DIR/bin/python"

FORCE_PIPELINE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE_PIPELINE=1 ;;
    --kill-port) KILL_PORT=1 ;;
    --debug) LOG_LEVEL=debug ;;
    --no-auto-port) AUTO_PORT=0 ;;
    --open) OPEN_BROWSER=1 ;;
  esac
done

echo "[demo] root: $ROOT_DIR"
echo "[demo] python: $PYTHON_BIN"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[demo] creating venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "[demo] installing dependencies"
"$VENV_PY" -m pip install --disable-pip-version-check -r requirements.txt >/dev/null

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

existing_pid="$(lsof -t -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$existing_pid" ]]; then
  if [[ "$KILL_PORT" -eq 1 ]]; then
    echo "[demo] port $PORT is in use (pid=$existing_pid) -> killing"
    # NOTE: lsof may return multiple PIDs (one per line). Intentionally unquoted.
    kill $existing_pid 2>/dev/null || true

    # Give the OS a moment to release the socket.
    released=0
    for _i in 1 2 3 4 5 6 7 8 9 10; do
      pid_check="$(lsof -t -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
      if [[ -z "$pid_check" ]]; then
        released=1
        break
      fi
      sleep 0.2
    done
    if [[ "$released" -ne 1 ]]; then
      if [[ "$AUTO_PORT" -eq 1 ]]; then
        start_port="$PORT"
        found=0
        for p in $(seq "$start_port" $((start_port + 20))); do
          pid="$(lsof -t -nP -iTCP:"$p" -sTCP:LISTEN 2>/dev/null || true)"
          if [[ -z "$pid" ]]; then
            PORT="$p"
            found=1
            break
          fi
        done
        if [[ "$found" -eq 0 ]]; then
          echo "[demo] ERROR: port $start_port still busy and no free port found nearby"
          exit 1
        fi
        echo "[demo] port $start_port still busy -> using free port $PORT"
      else
        echo "[demo] ERROR: port $PORT is still in use after kill attempt"
        exit 1
      fi
    fi
  else
    if [[ "$AUTO_PORT" -eq 1 ]]; then
      start_port="$PORT"
      found=0
      for p in $(seq "$start_port" $((start_port + 20))); do
        pid="$(lsof -t -nP -iTCP:"$p" -sTCP:LISTEN 2>/dev/null || true)"
        if [[ -z "$pid" ]]; then
          PORT="$p"
          found=1
          break
        fi
      done
      if [[ "$found" -eq 0 ]]; then
        echo "[demo] ERROR: no free port found near $start_port"
        exit 1
      fi
      echo "[demo] port $start_port is busy (pid=$existing_pid) -> using free port $PORT"
    else
      echo "[demo] ERROR: port $PORT is already in use (pid=$existing_pid)"
      echo "[demo] Fix: run with '--kill-port' or choose a different port:"
      echo "[demo]   PORT=8502 bash scripts/start_demo_local.sh"
      exit 1
    fi
  fi
fi

url="http://127.0.0.1:${PORT}"
echo "[demo] starting dashboard: $url"

if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  if command -v open >/dev/null 2>&1; then
    (open "$url" >/dev/null 2>&1 || true) &
  elif command -v xdg-open >/dev/null 2>&1; then
    (xdg-open "$url" >/dev/null 2>&1 || true) &
  fi
fi

exec "$VENV_PY" -m streamlit run app/dashboard.py \
  --server.headless true \
  --server.address 127.0.0.1 \
  --server.port "$PORT" \
  --logger.level "$LOG_LEVEL"
