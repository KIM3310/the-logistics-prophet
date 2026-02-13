from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import SERVICE_DB_PATH
from control_tower.service_store import (
    allowed_next_statuses,
    authenticate_user,
    bulk_update_queue_actions,
    create_or_update_user,
    derive_incident_recommendations,
    fetch_activity,
    fetch_ops_health,
    fetch_queue,
    fetch_queue_summary,
    fetch_service_core_snapshot,
    fetch_service_core_worklist,
    fetch_workflow_sla_snapshot,
    has_permission,
    init_service_store,
    list_service_core_terms,
    list_incidents,
    list_pipeline_runs,
    list_recent_activity,
    list_users,
    service_core_term_for_status,
    update_queue_action,
    upsert_incident,
    upsert_incident_from_recommendation,
    verify_audit_chain,
)

METRICS_PATH = ROOT / "data" / "output" / "monitoring_metrics.json"
QUALITY_PATH = ROOT / "data" / "output" / "data_quality_report.json"
SPARQL_PATH = ROOT / "data" / "output" / "sparql_results.json"
SHAP_GLOBAL_PATH = ROOT / "data" / "output" / "shap_global_importance.csv"
TRAINING_PATH = ROOT / "data" / "output" / "training_summary.json"
PIPELINE_STATUS_PATH = ROOT / "data" / "output" / "pipeline_status.json"
SQLITE_PATH = ROOT / "data" / "processed" / "control_tower.db"

STATUS_TEXT = {
    "New": "Start",
    "Investigating": "Check",
    "Mitigating": "Fix",
    "Resolved": "Done",
    "Dismissed": "Skip",
}

REASON_TEXT = {
    "critical_unassigned": "Critical + no owner",
    "eta_breached": "Past ETA",
    "stale_24h": "No update 24h+",
    "priority_backlog": "High-priority backlog",
}

DRIVER_TEXT = {
    "distance_km": "Distance",
    "weather_severity": "Weather",
    "warehouse_load_pct": "Warehouse Load",
    "carrier_reliability_score": "Carrier Risk",
    "promised_days": "Tight Promise",
    "order_value_usd": "High Value",
    "peak_flag": "Peak Day",
    "avg_pick_minutes": "Slow Pick",
    "product_weight_kg": "Heavy Item",
}

FORWARD_STATUS = {
    "New": "Investigating",
    "Investigating": "Mitigating",
    "Mitigating": "Resolved",
}

st.set_page_config(
    page_title="The Logistics Prophet",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_shap_global(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["feature", "mean_abs_contribution"])
    frame = pd.read_csv(path)
    if "mean_abs_contribution" in frame.columns:
        frame["mean_abs_contribution"] = pd.to_numeric(frame["mean_abs_contribution"], errors="coerce").fillna(0.0)
    return frame.sort_values("mean_abs_contribution", ascending=False)


@st.cache_data(show_spinner=False)
def load_kpi_series(limit: int = 40) -> pd.DataFrame:
    if not SQLITE_PATH.exists():
        return pd.DataFrame()
    import sqlite3

    conn = sqlite3.connect(SQLITE_PATH)
    try:
        frame = pd.read_sql_query(
            """
            SELECT ship_date, on_time_rate, avg_delay_hours, sla_breach_count, total_shipments
            FROM kpi_daily
            ORDER BY ship_date DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    finally:
        conn.close()

    frame = frame.sort_values("ship_date").reset_index(drop=True)
    return frame


def load_service_queue() -> pd.DataFrame:
    rows = fetch_queue(path=SERVICE_DB_PATH, limit=1000)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "shipment_id",
                "ship_date",
                "order_id",
                "risk_score",
                "risk_band",
                "status",
                "owner",
                "key_driver",
                "driver_2",
                "driver_3",
                "recommended_action",
                "note",
                "eta_action_at",
                "updated_at",
            ]
        )
    frame["risk_score"] = pd.to_numeric(frame["risk_score"], errors="coerce").fillna(0.0)
    return frame.sort_values("risk_score", ascending=False).reset_index(drop=True)


def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Space+Grotesk:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --paper: #f1ede3;
  --ink: #13110d;
  --muted: #5c554a;
  --line: #d9d0c4;
  --ocean: #0f625b;
  --clay: #ad5e4a;
  --glass: rgba(255,255,255,0.68);
}

#MainMenu, header, footer { visibility: hidden; }

.stApp {
  color: var(--ink);
  background:
    radial-gradient(1100px 500px at -5% -12%, #ddeae6 0%, transparent 60%),
    radial-gradient(900px 500px at 104% 3%, #eddcd2 0%, transparent 56%),
    linear-gradient(180deg, #f6f2ea 0%, #ece5d7 100%);
}

* {
  font-family: "Space Grotesk", "Avenir Next", sans-serif;
}

.block-container {
  max-width: 1240px;
  padding-top: 1.2rem;
  padding-bottom: 2.5rem;
}

.hero {
  position: relative;
  border: 1px solid var(--line);
  border-radius: 28px;
  background: var(--glass);
  backdrop-filter: blur(7px);
  overflow: hidden;
  box-shadow: 0 30px 50px rgba(46,36,28,0.1);
  margin-bottom: 16px;
}

.hero::before {
  content: "";
  position: absolute;
  width: 460px;
  height: 460px;
  border-radius: 50%;
  top: -260px;
  right: -120px;
  background: radial-gradient(circle at center, rgba(15,98,91,.24) 0%, rgba(15,98,91,.04) 55%, transparent 75%);
}

.hero-inner {
  padding: 30px 30px 26px 30px;
  position: relative;
  z-index: 2;
}

.kicker {
  font-size: .72rem;
  letter-spacing: .12rem;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
}

.title-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
}

.hero h1 {
  font-family: "Cormorant Garamond", Georgia, serif;
  margin: 0;
  font-size: clamp(2.2rem, 5.2vw, 4.4rem);
  line-height: .96;
  letter-spacing: -.02em;
  font-weight: 600;
}

.hero-sub {
  margin-top: 12px;
  margin-bottom: 0;
  color: var(--muted);
  max-width: 760px;
  font-size: 1rem;
}

.stamp {
  font-family: "IBM Plex Mono", monospace;
  font-size: .72rem;
  border: 1px solid #c9dfdb;
  background: #dff0ec;
  color: #235d57;
  border-radius: 999px;
  padding: 8px 12px;
  white-space: nowrap;
}

.mosaic {
  display: grid;
  grid-template-columns: 1.2fr 1.2fr 0.9fr 0.9fr;
  gap: 10px;
  margin-top: 12px;
}

.tile {
  background: rgba(255,255,255,.78);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px 12px;
  min-height: 92px;
}

.tile.wide {
  grid-column: span 2;
}

.tile-label {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .06rem;
  font-size: .68rem;
}

.tile-value {
  margin-top: 8px;
  font-size: 1.55rem;
  font-weight: 600;
}

.tile-meta {
  margin-top: 4px;
  font-size: .75rem;
  color: var(--muted);
}

.panel {
  border: 1px solid var(--line);
  border-radius: 20px;
  background: rgba(255,255,255,.72);
  padding: 14px 16px 12px 16px;
  box-shadow: 0 18px 30px rgba(44,35,25,0.08);
  margin-top: 12px;
}

.panel h3 {
  margin: 2px 0 12px 0;
  font-weight: 500;
  font-size: 1rem;
}

.query-chip {
  font-family: "IBM Plex Mono", monospace;
  display: inline-block;
  font-size: .72rem;
  border-radius: 999px;
  padding: 3px 9px;
  margin-bottom: 8px;
  border: 1px solid #d5cdc0;
  background: #f9f4ea;
}

.badge {
  border-radius: 999px;
  font-family: "IBM Plex Mono", monospace;
  font-size: .72rem;
  padding: 4px 10px;
}

.badge.ok { background: #e5efda; color: #4f6634; }
.badge.warn { background: #f3e8ce; color: #7c5d1b; }
.badge.fail { background: #efd6cf; color: #894335; }

.risk-table {
  width: 100%;
  border-collapse: collapse;
}

.risk-table th {
  text-align: left;
  font-size: .72rem;
  text-transform: uppercase;
  letter-spacing: .06rem;
  color: var(--muted);
  border-bottom: 1px solid var(--line);
  padding: 9px 8px;
}

.risk-table td {
  border-bottom: 1px solid #e7e0d3;
  padding: 8px;
  font-size: .84rem;
  vertical-align: top;
}

.band {
  font-family: "IBM Plex Mono", monospace;
  border-radius: 999px;
  padding: 3px 9px;
  display: inline-block;
  font-size: .72rem;
}

.band-critical { background: #efd6cf; color: #8a4535; }
.band-high { background: #f4e7cb; color: #8a6324; }
.band-medium { background: #dcebe7; color: #205e57; }
.band-low { background: #e7eedc; color: #4f6636; }

.auth-box {
  max-width: 520px;
  margin: 10vh auto;
  border: 1px solid var(--line);
  border-radius: 20px;
  background: rgba(255,255,255,.78);
  padding: 22px;
}

@media (max-width: 980px) {
  .title-row {
    flex-direction: column;
  }

  .mosaic {
    grid-template-columns: 1fr 1fr;
  }

  .tile.wide {
    grid-column: span 2;
  }

  .risk-table th:nth-child(8),
  .risk-table td:nth-child(8) {
    display: none;
  }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def quality_badge(status: str) -> str:
    normalized = str(status).lower()
    if normalized == "pass":
        return "ok"
    if normalized == "warn":
        return "warn"
    return "fail"


def risk_band_class(band: str) -> str:
    mapping = {
        "Critical": "band-critical",
        "High": "band-high",
        "Medium": "band-medium",
        "Low": "band-low",
    }
    return mapping.get(str(band), "band-low")


def core_stage_label(status: str) -> str:
    term = service_core_term_for_status(str(status))
    return str(term.get("core_label", "Unknown"))


def status_display(status: str) -> str:
    return STATUS_TEXT.get(str(status), str(status))


def reason_display(raw: str) -> str:
    tokens = [token.strip() for token in str(raw or "").split(",") if token.strip()]
    if not tokens:
        return ""
    return ", ".join(REASON_TEXT.get(token, token.replace("_", " ")) for token in tokens)


def driver_display(raw: object) -> str:
    key = _clean_text(raw).strip()
    return DRIVER_TEXT.get(key, key.replace("_", " ").title())


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def _parse_iso_dt(value: str) -> datetime | None:
    raw = str(value or "").strip()
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


def _age_text(value_utc: str) -> str:
    dt = _parse_iso_dt(value_utc)
    if dt is None:
        return ""
    hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    if hours < 1.0:
        return f"{int(hours * 60)}m ago"
    if hours < 24.0:
        return f"{hours:.1f}h ago"
    days = hours / 24.0
    return f"{days:.1f}d ago"


def next_status_for_quick_move(current_status: str, actor_role: str) -> str:
    normalized = str(current_status).strip()
    allowed = allowed_next_statuses(normalized, actor_role=actor_role)
    preferred = FORWARD_STATUS.get(normalized)
    if preferred and preferred in allowed:
        return preferred
    for candidate in allowed:
        if candidate != normalized:
            return candidate
    return normalized


def render_login_gate() -> Dict[str, object]:
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None

    user = st.session_state.auth_user
    if user:
        return user

    st.markdown(
        """
<div class="auth-box">
  <div class="kicker">Secure Access</div>
  <h2 style="margin-top:0">The Logistics Prophet Login</h2>
  <p style="color:#5c554a">Role-based access is enabled. Actions are audit-tracked.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    with st.expander("Demo Credentials"):
        st.code("admin / admin123!\noperator / ops123!\nviewer / view123!")

    if submitted:
        auth = authenticate_user(username.strip(), password, path=SERVICE_DB_PATH)
        if auth:
            st.session_state.auth_user = auth
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()


def render_session_bar(user: Dict[str, object], pipeline_status: Dict[str, object] | None = None, metrics: Dict[str, object] | None = None) -> None:
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1.25, 1, 1])
    with c1:
        st.caption(f"Signed in as `{user.get('display_name')} ({user.get('username')})`")
    with c2:
        st.caption(f"Role: `{user.get('role')}`")
    with c3:
        run_id = _clean_text((pipeline_status or {}).get("run_id", ""))
        run_status = _clean_text((pipeline_status or {}).get("status", ""))
        finished_at = _clean_text((pipeline_status or {}).get("finished_at_utc", ""))
        data_ts = _clean_text((metrics or {}).get("timestamp_utc", ""))
        data_age = _age_text(data_ts)
        run_age = _age_text(finished_at)
        if run_id:
            st.caption(f"Run: `{run_status}` {run_age}")
        elif data_age:
            st.caption(f"Data: {data_age}")
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with c4:
        role = str(user.get("role", "viewer"))
        if role == "admin":
            run_pipeline = st.button("Run Pipeline", use_container_width=True)
            if run_pipeline:
                with st.spinner("Running pipeline..."):
                    result = subprocess.run(
                        [sys.executable, "scripts/run_pipeline.py"],
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                    )
                if result.returncode == 0:
                    st.success("Pipeline finished.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Pipeline failed.")
                    st.code((result.stdout or "").strip()[-1800:])
                    st.code((result.stderr or "").strip()[-1800:])
        else:
            st.caption("")
    with c5:
        if st.button("Logout", use_container_width=True):
            st.session_state.auth_user = None
            st.rerun()


def build_kpi_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=frame["ship_date"],
            y=frame["on_time_rate"] * 100,
            mode="lines",
            name="On-Time %",
            line=dict(color="#0f625b", width=2.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["ship_date"],
            y=frame["avg_delay_hours"],
            mode="lines",
            name="Avg Delay Hrs",
            yaxis="y2",
            line=dict(color="#ad5e4a", width=2.2, dash="dot"),
        )
    )

    fig.update_layout(
        margin=dict(l=8, r=8, t=4, b=8),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,.08)", title="On-Time %"),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Delay Hrs"),
    )
    return fig


def build_shap_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    top = frame.head(8).iloc[::-1]
    fig.add_trace(
        go.Bar(
            x=top["mean_abs_contribution"],
            y=top["feature"],
            orientation="h",
            marker_color="#0f625b",
            opacity=0.88,
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=4, b=8),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,.08)", title="Mean |Contribution|"),
        yaxis=dict(showgrid=False, title=""),
    )
    return fig


def render_hero(
    metrics: Dict[str, object],
    training: Dict[str, object],
    quality: Dict[str, object],
    queue: pd.DataFrame,
    service_summary: Dict[str, object],
) -> None:
    latest = metrics.get("kpi_latest", {})
    model_metrics = metrics.get("model_test_metrics", {})

    on_time = float(latest.get("on_time_rate", 0.0)) * 100
    breaches = int(float(latest.get("sla_breach_count", 0)))
    auc = float(model_metrics.get("auc", 0.0))
    max_score = float(queue["risk_score"].max()) if not queue.empty else 0.0

    st.markdown(
        f"""
<section class="hero">
  <div class="hero-inner">
    <div class="kicker">Live Ops Board</div>
    <div class="title-row">
      <h1>The Logistics Prophet</h1>
      <span class="stamp">{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
    <p class="hero-sub">
      Predict delay risk, see why, and take action fast.
    </p>
    <div class="mosaic">
      <article class="tile wide"><div class="tile-label">On-Time %</div><div class="tile-value">{on_time:.1f}%</div><div class="tile-meta">target 88.0%</div></article>
      <article class="tile"><div class="tile-label">Late Deliveries</div><div class="tile-value">{breaches}</div><div class="tile-meta">latest run</div></article>
      <article class="tile"><div class="tile-label">Model AUC</div><div class="tile-value">{auc:.3f}</div><div class="tile-meta">model {training.get('selected_model','n/a')}</div></article>
      <article class="tile"><div class="tile-label">Critical Waiting</div><div class="tile-value">{service_summary.get('critical_open',0)}</div><div class="tile-meta">start/check</div></article>
      <article class="tile"><div class="tile-label">Work Items</div><div class="tile-value">{service_summary.get('unresolved',0)}</div><div class="tile-meta">start/check/fix</div></article>
      <article class="tile"><div class="tile-label">Top Risk</div><div class="tile-value">{max_score:.3f}</div><div class="tile-meta">highest score</div></article>
      <article class="tile wide"><div class="tile-label">Data Quality</div><div class="tile-value">{quality.get('status','unknown')}</div><div class="tile-meta">fails {quality.get('fail_count',0)} | warns {quality.get('warn_count',0)}</div></article>
    </div>
  </div>
</section>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_panel(kpi_series: pd.DataFrame, shap_global: pd.DataFrame, quality: Dict[str, object]) -> None:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown('<section class="panel"><h3>Trend Over Time</h3>', unsafe_allow_html=True)
        st.plotly_chart(build_kpi_chart(kpi_series), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</section>", unsafe_allow_html=True)
    with c2:
        badge = quality_badge(str(quality.get("status", "unknown")))
        st.markdown('<section class="panel"><h3>Top Model Reasons</h3>', unsafe_allow_html=True)
        st.plotly_chart(build_shap_chart(shap_global), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"<span class='badge {badge}'>quality {quality.get('status','unknown')}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</section>", unsafe_allow_html=True)


def render_semantic_panel(sparql: Dict[str, object]) -> None:
    queries = sparql.get("queries", []) if isinstance(sparql, dict) else []
    st.markdown('<section class="panel"><h3>Data Graph View</h3>', unsafe_allow_html=True)
    if not queries:
        st.write("No graph query results yet. Run the pipeline first.")
    else:
        for query in queries[:4]:
            st.markdown(f"<span class='query-chip'>{query.get('id')}</span>", unsafe_allow_html=True)
            df = pd.DataFrame(query.get("rows", []))
            st.dataframe(df.head(8), use_container_width=True, height=220)
    st.markdown("</section>", unsafe_allow_html=True)


def render_service_core_board(core_snapshot: Dict[str, object]) -> None:
    st.markdown('<section class="panel"><h3>Core Board</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.caption("Step Load")
        stage_df = pd.DataFrame(core_snapshot.get("stage_backlog", []))
        if not stage_df.empty:
            show = stage_df[["stage", "label", "count", "share_pct"]].rename(
                columns={"stage": "Step", "label": "Step Name", "count": "Count", "share_pct": "Share %"}
            )
            st.dataframe(show, use_container_width=True, height=220)
        else:
            st.caption("No step data.")

        driver_df = pd.DataFrame(core_snapshot.get("driver_hotspots", []))
        if not driver_df.empty:
            st.caption("Top Causes")
            if "driver" in driver_df.columns:
                driver_df["driver"] = driver_df["driver"].map(lambda x: driver_display(x))
            st.dataframe(driver_df.rename(columns={"driver": "Cause", "count": "Count"}), use_container_width=True, height=180)
    with c2:
        st.caption("Urgent Now")
        escalation_df = pd.DataFrame(core_snapshot.get("escalation_candidates", []))
        if not escalation_df.empty:
            if "status" in escalation_df.columns:
                escalation_df["status"] = escalation_df["status"].map(lambda x: status_display(str(x)))
            if "reasons" in escalation_df.columns:
                escalation_df["reasons"] = escalation_df["reasons"].map(lambda x: reason_display(str(x)))
            show_cols = [
                "shipment_id",
                "risk_band",
                "risk_score",
                "status",
                "owner",
                "stale_hours",
                "reasons",
            ]
            existing = [col for col in show_cols if col in escalation_df.columns]
            shown = escalation_df[existing].rename(
                columns={
                    "shipment_id": "Shipment",
                    "risk_band": "Risk",
                    "risk_score": "Score",
                    "status": "Step",
                    "owner": "Owner",
                    "stale_hours": "No Update (h)",
                    "reasons": "Why",
                }
            )
            st.dataframe(shown, use_container_width=True, height=420)
        else:
            st.caption("No urgent items right now.")
    st.markdown("</section>", unsafe_allow_html=True)


def render_service_core_worklist(worklist: Dict[str, object], user: Dict[str, object], queue: pd.DataFrame) -> None:
    st.markdown('<section class="panel"><h3>Next Actions</h3>', unsafe_allow_html=True)
    st.caption("Top items for each step. Higher score = do first.")
    stages = worklist.get("stages", []) if isinstance(worklist, dict) else []
    stage_map = {str(item.get("stage")): item for item in stages if isinstance(item, dict)}
    actor = str(user.get("username", ""))
    role = str(user.get("role", "viewer"))
    can_update = has_permission(role, "queue_update")
    if not can_update:
        st.caption("Quick actions are read-only for your role.")

    queue_map: Dict[str, Dict[str, object]] = {}
    if not queue.empty:
        for _, row in queue.iterrows():
            shipment_id = _clean_text(row.get("shipment_id", ""))
            if shipment_id:
                queue_map[shipment_id] = row.to_dict()

    cols = st.columns(3)
    for idx, stage in enumerate(["Start", "Check", "Fix"]):
        with cols[idx]:
            entry = stage_map.get(stage, {})
            item_lookup = {
                _clean_text(item.get("shipment_id", "")): item
                for item in entry.get("items", [])
                if isinstance(item, dict) and _clean_text(item.get("shipment_id", ""))
            }
            st.subheader(f"{stage} ({int(entry.get('count', 0))})")
            items = pd.DataFrame(entry.get("items", []))
            if items.empty:
                st.caption("No items.")
                continue
            show_cols = [
                "shipment_id",
                "risk_band",
                "owner",
                "age_hours",
                "urgency_score",
                "next_step",
                "why",
            ]
            existing = [col for col in show_cols if col in items.columns]
            st.dataframe(
                items[existing].rename(
                    columns={
                        "shipment_id": "Shipment",
                        "risk_band": "Risk",
                        "owner": "Owner",
                        "age_hours": "Open (h)",
                        "urgency_score": "Urgency",
                        "next_step": "Next",
                        "why": "Why",
                    }
                ),
                use_container_width=True,
                height=270,
            )
            shipment_choices = [shipment_id for shipment_id in item_lookup.keys() if shipment_id]
            if not shipment_choices:
                continue

            selected = st.selectbox("Pick Item", options=shipment_choices, key=f"core_pick_{stage}")
            source = item_lookup.get(selected, {})
            current = queue_map.get(selected, {})

            current_status = _clean_text(current.get("status", "")) or _clean_text(source.get("status", "New")) or "New"
            current_owner = _clean_text(current.get("owner", "")) or _clean_text(source.get("owner", ""))
            current_note = _clean_text(current.get("note", ""))
            current_eta = _clean_text(current.get("eta_action_at", "")) or _clean_text(source.get("eta_action_at", ""))
            next_status = next_status_for_quick_move(current_status=current_status, actor_role=role)

            st.caption(f"Now: {status_display(current_status)} | Next: {status_display(next_status)} | Owner: {current_owner or 'None'}")
            b1, b2, b3 = st.columns(3)
            with b1:
                assign = st.button("Assign Me", key=f"core_assign_{stage}", use_container_width=True, disabled=not can_update)
            with b2:
                move_disabled = (next_status == current_status) or (not can_update)
                move = st.button(
                    f"Move Next ({status_display(next_status)})",
                    key=f"core_move_{stage}",
                    use_container_width=True,
                    disabled=move_disabled,
                )
            with b3:
                eta_plus = st.button("ETA +2h", key=f"core_eta_{stage}", use_container_width=True, disabled=not can_update)

            if assign:
                try:
                    update_queue_action(
                        shipment_id=selected,
                        status=current_status,
                        owner=actor,
                        note=current_note,
                        eta_action_at=current_eta,
                        actor=actor,
                        actor_role=role,
                        path=SERVICE_DB_PATH,
                    )
                    st.success(f"Assigned {selected} to {actor}.")
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))
            if move:
                try:
                    update_queue_action(
                        shipment_id=selected,
                        status=next_status,
                        owner=current_owner,
                        note=current_note,
                        eta_action_at=current_eta,
                        actor=actor,
                        actor_role=role,
                        path=SERVICE_DB_PATH,
                    )
                    st.success(f"Moved {selected} to {status_display(next_status)}.")
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))
            if eta_plus:
                try:
                    eta_new = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
                    update_queue_action(
                        shipment_id=selected,
                        status=current_status,
                        owner=current_owner,
                        note=current_note,
                        eta_action_at=eta_new,
                        actor=actor,
                        actor_role=role,
                        path=SERVICE_DB_PATH,
                    )
                    st.success(f"ETA +2h set for {selected}.")
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))
    st.markdown("</section>", unsafe_allow_html=True)


def render_workflow_sla_panel(sla_snapshot: Dict[str, object]) -> None:
    st.markdown('<section class="panel"><h3>Time Watch</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Work Items", int(sla_snapshot.get("unresolved_total", 0)))
    with c2:
        st.metric("Late Work", int(sla_snapshot.get("breached_total", 0)))
    with c3:
        st.metric("Late Work %", f"{float(sla_snapshot.get('breach_rate_pct', 0.0)):.1f}%")

    left, right = st.columns([1, 1.25])
    with left:
        buckets = pd.DataFrame(sla_snapshot.get("age_buckets", []))
        if not buckets.empty:
            st.caption("Age Buckets")
            st.dataframe(buckets.rename(columns={"bucket": "Time", "count": "Count"}), use_container_width=True, height=180)
        stages = pd.DataFrame(sla_snapshot.get("stage_sla", []))
        if not stages.empty:
            if "status" in stages.columns:
                stages["status"] = stages["status"].map(lambda x: status_display(str(x)))
            show = stages[["status", "threshold_hours", "in_stage", "breached"]]
            st.caption("Step Limits")
            st.dataframe(
                show.rename(
                    columns={
                        "status": "Step",
                        "threshold_hours": "Limit (h)",
                        "in_stage": "In Step",
                        "breached": "Late",
                    }
                ),
                use_container_width=True,
                height=220,
            )
    with right:
        candidates = pd.DataFrame(sla_snapshot.get("breached_candidates", []))
        if not candidates.empty:
            if "status" in candidates.columns:
                candidates["status"] = candidates["status"].map(lambda x: status_display(str(x)))
            show_cols = [
                "shipment_id",
                "status",
                "risk_band",
                "risk_score",
                "owner",
                "age_hours",
                "threshold_hours",
                "over_by_hours",
            ]
            existing = [col for col in show_cols if col in candidates.columns]
            st.caption("Late Work List")
            st.dataframe(
                candidates[existing].rename(
                    columns={
                        "shipment_id": "Shipment",
                        "status": "Step",
                        "risk_band": "Risk",
                        "risk_score": "Score",
                        "owner": "Owner",
                        "age_hours": "Open Time (h)",
                        "threshold_hours": "Limit (h)",
                        "over_by_hours": "Late By (h)",
                    }
                ),
                use_container_width=True,
                height=320,
            )
        else:
            st.caption("No late items.")
    st.markdown("</section>", unsafe_allow_html=True)


def render_queue_panel(queue: pd.DataFrame) -> pd.DataFrame:
    if queue.empty:
        st.warning("Work queue is empty. Run `python3 scripts/run_pipeline.py` first.")
        return queue

    bands = ["All", "Critical", "High", "Medium", "Low"]
    core_terms = list_service_core_terms()
    statuses = ["All"] + [str(row.get("status")) for row in core_terms]
    owners = ["All"] + sorted([owner for owner in queue["owner"].fillna("").unique().tolist() if owner])

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        band = st.selectbox("Risk Level", bands, index=0)
    with c2:
        status = st.selectbox("Step", statuses, index=0, format_func=lambda s: "All" if s == "All" else status_display(str(s)))
    with c3:
        owner = st.selectbox("Owner", owners, index=0)
    with c4:
        top_n = st.slider("Rows", min_value=10, max_value=min(150, len(queue)), value=min(40, len(queue)), step=10)

    filtered = queue.copy()
    if band != "All":
        filtered = filtered[filtered["risk_band"] == band]
    if status != "All":
        filtered = filtered[filtered["status"] == status]
    if owner != "All":
        filtered = filtered[filtered["owner"] == owner]

    rows_html = []
    for _, row in filtered.head(top_n).iterrows():
        rows_html.append(
            "<tr>"
            f"<td>{row.get('shipment_id','')}</td>"
            f"<td>{row.get('order_id','')}</td>"
            f"<td>{float(row.get('risk_score',0.0)):.4f}</td>"
            f"<td><span class='band {risk_band_class(str(row.get('risk_band','Low')))}'>{row.get('risk_band','')}</span></td>"
            f"<td>{status_display(str(row.get('status','')))}</td>"
            f"<td>{row.get('owner','')}</td>"
            f"<td>{driver_display(row.get('key_driver',''))}</td>"
            f"<td>{row.get('recommended_action','')}</td>"
            "</tr>"
        )

    st.markdown(
        f"""
<section class="panel">
  <h3>Queue</h3>
  <table class="risk-table">
    <thead><tr><th>Shipment</th><th>Order</th><th>Score</th><th>Risk</th><th>Step</th><th>Owner</th><th>Cause</th><th>Action</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</section>
        """,
        unsafe_allow_html=True,
    )
    return filtered


def render_execution_console(queue: pd.DataFrame, user: Dict[str, object]) -> None:
    st.markdown('<section class="panel"><h3>Update</h3>', unsafe_allow_html=True)
    can_update = has_permission(str(user.get("role", "viewer")), "queue_update")

    if queue.empty:
        st.write("No work rows available.")
        st.markdown("</section>", unsafe_allow_html=True)
        return

    if not can_update:
        st.info("Your role is read-only for updates.")

    selected = st.selectbox("Shipment", options=queue["shipment_id"].tolist())
    selected_row = queue[queue["shipment_id"] == selected].iloc[0]

    role = str(user.get("role", "viewer"))
    status_options = [str(row.get("status")) for row in list_service_core_terms()]
    current_status = str(selected_row.get("status", "New"))
    allowed_options = allowed_next_statuses(current_status, actor_role=role)
    status_index = allowed_options.index(current_status) if current_status in allowed_options else 0
    bulk_status_index = status_options.index(current_status) if current_status in status_options else 0

    with st.form("action_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            status = st.selectbox(
                "Step",
                allowed_options,
                index=status_index,
                format_func=lambda s: status_display(str(s)),
                disabled=not can_update,
            )
            owner = st.text_input("Owner", value=str(selected_row.get("owner", "")), disabled=not can_update)
        with c2:
            eta = st.text_input(
                "ETA (ISO-8601)",
                value=str(selected_row.get("eta_action_at", "")),
                disabled=not can_update,
                help="Examples: 2026-02-13T12:30:00+00:00 or empty.",
            )
            note = st.text_area("Note", value=str(selected_row.get("note", "")), height=110, disabled=not can_update)

        submit = st.form_submit_button("Save Action", disabled=not can_update)

    if submit:
        try:
            update_queue_action(
                shipment_id=selected,
                status=status,
                owner=owner,
                note=note,
                eta_action_at=eta,
                actor=str(user.get("username")),
                actor_role=str(user.get("role")),
                path=SERVICE_DB_PATH,
            )
            st.success(f"Updated {selected}")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    if can_update:
        with st.form("bulk_action_form", clear_on_submit=False):
            st.caption("Bulk Update")
            bulk_shipments = st.multiselect("Shipments", options=queue["shipment_id"].tolist())
            c1, c2 = st.columns(2)
            with c1:
                bulk_status = st.selectbox(
                    "Bulk Step",
                    status_options,
                    index=bulk_status_index,
                    format_func=lambda s: status_display(str(s)),
                )
                bulk_owner = st.text_input("Bulk Owner", value=str(selected_row.get("owner", "")))
            with c2:
                bulk_eta = st.text_input(
                    "Bulk ETA (ISO-8601)",
                    value=str(selected_row.get("eta_action_at", "")),
                    help="Examples: 2026-02-13T12:30:00+00:00 or empty.",
                )
                bulk_note = st.text_area("Bulk Note", value="", height=90)
            bulk_submit = st.form_submit_button("Apply Bulk Change")

        if bulk_submit:
            if not bulk_shipments:
                st.error("Select at least one shipment for bulk update.")
            else:
                try:
                    result = bulk_update_queue_actions(
                        shipment_ids=bulk_shipments,
                        status=bulk_status,
                        owner=bulk_owner,
                        note=bulk_note,
                        eta_action_at=bulk_eta,
                        actor=str(user.get("username")),
                        actor_role=str(user.get("role")),
                        path=SERVICE_DB_PATH,
                    )
                    st.success(f"Bulk updated {result.get('updated', 0)} shipments.")
                    missing = result.get("missing", [])
                    invalid_transitions = result.get("invalid_transitions", [])
                    if missing:
                        st.warning(f"Missing shipments skipped: {', '.join(missing[:5])}")
                    if invalid_transitions:
                        st.warning(f"Skipped {len(invalid_transitions)} rows due to status flow rules.")
                        invalid_df = pd.DataFrame(invalid_transitions).head(12)
                        if "current_status" in invalid_df.columns:
                            invalid_df["current_status"] = invalid_df["current_status"].map(lambda x: status_display(str(x)))
                        if "requested_status" in invalid_df.columns:
                            invalid_df["requested_status"] = invalid_df["requested_status"].map(lambda x: status_display(str(x)))
                        st.dataframe(invalid_df, use_container_width=True, height=180)
                    if not missing and not invalid_transitions:
                        st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    activity = fetch_activity(selected, path=SERVICE_DB_PATH, limit=12)
    if activity:
        activity_df = pd.DataFrame(
            [
                {
                    "created_at": row.get("created_at"),
                    "actor": row.get("actor"),
                    "role": row.get("actor_role"),
                    "action": row.get("action"),
                    "reason": row.get("reason"),
                    "payload": json.dumps(row.get("payload", {}), ensure_ascii=True),
                }
                for row in activity
            ]
        )
        st.dataframe(activity_df, use_container_width=True, height=230)
    else:
        st.caption("No activity for this shipment.")

    st.markdown("</section>", unsafe_allow_html=True)


def render_incident_console(user: Dict[str, object]) -> None:
    st.markdown('<section class="panel"><h3>Incident</h3>', unsafe_allow_html=True)
    can_manage = has_permission(str(user.get("role", "viewer")), "incident_manage")
    if not can_manage:
        st.info("Your role is read-only for incidents.")

    with st.form("incident_form", clear_on_submit=True):
        title = st.text_input(
            "Incident Title",
            value="Carrier instability impacting critical queue",
            disabled=not can_manage,
        )
        c1, c2 = st.columns(2)
        with c1:
            severity = st.selectbox("Level", ["SEV-1", "SEV-2", "SEV-3"], index=1, disabled=not can_manage)
        with c2:
            status = st.selectbox("Status", ["Open", "Monitoring", "Closed"], index=0, disabled=not can_manage)
        owner = st.text_input("Incident Owner", value=str(user.get("username")), disabled=not can_manage)
        description = st.text_area("Description", height=90, disabled=not can_manage)
        create = st.form_submit_button("Save Incident", disabled=not can_manage)

    if create and title.strip():
        try:
            incident_id = upsert_incident(
                title=title.strip(),
                severity=severity,
                description=description.strip(),
                status=status,
                owner=owner,
                actor=str(user.get("username")),
                actor_role=str(user.get("role")),
                path=SERVICE_DB_PATH,
            )
            st.success(f"Incident saved: {incident_id}")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    incidents = list_incidents(SERVICE_DB_PATH, limit=15)
    if incidents:
        st.dataframe(pd.DataFrame(incidents), use_container_width=True, height=230)
    else:
        st.caption("No incidents recorded.")

    runs = list_pipeline_runs(SERVICE_DB_PATH, limit=10)
    if runs:
        st.caption("Pipeline Run History")
        st.dataframe(pd.DataFrame(runs), use_container_width=True, height=220)

    recommendations = derive_incident_recommendations(path=SERVICE_DB_PATH, max_items=5)
    st.caption("Auto Suggestions")
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True, height=210)

        if can_manage:
            labels = [
                f"{item.get('severity')} | {item.get('title')} ({item.get('rule_id')})"
                for item in recommendations
            ]
            selected_label = st.selectbox(
                "Pick",
                options=labels,
                key="recommendation_select",
            )
            if st.button("Create Incident", use_container_width=True):
                selected_idx = labels.index(selected_label)
                selected = recommendations[selected_idx]
                result = upsert_incident_from_recommendation(
                    recommendation=selected,
                    owner=str(user.get("username")),
                    actor=str(user.get("username")),
                    actor_role=str(user.get("role")),
                    path=SERVICE_DB_PATH,
                )
                if bool(result.get("deduplicated")):
                    st.success(f"Incident updated (deduplicated): {result.get('incident_id')}")
                else:
                    st.success(f"Incident created from recommendation: {result.get('incident_id')}")
                st.rerun()
    else:
        st.caption("No suggestion rules are active now.")

    st.markdown("</section>", unsafe_allow_html=True)


def render_admin_panel(user: Dict[str, object]) -> None:
    if not has_permission(str(user.get("role", "viewer")), "user_manage"):
        return

    st.markdown('<section class="panel"><h3>Admin: Users & Audit</h3>', unsafe_allow_html=True)

    with st.form("user_admin_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            username = st.text_input("Username", value="new-user")
            display_name = st.text_input("Display Name", value="New User")
        with c2:
            role = st.selectbox("Role", ["viewer", "operator", "admin"], index=0)
            is_active = st.selectbox("Active", ["yes", "no"], index=0)
        with c3:
            password = st.text_input("Password", type="password")
        save_user = st.form_submit_button("Create / Update User")

    if save_user:
        if not password.strip():
            st.error("Password is required.")
        else:
            create_or_update_user(
                username=username.strip(),
                display_name=display_name.strip(),
                role=role,
                password=password,
                is_active=(is_active == "yes"),
                actor=str(user.get("username")),
                actor_role=str(user.get("role")),
                path=SERVICE_DB_PATH,
            )
            st.success(f"User upserted: {username}")
            st.rerun()

    users_df = pd.DataFrame(list_users(SERVICE_DB_PATH, include_inactive=True))
    st.caption("Users")
    st.dataframe(users_df, use_container_width=True, height=180)

    audit_result = verify_audit_chain(path=SERVICE_DB_PATH, limit=5000)
    st.caption(f"Audit Chain Valid: {audit_result.get('valid')} (checked={audit_result.get('checked')})")

    recent = pd.DataFrame(list_recent_activity(SERVICE_DB_PATH, limit=20))
    if not recent.empty:
        show = recent[["created_at", "actor", "actor_role", "action", "entity_type", "entity_id", "reason"]]
        st.dataframe(show, use_container_width=True, height=220)

    st.markdown("</section>", unsafe_allow_html=True)


def main() -> None:
    init_service_store(SERVICE_DB_PATH)
    inject_css()
    user = render_login_gate()

    metrics = load_json(METRICS_PATH)
    quality = load_json(QUALITY_PATH)
    sparql = load_json(SPARQL_PATH)
    shap_global = load_shap_global(SHAP_GLOBAL_PATH)
    training = load_json(TRAINING_PATH)
    pipeline_status = load_json(PIPELINE_STATUS_PATH)
    kpi_series = load_kpi_series()
    queue = load_service_queue()
    service_summary = fetch_queue_summary(SERVICE_DB_PATH)
    ops_health = fetch_ops_health(SERVICE_DB_PATH)
    core_snapshot = fetch_service_core_snapshot(SERVICE_DB_PATH)
    core_worklist = fetch_service_core_worklist(SERVICE_DB_PATH, per_stage_limit=6)
    workflow_sla = fetch_workflow_sla_snapshot(SERVICE_DB_PATH)

    if not metrics:
        st.error("Pipeline output not found. Run the pipeline first.")
        if str(user.get("role", "viewer")) == "admin":
            if st.button("Run Pipeline Now", use_container_width=True):
                with st.spinner("Running pipeline..."):
                    result = subprocess.run(
                        [sys.executable, "scripts/run_pipeline.py"],
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                    )
                if result.returncode == 0:
                    st.success("Pipeline finished.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Pipeline failed.")
                    st.code((result.stdout or "").strip()[-1800:])
                    st.code((result.stderr or "").strip()[-1800:])
        return

    render_session_bar(user, pipeline_status=pipeline_status, metrics=metrics)
    render_hero(metrics, training, quality, queue, service_summary)
    with st.expander("Guide"):
        st.markdown(
            """
**Tiles**
- **On-Time %**: percent delivered on time (latest day)
- **Late Deliveries**: count delivered late (latest day)
- **Model AUC**: how well the model separates late vs not-late (test score)
- **Critical Waiting**: critical items still in **Start/Check**
- **Work Items**: items not finished (**Start/Check/Fix**)
- **Top Risk**: highest risk score in the queue (0 to 1)
- **Data Quality**: pass/warn/fail gate for the pipeline

**How To Use**
1. Go to **Next Actions** and start with the highest urgency.
2. Use quick buttons: **Assign Me**, **Move Next**, **ETA +2h**.
3. Use **Update** to add note/ETA or do bulk changes.
4. Use **Incident** when suggestions trigger or risk grows.
            """.strip()
        )
    render_analysis_panel(kpi_series, shap_global, quality)
    render_service_core_board(core_snapshot)
    render_service_core_worklist(core_worklist, user=user, queue=queue)
    render_workflow_sla_panel(workflow_sla)

    left, right = st.columns([1.35, 1])
    with left:
        render_semantic_panel(sparql)
    with right:
        render_incident_console(user)

    filtered = render_queue_panel(queue)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        render_execution_console(filtered if not filtered.empty else queue, user)
    with c2:
        st.markdown('<section class="panel"><h3>Health</h3>', unsafe_allow_html=True)
        status_df = pd.DataFrame(service_summary.get("status_breakdown", []))
        risk_df = pd.DataFrame(service_summary.get("risk_breakdown", []))
        if not status_df.empty and "status" in status_df.columns:
            status_df["status"] = status_df["status"].map(lambda s: status_display(str(s)))
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Past ETA", int(ops_health.get("overdue_eta", 0)))
            st.metric("Stale 24h+", int(ops_health.get("stale_24h", 0)))
        with m2:
            st.metric("Critical No Owner", int(ops_health.get("critical_unassigned", 0)))
            st.metric("Avg Age (h)", float(ops_health.get("avg_unresolved_age_hours", 0.0)))
        if not status_df.empty:
            st.caption("Step Mix")
            st.dataframe(status_df, use_container_width=True, height=180)
        if not risk_df.empty:
            st.caption("Risk Mix")
            st.dataframe(risk_df, use_container_width=True, height=180)
        owner_df = pd.DataFrame(ops_health.get("owner_backlog", []))
        if not owner_df.empty:
            st.caption("Owner Load")
            st.dataframe(owner_df, use_container_width=True, height=180)
        st.markdown("</section>", unsafe_allow_html=True)

    render_admin_panel(user)


if __name__ == "__main__":
    main()
