from __future__ import annotations

import json
import hashlib
import io
import os
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import SERVICE_DB_PATH
from control_tower.data_access import fetch_shipment_feature_row
from control_tower.semantic_queries import load_instance_graph, query_shipment_evidence
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
    update_queue_action,
    upsert_incident,
    upsert_incident_from_recommendation,
    verify_audit_chain,
)

METRICS_PATH = ROOT / "data" / "output" / "monitoring_metrics.json"
QUALITY_PATH = ROOT / "data" / "output" / "data_quality_report.json"
SPARQL_PATH = ROOT / "data" / "output" / "sparql_results.json"
SHAP_GLOBAL_PATH = ROOT / "data" / "output" / "shap_global_importance.csv"
MODEL_COMPARISON_PATH = ROOT / "data" / "output" / "model_comparison.json"
TRAINING_PATH = ROOT / "data" / "output" / "training_summary.json"
PIPELINE_STATUS_PATH = ROOT / "data" / "output" / "pipeline_status.json"
SQLITE_PATH = ROOT / "data" / "processed" / "control_tower.db"
SHAP_LOCAL_PATH = ROOT / "data" / "output" / "shap_local_explanations.csv"
RDF_INSTANCE_PATH = ROOT / "data" / "semantic" / "instance_graph.ttl"
ASSET_DIR = ROOT / "app" / "assets"
AURORA_BANNER_PATH = ASSET_DIR / "aurora_banner.gif"

SELECTED_SHIPMENT_KEY = "selected_shipment_id"

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


def _query_param(name: str) -> Optional[str]:
    # Support both older and newer Streamlit query param APIs.
    try:
        params = st.query_params  # type: ignore[attr-defined]
        if name in params:
            value = params.get(name)
            if isinstance(value, list):
                return value[0] if value else None
            return str(value) if value is not None else None
    except Exception:
        pass

    try:
        params = st.experimental_get_query_params()  # type: ignore[attr-defined]
        if name in params and params[name]:
            return str(params[name][0])
    except Exception:
        pass
    return None


def cinematic_ui_enabled() -> bool:
    """
    Legacy flag kept for backward compatibility with older demo links.
    Enable via:
      1) `?cinematic=1` query param, or
      2) `LP_CINEMATIC_UI=1` environment variable.
    """
    qp = _query_param("cinematic")
    if qp is not None:
        return qp.strip().lower() in {"1", "true", "yes", "on"}
    env = os.getenv("LP_CINEMATIC_UI", "").strip().lower()
    if not env:
        return False
    return env in {"1", "true", "yes", "on"}


def ui_mode() -> str:
    """
    UI modes:
      - plain: minimal (troubleshooting; hides non-essential visuals)
      - safe: Streamlit theme only (default)
      - cinematic: legacy alias (currently same as safe)

    Controls:
      - query: `?ui=plain|safe|cinematic`
      - query: `?cinematic=1` (legacy alias for `ui=cinematic`)
      - env: `LP_UI_MODE=plain|safe|cinematic`
      - env: `LP_CINEMATIC_UI=1` (legacy alias)
    """
    qp_ui = _query_param("ui")
    if qp_ui:
        candidate = qp_ui.strip().lower()
        if candidate in {"plain", "safe", "cinematic"}:
            return candidate

    if cinematic_ui_enabled():
        return "cinematic"

    env = os.getenv("LP_UI_MODE", "").strip().lower()
    if env in {"plain", "safe", "cinematic"}:
        return env

    return "safe"


def apply_cinematic_css_if_enabled(mode: str) -> None:
    if mode != "cinematic":
        return
    css_path = ASSET_DIR / "cinematic.css"
    if not css_path.exists():
        return
    try:
        css = css_path.read_text(encoding="utf-8")
    except OSError:
        return
    # Cinematic mode intentionally uses a small amount of CSS for atmosphere.
    # Safe/plain modes avoid custom CSS to maximize browser compatibility.
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_nav(mode: str) -> str:
    pages = [
        "Control Tower",
        "Worklist",
        "Queue + Update",
        "Incidents",
        "Insights",
        "Governance",
    ]
    default = (_query_param("page") or "").strip().lower().replace("_", "-")
    default_map = {
        "control": "Control Tower",
        "control-tower": "Control Tower",
        "tower": "Control Tower",
        "worklist": "Worklist",
        "queue": "Queue + Update",
        "queue-update": "Queue + Update",
        "incidents": "Incidents",
        "insights": "Insights",
        "governance": "Governance",
    }
    default_label = default_map.get(default, "Control Tower")
    try:
        default_index = pages.index(default_label)
    except ValueError:
        default_index = 0

    st.caption("Navigate")
    page = st.radio(
        "Navigate",
        options=pages,
        index=default_index,
        horizontal=True,
        key="lp_nav_main",
        label_visibility="collapsed",
    )
    if mode == "plain":
        st.caption("Plain mode is enabled (`?ui=plain`). Visual polish is reduced for troubleshooting.")
    return str(page)


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
def load_shap_local(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "shipment_id",
                "risk_score",
                "feature_1",
                "contribution_1",
                "feature_2",
                "contribution_2",
                "feature_3",
                "contribution_3",
            ]
        )
    frame = pd.read_csv(path)
    for col in ["risk_score", "contribution_1", "contribution_2", "contribution_3"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    if "shipment_id" in frame.columns:
        frame["shipment_id"] = frame["shipment_id"].astype(str)
    return frame


@st.cache_resource(show_spinner=False)
def load_rdf_graph(graph_path: str):
    path = Path(graph_path)
    if not path.exists():
        return None
    return load_instance_graph(path)


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


@contextmanager
def panel(title: str, *, caption: str | None = None):
    """Bordered section wrapper built from Streamlit primitives (no custom HTML required)."""
    with st.container(border=True):
        st.subheader(title)
        if caption:
            st.caption(caption)
        yield

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


def set_selected_shipment(shipment_id: str) -> None:
    st.session_state[SELECTED_SHIPMENT_KEY] = str(shipment_id or "").strip()


def get_selected_shipment(queue: pd.DataFrame | None = None) -> str:
    current = str(st.session_state.get(SELECTED_SHIPMENT_KEY, "") or "").strip()
    if current:
        return current
    if queue is not None and not queue.empty and "shipment_id" in queue.columns:
        first = str(queue.iloc[0].get("shipment_id", "")).strip()
        if first:
            set_selected_shipment(first)
            return first
    return ""


def render_login_gate() -> Dict[str, object]:
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None

    user = st.session_state.auth_user
    if user:
        return user

    # Avoid relying on custom HTML/CSS for the login gate.
    # If a user's browser blocks custom styles or scripts, standard Streamlit widgets
    # should still render clearly.
    st.title("The Logistics Prophet")
    st.caption("Secure access. Role-based permissions are enabled and actions are audit-tracked.")
    st.caption("If the page looks blank, try `?ui=plain` or run `make demo-local-debug`.")

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
    *,
    show_banner: bool = False,
) -> None:
    latest = metrics.get("kpi_latest", {})
    model_metrics = metrics.get("model_test_metrics", {})

    on_time = float(latest.get("on_time_rate", 0.0)) * 100
    breaches = int(float(latest.get("sla_breach_count", 0)))
    auc = float(model_metrics.get("auc", 0.0))
    max_score = float(queue["risk_score"].max()) if not queue.empty else 0.0

    # Render hero media via Streamlit's media endpoint (not base64 in deltas),
    # which is significantly more reliable across browsers.
    if show_banner and AURORA_BANNER_PATH.exists():
        st.image(str(AURORA_BANNER_PATH), use_container_width=True)

    with st.container(border=True):
        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.caption("Live Ops Board")
            st.title("The Logistics Prophet")
            st.caption("Predict delay risk, see why, and take action fast.")
        with header_right:
            st.caption("Snapshot")
            st.markdown(f"**{datetime.now().strftime('%Y-%m-%d %H:%M')}**")
            selected_model = str(training.get("selected_model", "") or "").strip()
            if selected_model:
                st.caption(f"Model: `{selected_model}`")

        st.divider()

        target_on_time = 88.0
        delta_on_time = on_time - target_on_time

        r1 = st.columns(4)
        with r1[0]:
            st.metric("On-Time %", f"{on_time:.1f}%", delta=f"{delta_on_time:+.1f}% vs target")
        with r1[1]:
            st.metric("Late Deliveries", int(breaches))
        with r1[2]:
            st.metric("Model AUC", f"{auc:.3f}")
        with r1[3]:
            st.metric("Critical Waiting", int(service_summary.get("critical_open", 0)))

        r2 = st.columns(4)
        with r2[0]:
            st.metric("Work Items", int(service_summary.get("unresolved", 0)))
        with r2[1]:
            st.metric("Top Risk", f"{max_score:.3f}")
        with r2[2]:
            st.metric("Queue Size", int(len(queue)) if queue is not None else 0)
        with r2[3]:
            status = str(quality.get("status", "unknown") or "unknown").strip().lower()
            st.metric(
                "Data Quality",
                status,
                delta=f"fails {int(quality.get('fail_count', 0))} | warns {int(quality.get('warn_count', 0))}",
            )
            if status == "pass":
                st.success("Gate: PASS")
            elif status == "warn":
                st.warning("Gate: WARN")
            elif status == "fail":
                st.error("Gate: FAIL")


def render_analysis_panel(kpi_series: pd.DataFrame, shap_global: pd.DataFrame, quality: Dict[str, object]) -> None:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        with panel("Trend Over Time"):
            st.plotly_chart(build_kpi_chart(kpi_series), use_container_width=True, config={"displayModeBar": False})
    with c2:
        with panel("Top Model Reasons"):
            st.plotly_chart(build_shap_chart(shap_global), use_container_width=True, config={"displayModeBar": False})

            status = str(quality.get("status", "unknown") or "unknown").strip().lower()
            if status == "pass":
                st.success("Quality gate: PASS")
            elif status == "warn":
                st.warning("Quality gate: WARN")
            elif status == "fail":
                st.error("Quality gate: FAIL")
            else:
                st.info(f"Quality gate: {status}")


def render_semantic_panel(sparql: Dict[str, object]) -> None:
    queries = sparql.get("queries", []) if isinstance(sparql, dict) else []
    with panel("Data Graph View"):
        if not queries:
            st.write("No graph query results yet. Run the pipeline first.")
        else:
            for query in queries[:4]:
                st.caption(str(query.get("id", "") or "").strip())
                df = pd.DataFrame(query.get("rows", []))
                st.dataframe(df.head(8), use_container_width=True, height=220)


def render_service_core_board(core_snapshot: Dict[str, object]) -> None:
    with panel("Core Board"):
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
                st.dataframe(
                    driver_df.rename(columns={"driver": "Cause", "count": "Count"}),
                    use_container_width=True,
                    height=180,
                )
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


def render_service_core_worklist(worklist: Dict[str, object], user: Dict[str, object], queue: pd.DataFrame) -> None:
    with panel("Next Actions", caption="Top items for each step. Higher score = do first."):
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
                set_selected_shipment(selected)
                source = item_lookup.get(selected, {})
                current = queue_map.get(selected, {})

                current_status = _clean_text(current.get("status", "")) or _clean_text(source.get("status", "New")) or "New"
                current_owner = _clean_text(current.get("owner", "")) or _clean_text(source.get("owner", ""))
                current_note = _clean_text(current.get("note", ""))
                current_eta = _clean_text(current.get("eta_action_at", "")) or _clean_text(source.get("eta_action_at", ""))
                next_status = next_status_for_quick_move(current_status=current_status, actor_role=role)

                st.caption(
                    f"Now: {status_display(current_status)} | Next: {status_display(next_status)} | Owner: {current_owner or 'None'}"
                )
                b1, b2, b3 = st.columns(3)
                with b1:
                    assign = st.button(
                        "Assign Me",
                        key=f"core_assign_{stage}",
                        use_container_width=True,
                        disabled=not can_update,
                    )
                with b2:
                    move_disabled = (next_status == current_status) or (not can_update)
                    move = st.button(
                        f"Move Next ({status_display(next_status)})",
                        key=f"core_move_{stage}",
                        use_container_width=True,
                        disabled=move_disabled,
                    )
                with b3:
                    eta_plus = st.button(
                        "ETA +2h",
                        key=f"core_eta_{stage}",
                        use_container_width=True,
                        disabled=not can_update,
                    )

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


def render_workflow_sla_panel(sla_snapshot: Dict[str, object]) -> None:
    with panel("Time Watch"):
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
                st.dataframe(
                    buckets.rename(columns={"bucket": "Time", "count": "Count"}),
                    use_container_width=True,
                    height=180,
                )
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


def render_ops_health_panel(service_summary: Dict[str, object], ops_health: Dict[str, object]) -> None:
    with panel("Ops Health"):
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


def render_quality_details_panel(quality: Dict[str, object]) -> None:
    with panel("Quality Gate Details"):
        checks = quality.get("checks", []) if isinstance(quality, dict) else []
        if not checks:
            st.caption("No quality checks found.")
            return

        df = pd.DataFrame(checks)
        if "status" in df.columns:
            df["status"] = df["status"].astype(str).str.lower()
        st.dataframe(df, use_container_width=True, height=360)


def render_model_selection_panel(training: Dict[str, object], model_comparison: Dict[str, object]) -> None:
    with panel("Model Selection"):
        st.caption(str(training.get("selection_reason", "") or model_comparison.get("selection_reason", "") or "").strip())

        selected = str(training.get("selected_model", "") or model_comparison.get("selected_model", "") or "").strip()
        threshold = training.get("selected_threshold", model_comparison.get("selected_threshold", ""))
        c1, c2, c3 = st.columns([1.1, 1, 1])
        with c1:
            st.metric("Selected Model", selected or "-")
        with c2:
            st.metric("Threshold", _format_number(threshold, 3) if threshold != "" else "-")
        with c3:
            st.metric("AUC (Test)", _format_number((training.get("test_metrics", {}) or {}).get("auc", ""), 4))

        comp = model_comparison or training.get("comparison", {}) or {}
        test_metrics = comp.get("test_metrics", {}) if isinstance(comp, dict) else {}
        if isinstance(test_metrics, dict) and test_metrics:
            df = pd.DataFrame.from_dict(test_metrics, orient="index").reset_index().rename(columns={"index": "model"})
            st.caption("Test Metrics (Baseline vs Challenger)")
            st.dataframe(df, use_container_width=True, height=240)


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

    search = st.text_input(
        "Search",
        value=str(st.session_state.get("queue_search", "") or ""),
        placeholder="shipment / order / owner ...",
        key="queue_search",
    )

    filtered = queue.copy()
    if band != "All":
        filtered = filtered[filtered["risk_band"] == band]
    if status != "All":
        filtered = filtered[filtered["status"] == status]
    if owner != "All":
        filtered = filtered[filtered["owner"] == owner]
    if str(search or "").strip():
        term = str(search).strip().lower()
        filtered = filtered[
            filtered["shipment_id"].astype(str).str.lower().str.contains(term)
            | filtered["order_id"].astype(str).str.lower().str.contains(term)
            | filtered["owner"].astype(str).str.lower().str.contains(term)
        ]

    with panel("Queue"):
        show = filtered.head(top_n).copy()
        if "status" in show.columns:
            show["status"] = show["status"].map(lambda x: status_display(str(x)))
        if "key_driver" in show.columns:
            show["key_driver"] = show["key_driver"].map(lambda x: driver_display(x))
        if "risk_score" in show.columns:
            show["risk_score"] = pd.to_numeric(show["risk_score"], errors="coerce").fillna(0.0)

        display_cols = [
            "shipment_id",
            "order_id",
            "risk_score",
            "risk_band",
            "status",
            "owner",
            "key_driver",
            "recommended_action",
        ]
        existing = [col for col in display_cols if col in show.columns]
        display = show[existing].rename(
            columns={
                "shipment_id": "Shipment",
                "order_id": "Order",
                "risk_score": "Score",
                "risk_band": "Risk",
                "status": "Step",
                "owner": "Owner",
                "key_driver": "Cause",
                "recommended_action": "Action",
            }
        )

        selection = st.dataframe(
            display,
            use_container_width=True,
            height=360,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="queue_table",
        )

        try:
            picked_rows = list(getattr(getattr(selection, "selection", None), "rows", []) or [])
        except Exception:
            picked_rows = []
        if picked_rows:
            try:
                picked = str(display.iloc[int(picked_rows[0])].get("Shipment", "") or "").strip()
            except Exception:
                picked = ""
            if picked:
                set_selected_shipment(picked)

        # Fallback selection for users who prefer a picker over table selection.
        inspect_options = filtered.head(top_n)["shipment_id"].astype(str).tolist()
        if inspect_options:
            current = get_selected_shipment(filtered)
            # Keep this picker synced with the global selection (best-effort).
            if current in inspect_options and st.session_state.get("queue_inspect") != current:
                st.session_state["queue_inspect"] = current
            inspect_idx = inspect_options.index(current) if current in inspect_options else 0
            inspected = st.selectbox(
                "Inspect Shipment",
                options=inspect_options,
                index=inspect_idx,
                key="queue_inspect",
            )
            set_selected_shipment(inspected)
    return filtered


def render_execution_console(queue: pd.DataFrame, user: Dict[str, object]) -> None:
    with panel("Update"):
        can_update = has_permission(str(user.get("role", "viewer")), "queue_update")

        if queue.empty:
            st.write("No work rows available.")
            return

        if not can_update:
            st.info("Your role is read-only for updates.")

        ship_options = queue["shipment_id"].astype(str).tolist()
        current = get_selected_shipment(queue)
        if current in ship_options and st.session_state.get("update_pick_shipment") != current:
            st.session_state["update_pick_shipment"] = current
        selected = st.selectbox("Shipment", options=ship_options, key="update_pick_shipment")
        set_selected_shipment(selected)
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
                note = st.text_area(
                    "Note",
                    value=str(selected_row.get("note", "")),
                    height=110,
                    disabled=not can_update,
                )

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
                                invalid_df["current_status"] = invalid_df["current_status"].map(
                                    lambda x: status_display(str(x))
                                )
                            if "requested_status" in invalid_df.columns:
                                invalid_df["requested_status"] = invalid_df["requested_status"].map(
                                    lambda x: status_display(str(x))
                                )
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


def _format_number(value: object, digits: int = 3) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value or "")
    return f"{num:.{digits}f}"


def _local_shap_row(shap_local: pd.DataFrame, shipment_id: str) -> Dict[str, object]:
    if shap_local.empty:
        return {}
    view = shap_local[shap_local["shipment_id"].astype(str) == str(shipment_id)]
    if view.empty:
        return {}
    return view.iloc[0].to_dict()


def _build_triage_note(
    *,
    shipment_id: str,
    queue_row: Dict[str, object],
    feature_row: Dict[str, object],
    shap_row: Dict[str, object],
    evidence: Dict[str, object],
    activity_rows: List[Dict[str, object]],
    actor: str,
) -> str:
    lines: List[str] = []
    lines.append(f"# Triage Note - {shipment_id}")
    lines.append("")
    lines.append(f"- generated_at_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- actor: {actor}")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- risk_score: {_format_number(queue_row.get('risk_score', ''), 4)}")
    lines.append(f"- risk_band: {queue_row.get('risk_band', '')}")
    lines.append(f"- status: {queue_row.get('status', '')}")
    lines.append(f"- owner: {queue_row.get('owner', '')}")
    lines.append(f"- recommended_action: {queue_row.get('recommended_action', '')}")
    lines.append("")

    if shap_row:
        lines.append("## Model Drivers (Local SHAP)")
        for idx in [1, 2, 3]:
            feat = str(shap_row.get(f"feature_{idx}", "")).strip()
            if not feat:
                continue
            contrib = shap_row.get(f"contribution_{idx}", 0.0)
            value = feature_row.get(feat, "")
            lines.append(f"- {feat}: value={value} contribution={_format_number(contrib, 4)}")
        lines.append("")

    if evidence:
        lines.append("## Semantic Evidence (RDF)")
        for key in ["carrier", "warehouse", "customer", "product", "risk_score", "risk_band"]:
            if str(evidence.get(key, "")).strip():
                lines.append(f"- {key}: {evidence.get(key)}")
        events = evidence.get("delay_events", []) if isinstance(evidence.get("delay_events", []), list) else []
        if events:
            lines.append("")
            lines.append("### Delay Events")
            for event in events[:8]:
                if not isinstance(event, dict):
                    continue
                lines.append(f"- {event.get('cause','')} ({event.get('severity','')})")
        lines.append("")

    if activity_rows:
        lines.append("## Recent Activity")
        for row in activity_rows[:12]:
            lines.append(
                f"- {row.get('created_at','')} | {row.get('actor','')} ({row.get('actor_role','')}) | {row.get('action','')} | {row.get('reason','')}"
            )
        lines.append("")

    lines.append("## Next Steps")
    lines.append("- Confirm the driver signals with dashboards (latency, carrier performance, warehouse load).")
    lines.append("- Assign an owner and move the item to the next step with a clear ETA and note.")
    lines.append("- Escalate to an incident if the pattern affects multiple critical items.")
    lines.append("")
    return "\n".join(lines)


def render_shipment_explain_panel(
    *,
    shipment_id: str,
    queue: pd.DataFrame,
    shap_local: pd.DataFrame,
    graph,
    user: Dict[str, object],
    panel_key: str = "default",
) -> None:
    with panel("Explain"):
        panel_key = str(panel_key or "default").strip() or "default"
        shipment_id = str(shipment_id or "").strip()
        if not shipment_id:
            st.caption("Pick a shipment from Worklist / Queue / Update to see explainability and evidence.")
            return

        queue_row: Dict[str, object] = {}
        if not queue.empty and "shipment_id" in queue.columns:
            view = queue[queue["shipment_id"].astype(str) == shipment_id]
            if not view.empty:
                queue_row = view.iloc[0].to_dict()

        role = str(user.get("role", "viewer"))
        actor = str(user.get("username", "") or "")
        can_update = has_permission(role, "queue_update")
        can_incident = has_permission(role, "incident_manage")

        current_status = str(queue_row.get("status", "New") or "New")
        current_owner = str(queue_row.get("owner", "") or "")
        current_note = str(queue_row.get("note", "") or "")
        current_eta = str(queue_row.get("eta_action_at", "") or "")
        next_status = next_status_for_quick_move(current_status=current_status, actor_role=role)

        if can_update:
            st.caption("Evidence next to action: validate drivers, then execute the next step with an auditable change.")
            b1, b2, b3 = st.columns([1, 1.35, 1])
            with b1:
                assign = st.button(
                    "Assign Me",
                    key=f"{panel_key}_ex_assign_{shipment_id}",
                    use_container_width=True,
                )
            with b2:
                move_disabled = next_status == current_status
                move = st.button(
                    f"Move Next ({status_display(next_status)})",
                    key=f"{panel_key}_ex_move_{shipment_id}",
                    use_container_width=True,
                    disabled=move_disabled,
                )
            with b3:
                eta_plus = st.button(
                    "ETA +2h",
                    key=f"{panel_key}_ex_eta_{shipment_id}",
                    use_container_width=True,
                )

            if assign:
                try:
                    update_queue_action(
                        shipment_id=shipment_id,
                        status=current_status,
                        owner=actor,
                        note=current_note,
                        eta_action_at=current_eta,
                        actor=actor,
                        actor_role=role,
                        path=SERVICE_DB_PATH,
                    )
                    st.success(f"Assigned {shipment_id} to {actor}.")
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))

            if move:
                try:
                    update_queue_action(
                        shipment_id=shipment_id,
                        status=next_status,
                        owner=current_owner,
                        note=current_note,
                        eta_action_at=current_eta,
                        actor=actor,
                        actor_role=role,
                        path=SERVICE_DB_PATH,
                    )
                    st.success(f"Moved {shipment_id} to {status_display(next_status)}.")
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))

            if eta_plus:
                try:
                    eta_new = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
                    update_queue_action(
                        shipment_id=shipment_id,
                        status=current_status,
                        owner=current_owner,
                        note=current_note,
                        eta_action_at=eta_new,
                        actor=actor,
                        actor_role=role,
                        path=SERVICE_DB_PATH,
                    )
                    st.success(f"ETA +2h set for {shipment_id}.")
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))
        else:
            st.caption("Role is read-only for queue updates; explainability is available but actions are disabled.")

        feature_row: Dict[str, object] = {}
        try:
            feature_row = fetch_shipment_feature_row(SQLITE_PATH, shipment_id)
        except Exception:
            feature_row = {}

        shap_row = _local_shap_row(shap_local, shipment_id)

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            st.metric("Risk", _format_number(queue_row.get("risk_score", ""), 4))
        with c2:
            st.metric("Band", str(queue_row.get("risk_band", "") or "-"))
        with c3:
            st.metric("Step", status_display(str(queue_row.get("status", "") or "-")))
        with c4:
            st.metric("Owner", str(queue_row.get("owner", "") or "-"))

        st.caption(
            f"Action: {queue_row.get('recommended_action','-')} | Cause: {driver_display(queue_row.get('key_driver',''))}"
        )

        # Local SHAP drivers.
        if shap_row:
            driver_rows: List[Dict[str, object]] = []
            for idx in [1, 2, 3]:
                feat = str(shap_row.get(f"feature_{idx}", "")).strip()
                if not feat:
                    continue
                driver_rows.append(
                    {
                        "Driver": driver_display(feat),
                        "feature": feat,
                        "value": feature_row.get(feat, ""),
                        "contribution": float(shap_row.get(f"contribution_{idx}", 0.0) or 0.0),
                    }
                )

            if driver_rows:
                df = pd.DataFrame(driver_rows)
                fig = go.Figure()
                colors = ["#0f625b" if v >= 0 else "#ad5e4a" for v in df["contribution"].tolist()]
                fig.add_trace(
                    go.Bar(
                        x=df["contribution"],
                        y=df["Driver"],
                        orientation="h",
                        marker_color=colors,
                        opacity=0.88,
                    )
                )
                fig.update_layout(
                    margin=dict(l=8, r=8, t=10, b=8),
                    height=220,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,.08)", title="Contribution"),
                    yaxis=dict(showgrid=False, title=""),
                )
                st.caption("Local SHAP Drivers (top-3)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                show = df[["Driver", "value", "contribution"]].copy()
                show["contribution"] = show["contribution"].map(lambda v: _format_number(v, 4))
                st.dataframe(show, use_container_width=True, height=160)
        else:
            st.caption("No local SHAP row found for this shipment. Re-run the pipeline to refresh explanations.")

        # Semantic evidence from RDF graph.
        evidence: Dict[str, object] = {}
        if graph is None:
            st.caption("Semantic evidence: instance graph not available yet.")
        else:
            try:
                evidence = query_shipment_evidence(graph, shipment_id).as_dict()
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Semantic evidence error: {exc}")
                evidence = {}

        if evidence:
            st.caption("Semantic Evidence (RDF)")
            e1, e2 = st.columns([1, 1])
            with e1:
                st.write(
                    {
                        "carrier": evidence.get("carrier", ""),
                        "warehouse": evidence.get("warehouse", ""),
                        "customer": evidence.get("customer", ""),
                        "product": evidence.get("product", ""),
                    }
                )
            with e2:
                st.write(
                    {
                        "risk_score": evidence.get("risk_score", ""),
                        "risk_band": evidence.get("risk_band", ""),
                        "delay_hours": evidence.get("delay_hours", ""),
                        "delivered_on_time": evidence.get("delivered_on_time", ""),
                    }
                )
            events = evidence.get("delay_events", [])
            if isinstance(events, list) and events:
                st.dataframe(pd.DataFrame(events), use_container_width=True, height=180)

        # Activity/audit preview (safe to show; already RBAC-gated via actions).
        activity_rows = fetch_activity(shipment_id, path=SERVICE_DB_PATH, limit=12)
        if activity_rows:
            activity_df = pd.DataFrame(
                [
                    {
                        "created_at": row.get("created_at"),
                        "actor": row.get("actor"),
                        "role": row.get("actor_role"),
                        "action": row.get("action"),
                        "reason": row.get("reason"),
                    }
                    for row in activity_rows
                ]
            )
            st.caption("Recent Activity (Audit)")
            st.dataframe(activity_df, use_container_width=True, height=210)

        note = _build_triage_note(
            shipment_id=shipment_id,
            queue_row=queue_row,
            feature_row=feature_row,
            shap_row=shap_row,
            evidence=evidence,
            activity_rows=activity_rows,
            actor=actor or "unknown",
        )
        st.download_button(
            "Download Triage Note (Markdown)",
            data=note.encode("utf-8"),
            file_name=f"triage-{shipment_id}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"{panel_key}_triage_{shipment_id}",
        )

        if can_incident:
            risk_band = str(queue_row.get("risk_band", "") or evidence.get("risk_band", "") or "Low")
            key_driver = str(queue_row.get("key_driver", "") or "").strip()
            severity = "SEV-3"
            if risk_band == "Critical":
                severity = "SEV-1"
            elif risk_band == "High":
                severity = "SEV-2"

            title = f"{risk_band} delay risk requires action ({driver_display(key_driver) or 'unknown driver'})"
            rule_id = f"manual_shipment_{shipment_id}"
            desc_lines = [
                f"Shipment: {shipment_id}",
                f"Risk: {risk_band} (score={_format_number(queue_row.get('risk_score', ''), 4)})",
                f"Driver: {driver_display(key_driver)}",
                f"Action: {queue_row.get('recommended_action','')}",
                f"Carrier: {evidence.get('carrier','')}",
                f"Warehouse: {evidence.get('warehouse','')}",
                "",
                "Triage note (excerpt):",
                note[:1200],
            ]
            description = "\n".join([str(x) for x in desc_lines if str(x).strip()])

            if st.button(
                "Escalate: Create/Update Incident (Deduplicated)",
                key=f"{panel_key}_ex_inc_{shipment_id}",
                use_container_width=True,
            ):
                try:
                    result = upsert_incident_from_recommendation(
                        recommendation={
                            "rule_id": rule_id,
                            "title": title,
                            "severity": severity,
                            "description": description,
                        },
                        owner=current_owner or actor,
                        actor=actor or "operator",
                        actor_role=role,
                        path=SERVICE_DB_PATH,
                    )
                    incident_id = result.get("incident_id")
                    if bool(result.get("deduplicated")):
                        st.success(f"Incident updated (deduplicated): {incident_id}")
                    else:
                        st.success(f"Incident created: {incident_id}")
                    st.rerun()
                except (ValueError, PermissionError) as exc:
                    st.error(str(exc))


def render_incident_console(user: Dict[str, object]) -> None:
    with panel("Incident"):
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
                labels = [f"{item.get('severity')} | {item.get('title')} ({item.get('rule_id')})" for item in recommendations]
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


def fetch_activity_window(*, start_utc: str, end_utc: str, limit: int = 200) -> List[Dict[str, object]]:
    import sqlite3

    start_raw = str(start_utc or "").strip()
    end_raw = str(end_utc or "").strip()
    if not start_raw:
        start_raw = "0001-01-01T00:00:00+00:00"
    if not end_raw:
        end_raw = datetime.now(timezone.utc).isoformat()
    if start_raw > end_raw:
        start_raw, end_raw = end_raw, start_raw

    init_service_store(SERVICE_DB_PATH)
    conn = sqlite3.connect(SERVICE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, actor, actor_role, action, entity_type, entity_id,
                   payload_json, previous_state_json, new_state_json,
                   reason, request_id, prev_hash, event_hash, created_at
            FROM service_activity_log
            WHERE created_at >= ? AND created_at <= ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (start_raw, end_raw, int(max(1, limit))),
        ).fetchall()
    finally:
        conn.close()

    parsed: List[Dict[str, object]] = []
    for row in rows:
        item = dict(row)
        for key in ["payload_json", "previous_state_json", "new_state_json"]:
            try:
                item[key.replace("_json", "")] = json.loads(item.get(key, "{}") or "{}")
            except json.JSONDecodeError:
                item[key.replace("_json", "")] = {"raw": item.get(key, "")}
        parsed.append(item)
    return parsed


def _build_postmortem_draft(incident: Dict[str, object], timeline_rows: List[Dict[str, object]]) -> str:
    incident_id = str(incident.get("incident_id", "") or "").strip()
    title = str(incident.get("title", "") or "").strip()
    severity = str(incident.get("severity", "") or "").strip()
    status = str(incident.get("status", "") or "").strip()
    owner = str(incident.get("owner", "") or "").strip()
    opened_at = str(incident.get("opened_at", "") or "").strip()
    closed_at = str(incident.get("closed_at", "") or "").strip()

    start_dt = _parse_iso_dt(opened_at) if opened_at else None
    end_dt = _parse_iso_dt(closed_at) if closed_at else None
    if end_dt is None:
        end_dt = datetime.now(timezone.utc)
    if start_dt is None:
        start_dt = end_dt - timedelta(hours=24)

    duration_sec = max(0.0, (end_dt - start_dt).total_seconds())
    duration_min = int(round(duration_sec / 60.0))
    duration_h = duration_sec / 3600.0

    rows = list(timeline_rows or [])
    # Timeline should read oldest -> newest.
    rows = list(reversed(rows))

    md: List[str] = []
    md.append("# Postmortem")
    md.append("")
    md.append("Status: Draft")
    md.append("")
    md.append("## Metadata")
    md.append(f"- Incident ID: {incident_id}")
    md.append(f"- Title: {title}")
    md.append(f"- Severity: {severity}")
    md.append(f"- Status: {status}")
    md.append(f"- Owners (DRI): {owner}")
    md.append(f"- Start time (UTC): {start_dt.isoformat()}")
    md.append(f"- End time (UTC): {end_dt.isoformat()}")
    md.append(f"- Duration: ~{duration_min} min ({duration_h:.2f}h)")
    md.append("")
    md.append("## Summary")
    md.append(
        "Describe what happened in 3-5 sentences (what, so-what, now-what). Include the operational outcome."
    )
    md.append("")
    md.append("## Customer / User Impact")
    md.append("- Who was impacted:")
    md.append("- What was impacted:")
    md.append("- Impact window:")
    md.append("- Business impact (if applicable):")
    md.append("")
    md.append("## Detection")
    md.append("- How was it detected (alert, operator report, dashboard, etc.):")
    md.append("- Time to detect (TTD):")
    md.append("- Gaps in detection (if any):")
    md.append("")
    md.append("## Root Cause")
    md.append("- Primary root cause:")
    md.append("- Contributing factors:")
    md.append("- Why it was not prevented:")
    md.append("")
    md.append("## Resolution & Recovery")
    md.append("- What stopped the impact:")
    md.append("- Recovery steps taken:")
    md.append("- Time to mitigate (TTM):")
    md.append("- Time to recover (TTR/MTTR):")
    md.append("")
    md.append("## Timeline (UTC)")
    md.append("| Time | Event | Actor | Notes |")
    md.append("| --- | --- | --- | --- |")
    if rows:
        for row in rows[:30]:
            t = str(row.get("created_at", "") or "")
            event = f"{row.get('action','')} ({row.get('entity_type','')}:{row.get('entity_id','')})"
            actor = f"{row.get('actor','')} ({row.get('actor_role','')})"
            notes = str(row.get("reason", "") or "")
            md.append(f"| {t} | {event} | {actor} | {notes} |")
    else:
        md.append("|  |  |  |  |")
    md.append("")
    md.append("## What Went Well")
    md.append("-")
    md.append("")
    md.append("## What Went Wrong")
    md.append("-")
    md.append("")
    md.append("## Action Items")
    md.append("| # | Action | Owner | Due Date | Priority | Status |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    md.append("| 1 |  |  |  | P0/P1/P2 | Open |")
    md.append("")
    md.append("## Follow-ups / Links")
    md.append("- Dashboards:")
    md.append("- Logs:")
    md.append("- Runbooks:")
    md.append("- Related docs:")
    md.append("")
    return "\n".join(md)


def render_postmortem_export_panel(user: Dict[str, object]) -> None:
    with panel("Postmortem Export"):
        if not has_permission(str(user.get("role", "viewer")), "incident_manage"):
            st.caption("Postmortem export is available in Operator/Admin roles.")
            return

        incidents = list_incidents(SERVICE_DB_PATH, limit=30)
        if not incidents:
            st.caption("No incidents to export yet.")
            return

        labels = [
            f"{row.get('incident_id')} | {row.get('severity')} | {row.get('status')} | {str(row.get('title',''))[:70]}"
            for row in incidents
        ]
        selected_label = st.selectbox("Select Incident", options=labels, key="pm_incident_pick")
        idx = labels.index(selected_label)
        incident = incidents[idx]

        max_events = st.slider("Timeline events", min_value=20, max_value=300, value=120, step=20)

        opened_at = str(incident.get("opened_at", "") or "").strip()
        closed_at = str(incident.get("closed_at", "") or "").strip()
        end_utc = closed_at or datetime.now(timezone.utc).isoformat()
        start_utc = opened_at or (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()

        activity_rows = fetch_activity_window(start_utc=start_utc, end_utc=end_utc, limit=max_events)
        postmortem = _build_postmortem_draft(incident, activity_rows)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.download_button(
                "Download Postmortem Draft (Markdown)",
                data=postmortem.encode("utf-8"),
                file_name=f"postmortem-{incident.get('incident_id')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with c2:
            incident_update = (
                f"[{incident.get('severity')}] {incident.get('title')}\n"
                f"Status: {incident.get('status')} | Owner: {incident.get('owner')}\n"
                f"Opened: {opened_at or '-'}\n"
                f"Next: review risk queue + assign work items; track actions in audit log.\n"
            )
            st.text_area("Incident Comment Draft", value=incident_update, height=120)

        with st.expander("Preview"):
            st.code(postmortem[:6000])


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_evidence_pack_bytes(*, include_instance_graph: bool = True, include_ops_report: bool = True) -> Tuple[str, bytes]:
    now_utc = datetime.now(timezone.utc).isoformat()

    paths: List[Tuple[str, Path]] = [
        ("README.md", ROOT / "README.md"),
        ("RUNBOOK.md", ROOT / "RUNBOOK.md"),
        ("specs/FLAGSHIP_V3_SPEC.md", ROOT / "specs" / "FLAGSHIP_V3_SPEC.md"),
        ("data/output/monitoring_metrics.json", METRICS_PATH),
        ("data/output/model_comparison.json", MODEL_COMPARISON_PATH),
        ("data/output/training_summary.json", TRAINING_PATH),
        ("data/output/data_quality_report.json", QUALITY_PATH),
        ("data/output/sparql_results.json", SPARQL_PATH),
        ("data/output/shap_global_importance.csv", SHAP_GLOBAL_PATH),
        ("data/output/shap_local_explanations.csv", SHAP_LOCAL_PATH),
        ("data/output/daily_risk_queue.csv", ROOT / "data" / "output" / "daily_risk_queue.csv"),
        ("data/output/pipeline_status.json", PIPELINE_STATUS_PATH),
        ("data/output/datadog_series_payload.json", ROOT / "data" / "output" / "datadog_series_payload.json"),
    ]
    if include_ops_report:
        paths.append(("data/output/ops_report.html", ROOT / "data" / "output" / "ops_report.html"))
    if include_instance_graph:
        paths.append(("data/semantic/instance_graph.ttl", RDF_INSTANCE_PATH))

    audit = verify_audit_chain(path=SERVICE_DB_PATH, limit=10000)
    manifest = {
        "generated_at_utc": now_utc,
        "project": "the-logistics-prophet",
        "audit_verification": audit,
        "files": [],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in paths:
            if not path.exists():
                continue
            data = path.read_bytes()
            zf.writestr(arcname, data)
            manifest["files"].append(
                {
                    "path": arcname,
                    "size_bytes": len(data),
                    "sha256": _sha256_hex(data),
                    "mtime_epoch": int(path.stat().st_mtime),
                }
            )

        manifest_bytes = json.dumps(manifest, ensure_ascii=True, indent=2).encode("utf-8")
        zf.writestr("evidence/manifest.json", manifest_bytes)

    filename = f"logistics-prophet-evidence-pack-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.zip"
    return filename, buf.getvalue()


def render_evidence_pack_panel(user: Dict[str, object]) -> None:
    with panel("Evidence Pack", caption="Build a reviewer-friendly ZIP with key artifacts + integrity manifest (offline)."):
        include_graph = st.checkbox("Include RDF instance graph (TTL)", value=True, key="pack_include_graph")
        include_ops = st.checkbox("Include ops report (HTML)", value=True, key="pack_include_ops")

        if st.button("Build Evidence Pack", use_container_width=True):
            try:
                filename, data = build_evidence_pack_bytes(
                    include_instance_graph=bool(include_graph),
                    include_ops_report=bool(include_ops),
                )
                st.session_state["evidence_pack_filename"] = filename
                st.session_state["evidence_pack_bytes"] = data
                st.success(f"Evidence pack built: {filename} ({len(data) / 1024 / 1024:.2f} MB)")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to build evidence pack: {exc}")

        if st.session_state.get("evidence_pack_bytes"):
            st.download_button(
                "Download Evidence Pack (ZIP)",
                data=st.session_state["evidence_pack_bytes"],
                file_name=str(st.session_state.get("evidence_pack_filename") or "evidence-pack.zip"),
                mime="application/zip",
                use_container_width=True,
            )


def render_governance_overview_panel() -> None:
    with panel("Governance"):
        audit_result = verify_audit_chain(path=SERVICE_DB_PATH, limit=5000)
        valid = audit_result.get("valid")
        checked = audit_result.get("checked")
        latest = audit_result.get("latest_hash", "")

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Audit Chain", "Valid" if valid else "Invalid")
        with c2:
            st.metric("Checked", int(checked or 0))
        with c3:
            st.caption(f"Latest hash: `{str(latest)[:18]}...`" if latest else "Latest hash: -")

        runs = list_pipeline_runs(SERVICE_DB_PATH, limit=12)
        if runs:
            st.caption("Pipeline Runs")
            st.dataframe(pd.DataFrame(runs), use_container_width=True, height=240)

        recent = pd.DataFrame(list_recent_activity(SERVICE_DB_PATH, limit=60))
        if not recent.empty:
            show_cols = ["created_at", "actor", "actor_role", "action", "entity_type", "entity_id", "reason"]
            show = recent[[col for col in show_cols if col in recent.columns]].copy()
            if "created_at" in show.columns:
                show.insert(0, "age", show["created_at"].map(lambda x: _age_text(str(x))))
            st.caption("Recent Activity")
            st.dataframe(show, use_container_width=True, height=320)
        else:
            st.caption("No recent activity.")


def render_admin_panel(user: Dict[str, object]) -> None:
    if not has_permission(str(user.get("role", "viewer")), "user_manage"):
        return
    with panel("Admin: Users"):
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


def main() -> None:
    init_service_store(SERVICE_DB_PATH)
    mode = ui_mode()
    apply_cinematic_css_if_enabled(mode)

    # Fast way to isolate "blank page" problems (browser/JS/websocket/etc).
    # If this doesn't render, the issue is almost certainly on the client side.
    smoke = (_query_param("smoke") or "").strip().lower()
    if smoke in {"1", "true", "yes", "on"}:
        st.title("Smoke Test")
        st.write("If you can read this, the Streamlit frontend is running.")
        st.write({"ui_mode": mode})
        st.stop()

    # Reliability-first: no global CSS injection. The UI is built with Streamlit primitives
    # (containers/columns/tables) so it remains stable across browsers and GPU stacks.

    user = render_login_gate()

    pipeline_status = load_json(PIPELINE_STATUS_PATH)
    metrics = load_json(METRICS_PATH)
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
    page = render_nav(mode)

    if page == "Control Tower":
        quality = load_json(QUALITY_PATH)
        training = load_json(TRAINING_PATH)
        queue = load_service_queue()
        service_summary = fetch_queue_summary(SERVICE_DB_PATH)
        ops_health = fetch_ops_health(SERVICE_DB_PATH)
        core_snapshot = fetch_service_core_snapshot(SERVICE_DB_PATH)
        workflow_sla = fetch_workflow_sla_snapshot(SERVICE_DB_PATH)

        render_hero(
            metrics,
            training,
            quality,
            queue,
            service_summary,
            show_banner=(mode != "plain"),
        )
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
1. Go to **Worklist** and start with the highest urgency.
2. Use quick buttons: **Assign Me**, **Move Next**, **ETA +2h**.
3. Use **Queue + Update** to add note/ETA or do bulk changes.
4. Use **Incidents** when suggestions trigger or risk grows.
                """.strip()
            )
        render_service_core_board(core_snapshot)
        c1, c2 = st.columns([1.15, 1])
        with c1:
            render_workflow_sla_panel(workflow_sla)
        with c2:
            render_ops_health_panel(service_summary, ops_health)
        return

    if page == "Worklist":
        queue = load_service_queue()
        core_worklist = fetch_service_core_worklist(SERVICE_DB_PATH, per_stage_limit=6)
        shap_local = load_shap_local(SHAP_LOCAL_PATH)
        with st.spinner("Loading semantic graph (RDF)..."):
            rdf_graph = load_rdf_graph(str(RDF_INSTANCE_PATH))

        left, right = st.columns([1.35, 1])
        with left:
            render_service_core_worklist(core_worklist, user=user, queue=queue)
        with right:
            render_shipment_explain_panel(
                shipment_id=get_selected_shipment(queue),
                queue=queue,
                shap_local=shap_local,
                graph=rdf_graph,
                user=user,
                panel_key="worklist",
            )
        return

    if page == "Queue + Update":
        queue = load_service_queue()
        shap_local = load_shap_local(SHAP_LOCAL_PATH)
        with st.spinner("Loading semantic graph (RDF)..."):
            rdf_graph = load_rdf_graph(str(RDF_INSTANCE_PATH))

        left, right = st.columns([1.55, 1])
        with left:
            filtered = render_queue_panel(queue)
        with right:
            render_execution_console(filtered if not filtered.empty else queue, user)

        render_shipment_explain_panel(
            shipment_id=get_selected_shipment(filtered if not filtered.empty else queue),
            queue=queue,
            shap_local=shap_local,
            graph=rdf_graph,
            user=user,
            panel_key="queue_update",
        )
        return

    if page == "Incidents":
        render_incident_console(user)
        render_postmortem_export_panel(user)
        return

    if page == "Insights":
        quality = load_json(QUALITY_PATH)
        sparql = load_json(SPARQL_PATH)
        shap_global = load_shap_global(SHAP_GLOBAL_PATH)
        model_comparison = load_json(MODEL_COMPARISON_PATH)
        training = load_json(TRAINING_PATH)
        kpi_series = load_kpi_series()

        render_analysis_panel(kpi_series, shap_global, quality)
        left, right = st.columns([1.1, 1])
        with left:
            render_model_selection_panel(training, model_comparison)
            render_quality_details_panel(quality)
        with right:
            render_semantic_panel(sparql)
        return

    # Governance
    render_evidence_pack_panel(user)
    render_governance_overview_panel()
    render_admin_panel(user)


if __name__ == "__main__":
    main()
