from __future__ import annotations

import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SCHEMA_VERSION


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.30, 0.80, 51):
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return round(best_t, 3)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 4),
        "support": int(len(y_true)),
    }
    return metrics


def train_model_suite(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    x_train = train_df[feature_names].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=int)
    x_test = test_df[feature_names].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=int)

    baseline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    baseline.fit(x_train, y_train)
    baseline_train_prob = baseline.predict_proba(x_train)[:, 1]
    baseline_test_prob = baseline.predict_proba(x_test)[:, 1]
    baseline_threshold = optimize_threshold(y_train, baseline_train_prob)

    challenger_raw = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=6,
        max_iter=320,
        min_samples_leaf=30,
        l2_regularization=0.08,
        random_state=42,
    )
    challenger_raw.fit(x_train, y_train)

    challenger = CalibratedClassifierCV(
        estimator=HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=6,
            max_iter=320,
            min_samples_leaf=30,
            l2_regularization=0.08,
            random_state=42,
        ),
        method="isotonic",
        cv=3,
    )
    challenger.fit(x_train, y_train)

    challenger_train_prob = challenger.predict_proba(x_train)[:, 1]
    challenger_test_prob = challenger.predict_proba(x_test)[:, 1]
    challenger_threshold = optimize_threshold(y_train, challenger_train_prob)

    baseline_train_metrics = compute_metrics(y_train, baseline_train_prob, baseline_threshold)
    baseline_test_metrics = compute_metrics(y_test, baseline_test_prob, baseline_threshold)

    challenger_train_metrics = compute_metrics(y_train, challenger_train_prob, challenger_threshold)
    challenger_test_metrics = compute_metrics(y_test, challenger_test_prob, challenger_threshold)

    if (
        challenger_test_metrics["auc"] > baseline_test_metrics["auc"]
        or (
            challenger_test_metrics["auc"] == baseline_test_metrics["auc"]
            and challenger_test_metrics["brier"] < baseline_test_metrics["brier"]
        )
    ):
        selected_name = "challenger_calibrated_hgb"
        selected_model = challenger
        explanation_model = challenger_raw
        selected_threshold = challenger_threshold
        selection_reason = "Selected challenger by higher test AUC (tie-breaker: lower Brier)."
    else:
        selected_name = "baseline_logistic"
        selected_model = baseline
        explanation_model = baseline
        selected_threshold = baseline_threshold
        selection_reason = "Selected baseline by higher or equal generalization quality."

    model_bundle = {
        "feature_names": feature_names,
        "selected_name": selected_name,
        "selected_threshold": float(selected_threshold),
        "selected_model": selected_model,
        "explanation_model": explanation_model,
        "baseline_model": baseline,
        "challenger_model": challenger,
    }

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names,
        "selected_model": selected_name,
        "selected_threshold": float(selected_threshold),
        "selection_reason": selection_reason,
        "train_metrics": {
            "baseline_logistic": baseline_train_metrics,
            "challenger_calibrated_hgb": challenger_train_metrics,
        },
        "test_metrics": {
            "baseline_logistic": baseline_test_metrics,
            "challenger_calibrated_hgb": challenger_test_metrics,
        },
    }

    return artifact, model_bundle


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def save_model_bundle(path: Path, bundle: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bundle, f)


def load_model_bundle(path: Path) -> Dict[str, object]:
    with path.open("rb") as f:
        return pickle.load(f)


def predict_scores(bundle: Dict[str, object], x: np.ndarray) -> np.ndarray:
    model = bundle["selected_model"]
    prob = model.predict_proba(x)[:, 1]
    return prob


def _tree_shap_values(model: object, x: np.ndarray) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values

    if hasattr(values, "values"):
        values = values.values

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def _linear_contributions(model: Pipeline, x: np.ndarray) -> np.ndarray:
    scaler: StandardScaler = model.named_steps["scaler"]
    linear_model: LogisticRegression = model.named_steps["model"]
    x_scaled = scaler.transform(x)
    try:
        import shap

        explainer = shap.LinearExplainer(linear_model, x_scaled)
        shap_values = explainer.shap_values(x_scaled)
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, 1]
        return values
    except Exception:
        return x_scaled * linear_model.coef_[0]


def feature_contributions(bundle: Dict[str, object], x: np.ndarray) -> np.ndarray:
    selected_name = str(bundle["selected_name"])
    if selected_name == "baseline_logistic":
        return _linear_contributions(bundle["selected_model"], x)
    return _tree_shap_values(bundle["explanation_model"], x)


def write_shap_reports(
    bundle: Dict[str, object],
    frame: pd.DataFrame,
    output_global_path: Path,
    output_local_path: Path,
    top_local: int = 120,
) -> Dict[str, object]:
    feature_names: List[str] = list(bundle["feature_names"])
    x = frame[feature_names].to_numpy(dtype=float)
    probs = predict_scores(bundle, x)
    contrib = feature_contributions(bundle, x)

    abs_contrib = np.abs(contrib)
    mean_abs = abs_contrib.mean(axis=0)

    global_rows = sorted(
        [{"feature": feature_names[i], "mean_abs_contribution": round(float(mean_abs[i]), 6)} for i in range(len(feature_names))],
        key=lambda r: r["mean_abs_contribution"],
        reverse=True,
    )

    output_global_path.parent.mkdir(parents=True, exist_ok=True)
    with output_global_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "mean_abs_contribution"])
        writer.writeheader()
        writer.writerows(global_rows)

    rank_idx = np.argsort(probs)[::-1][: min(top_local, len(probs))]
    local_rows: List[Dict[str, object]] = []
    for idx in rank_idx:
        local_abs = abs_contrib[idx]
        top3 = np.argsort(local_abs)[::-1][:3]
        row: Dict[str, object] = {
            "shipment_id": str(frame.iloc[idx].get("shipment_id", "")),
            "risk_score": round(float(probs[idx]), 6),
        }
        for j, feat_idx in enumerate(top3, start=1):
            row[f"feature_{j}"] = feature_names[int(feat_idx)]
            row[f"contribution_{j}"] = round(float(contrib[idx, int(feat_idx)]), 6)
        local_rows.append(row)

    output_local_path.parent.mkdir(parents=True, exist_ok=True)
    with output_local_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shipment_id",
                "risk_score",
                "feature_1",
                "contribution_1",
                "feature_2",
                "contribution_2",
                "feature_3",
                "contribution_3",
            ],
        )
        writer.writeheader()
        writer.writerows(local_rows)

    return {
        "global_top_feature": global_rows[0]["feature"] if global_rows else "",
        "local_rows": len(local_rows),
    }
