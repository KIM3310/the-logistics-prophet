#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.config import (
    FEATURE_COLUMNS,
    MODEL_BINARY_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    SHAP_GLOBAL_PATH,
    SHAP_LOCAL_PATH,
    SQLITE_PATH,
    TARGET_COLUMN,
)
from control_tower.data_access import fetch_feature_frame, split_frame_by_time
from control_tower.modeling import save_json, save_model_bundle, train_model_suite, write_shap_reports
from control_tower.ops_output import write_training_summary


def main() -> None:
    frame = fetch_feature_frame(SQLITE_PATH)
    train_df, test_df = split_frame_by_time(frame, test_fraction=0.2)

    artifact, bundle = train_model_suite(train_df, test_df, FEATURE_COLUMNS, TARGET_COLUMN)

    save_json(MODEL_PATH, artifact)
    save_json(MODEL_COMPARISON_PATH, artifact)
    save_model_bundle(MODEL_BINARY_PATH, bundle)

    training_summary = write_training_summary(artifact)
    shap_summary = write_shap_reports(
        bundle=bundle,
        frame=test_df,
        output_global_path=SHAP_GLOBAL_PATH,
        output_local_path=SHAP_LOCAL_PATH,
    )

    selected = str(artifact.get("selected_model"))
    payload = {
        "artifact": str(MODEL_PATH),
        "bundle": str(MODEL_BINARY_PATH),
        "selected_model": selected,
        "selected_threshold": artifact.get("selected_threshold"),
        "selected_test_metrics": artifact.get("test_metrics", {}).get(selected, {}),
        "comparison": artifact.get("test_metrics", {}),
        "shap": shap_summary,
        "training_summary": training_summary,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
