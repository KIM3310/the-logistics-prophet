#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control_tower.semantic_layer import build_instance_graph, run_competency_queries


def main() -> None:
    graph_summary = build_instance_graph()
    query_summary = run_competency_queries()

    payload = {
        "graph": graph_summary,
        "queries": {
            "triple_count": query_summary.get("triple_count"),
            "query_count": len(query_summary.get("queries", [])),
            "output_path": "data/output/sparql_results.json",
        },
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
