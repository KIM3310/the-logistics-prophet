from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

_openai: Any | None = None
try:  # pragma: no cover - exercised when optional dependency is installed
    import openai as _openai  # type: ignore[import]
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    pass

openai: Any | None = _openai

logger = logging.getLogger("control_tower.screening_agent")

RISK_BAND_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

_ACTION_TEMPLATES: Dict[str, str] = {
    "Critical": "Immediate escalation required",
    "High": "Priority review needed",
    "Medium": "Preventive monitoring recommended",
    "Low": "Routine tracking sufficient",
}


def _band_sort_key(shipment: Dict[str, object]) -> Tuple[int, float]:
    band = str(shipment.get("risk_band", "Low"))
    score = float(shipment.get("risk_score", 0.0) or 0.0)
    return (RISK_BAND_ORDER.get(band, 99), -score)


def prioritize_shipments(
    shipments: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Sort shipments by risk band (Critical first) then descending risk score."""
    return sorted(shipments, key=_band_sort_key)


def _count_by_band(shipments: List[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for s in shipments:
        band = str(s.get("risk_band", "Low"))
        if band in counts:
            counts[band] += 1
        else:
            counts[band] = counts.get(band, 0) + 1
    return counts


def _top_drivers(shipments: List[Dict[str, object]], top_k: int = 3) -> List[str]:
    driver_freq: Dict[str, int] = {}
    for s in shipments:
        driver = str(s.get("key_driver", "")).strip()
        if driver:
            driver_freq[driver] = driver_freq.get(driver, 0) + 1
    sorted_drivers = sorted(driver_freq.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in sorted_drivers[:top_k]]


def generate_ops_summary(
    shipments: List[Dict[str, object]],
    batch_label: Optional[str] = None,
) -> str:
    """Generate a natural language operations summary for a batch of shipments."""
    if not shipments:
        return "No shipments in batch. Operations queue is clear."

    counts = _count_by_band(shipments)
    total = len(shipments)
    critical = counts.get("Critical", 0)
    high = counts.get("High", 0)
    medium = counts.get("Medium", 0)
    low = counts.get("Low", 0)

    top_drivers = _top_drivers(shipments)
    driver_str = ", ".join(top_drivers) if top_drivers else "mixed signals"

    avg_score = sum(float(s.get("risk_score", 0.0) or 0.0) for s in shipments) / total

    label_prefix = f"Batch '{batch_label}': " if batch_label else ""

    lines: List[str] = [
        f"{label_prefix}{total} shipments screened. "
        f"Risk distribution: {critical} Critical, {high} High, {medium} Medium, {low} Low.",
        f"Average risk score: {avg_score:.3f}. Top delay drivers: {driver_str}.",
    ]

    if critical > 0:
        lines.append(
            f"URGENT: {critical} shipment(s) require immediate escalation. "
            "Assign owners and initiate ETA recovery protocols now."
        )
    elif high > 0:
        lines.append(
            f"{high} high-risk shipment(s) require priority review within the next 2 hours."
        )
    else:
        lines.append(
            "No critical or high-risk items. Continue standard monitoring procedures."
        )

    return " ".join(lines)


def generate_action_items(
    shipments: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Generate structured action items for the operations team."""
    prioritized = prioritize_shipments(shipments)
    action_items: List[Dict[str, object]] = []

    for shipment in prioritized:
        band = str(shipment.get("risk_band", "Low"))
        score = float(shipment.get("risk_score", 0.0) or 0.0)
        shipment_id = str(shipment.get("shipment_id", "unknown"))
        key_driver = str(shipment.get("key_driver", "unknown"))
        recommended_action = str(
            shipment.get("recommended_action")
            or _ACTION_TEMPLATES.get(band, "Monitor shipment.")
        )

        action_items.append(
            {
                "shipment_id": shipment_id,
                "risk_band": band,
                "risk_score": round(score, 4),
                "key_driver": key_driver,
                "action": recommended_action,
                "priority": RISK_BAND_ORDER.get(band, 99),
            }
        )

    return action_items


class BatchScreeningAgent:
    """Screens a batch of shipments, prioritises by risk, and produces ops outputs."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        use_llm_summary: bool = False,
    ) -> None:
        self._openai_model = openai_model
        self._use_llm_summary = use_llm_summary
        self._client: Optional[object] = None

        if use_llm_summary:
            resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
            if resolved_key:
                try:
                    if openai is None:
                        raise ImportError("openai package not installed")
                    self._client = openai.OpenAI(api_key=resolved_key)
                    logger.info("BatchScreeningAgent: OpenAI client initialised")
                except ImportError:
                    logger.warning("openai package not installed; LLM summary disabled")
                except Exception as exc:
                    logger.warning(
                        "OpenAI client init failed: %s; LLM summary disabled", exc
                    )
            else:
                logger.info("OPENAI_API_KEY not set; LLM summary disabled")

    def screen(
        self,
        shipments: List[Dict[str, object]],
        batch_label: Optional[str] = None,
    ) -> Dict[str, object]:
        """Screen a batch of shipments and return prioritised results with ops summary.

        Args:
            shipments: List of shipment dicts with risk_score, risk_band, key_driver, etc.
            batch_label: Optional label for the batch (e.g. date or run ID).

        Returns:
            Dict with keys: prioritized_shipments, ops_summary, action_items,
            risk_counts, total_screened.
        """
        logger.info(
            "Screening batch: %d shipments, label=%s", len(shipments), batch_label
        )

        prioritized = prioritize_shipments(shipments)
        risk_counts = _count_by_band(shipments)

        if self._use_llm_summary and self._client is not None:
            ops_summary = self._llm_summary(shipments, batch_label)
        else:
            ops_summary = generate_ops_summary(shipments, batch_label)

        action_items = generate_action_items(shipments)

        logger.info(
            "Screening complete: critical=%d, high=%d, total=%d",
            risk_counts.get("Critical", 0),
            risk_counts.get("High", 0),
            len(shipments),
        )

        return {
            "prioritized_shipments": prioritized,
            "ops_summary": ops_summary,
            "action_items": action_items,
            "risk_counts": risk_counts,
            "total_screened": len(shipments),
            "batch_label": batch_label or "",
        }

    def _llm_summary(
        self,
        shipments: List[Dict[str, object]],
        batch_label: Optional[str],
    ) -> str:
        """Generate LLM-powered ops summary; falls back to template on error."""
        counts = _count_by_band(shipments)
        top_drivers = _top_drivers(shipments)
        prompt = (
            f"You are a logistics control tower operations AI. "
            f"Summarize this shipment batch for the operations team in 3-4 sentences. "
            f"Batch: {len(shipments)} shipments. "
            f"Risk counts: {counts}. "
            f"Top delay drivers: {top_drivers}. "
            f"Label: {batch_label or 'unspecified'}. "
            "Be concise and operational. Highlight urgent items first. No markdown."
        )
        try:
            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self._openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.2,
            )
            text = response.choices[0].message.content or ""
            return text.strip()
        except Exception as exc:
            logger.warning("LLM summary failed: %s; using template fallback", exc)
            return generate_ops_summary(shipments, batch_label)
