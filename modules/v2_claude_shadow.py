"""
V2 Claude shadow schema and validator.

Claude output is LOGGED ONLY in V2A — it cannot create orders.
Any invalid, missing, or unknown output → unavailable sentinel (not fabricated neutral).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CLAUDE_SHADOW_SCHEMA_VERSION = "shadow_v1"

_SIGNAL_DIRECTIONS = {"bullish", "bearish", "neutral", "unavailable"}
_EVIDENCE_STRENGTHS = {"strong", "moderate", "weak", "unavailable"}
_GUIDANCE_CHANGES = {"raised", "lowered", "maintained", "not_reported", "unavailable"}
_REVISION_DIRECTIONS = {"upward", "downward", "unchanged", "unavailable"}
_TRENDS = {"improving", "stable", "deteriorating", "unavailable"}
_QUALITIES = {"high", "medium", "low", "unavailable"}
_STRENGTHS = {"strong", "moderate", "weak", "none", "unavailable"}
_UNCERTAINTIES = {"high", "medium", "low", "unavailable"}


def shadow_unavailable(reason: str = "not_requested") -> dict[str, Any]:
    """Return an unavailable shadow record — used when Claude is not called."""
    return {
        "schema_version": CLAUDE_SHADOW_SCHEMA_VERSION,
        "signal_direction": "unavailable",
        "evidence_strength": "unavailable",
        "guidance_change": "unavailable",
        "estimate_revision_direction": "unavailable",
        "margin_trend": "unavailable",
        "earnings_quality": "unavailable",
        "capital_allocation_quality": "unavailable",
        "thesis_risks": [],
        "catalyst_strength": "unavailable",
        "uncertainty": "unavailable",
        "source_ids": [],
        "source_published_at": [],
        "model_id": None,
        "prompt_version": None,
        "generated_at": None,
        "data_cutoff_at": None,
        "content_hash": None,
        "unavailable_reason": reason,
        "order_creation_blocked": True,
    }


def validate_claude_shadow_output(raw: Any) -> dict[str, Any]:
    """
    Validate and normalize Claude shadow output against the V2 schema.

    Any missing, invalid, or unknown field → "unavailable".
    Does NOT use regex as primary validation — uses explicit enum membership.
    order_creation_blocked is always True regardless of input.
    """
    if not isinstance(raw, dict):
        return {
            **shadow_unavailable("invalid_type"),
            "raw_error": f"Expected dict, got {type(raw).__name__}",
        }

    def _enum(key: str, valid: set) -> str:
        v = raw.get(key)
        return v if (v is not None and v in valid) else "unavailable"

    def _lst(key: str) -> list:
        v = raw.get(key)
        return [str(x) for x in v if x is not None] if isinstance(v, list) else []

    def _str_or_none(key: str) -> str | None:
        v = raw.get(key)
        return v.strip() if (v and isinstance(v, str) and v.strip()) else None

    validated: dict[str, Any] = {
        "schema_version": CLAUDE_SHADOW_SCHEMA_VERSION,
        "signal_direction": _enum("signal_direction", _SIGNAL_DIRECTIONS),
        "evidence_strength": _enum("evidence_strength", _EVIDENCE_STRENGTHS),
        "guidance_change": _enum("guidance_change", _GUIDANCE_CHANGES),
        "estimate_revision_direction": _enum("estimate_revision_direction", _REVISION_DIRECTIONS),
        "margin_trend": _enum("margin_trend", _TRENDS),
        "earnings_quality": _enum("earnings_quality", _QUALITIES),
        "capital_allocation_quality": _enum("capital_allocation_quality", _QUALITIES),
        "thesis_risks": _lst("thesis_risks"),
        "catalyst_strength": _enum("catalyst_strength", _STRENGTHS),
        "uncertainty": _enum("uncertainty", _UNCERTAINTIES),
        "source_ids": _lst("source_ids"),
        "source_published_at": _lst("source_published_at"),
        "model_id": _str_or_none("model_id"),
        "prompt_version": _str_or_none("prompt_version"),
        "generated_at": _str_or_none("generated_at"),
        "data_cutoff_at": _str_or_none("data_cutoff_at"),
        "order_creation_blocked": True,  # Always True in V2A — not overridable by input
    }

    canonical = json.dumps(
        {k: v for k, v in validated.items() if k != "content_hash"},
        sort_keys=True, default=str,
    )
    validated["content_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return validated


def assert_shadow_cannot_create_order(shadow_output: dict[str, Any]) -> None:
    """Raise RuntimeError if shadow output has order_creation_blocked != True."""
    if not shadow_output.get("order_creation_blocked", False):
        raise RuntimeError(
            "V2A invariant violated: Claude shadow output must have "
            "order_creation_blocked=True. Shadow analysis cannot create orders."
        )
