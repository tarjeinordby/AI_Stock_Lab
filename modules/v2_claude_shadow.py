"""
V2 Claude shadow schema and validator.

Claude output is LOGGED ONLY in V2A — it cannot create orders.
Any invalid, missing, or unknown output → unavailable sentinel (not fabricated neutral).
Provenance is validated: ISO 8601 with timezone, parallel source list lengths,
generated_at >= data_cutoff_at, source_published_at[i] <= data_cutoff_at.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
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

# Canonical JSON schema for external documentation / validation tooling
CLAUDE_SHADOW_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Claude Shadow Output V1",
    "type": "object",
    "required": [
        "schema_version", "signal_direction", "evidence_strength",
        "guidance_change", "estimate_revision_direction", "margin_trend",
        "earnings_quality", "capital_allocation_quality", "thesis_risks",
        "catalyst_strength", "uncertainty", "source_ids", "source_published_at",
        "model_id", "prompt_version", "generated_at", "data_cutoff_at",
        "order_creation_blocked",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": "shadow_v1"},
        "signal_direction": {"type": "string", "enum": sorted(_SIGNAL_DIRECTIONS)},
        "evidence_strength": {"type": "string", "enum": sorted(_EVIDENCE_STRENGTHS)},
        "guidance_change": {"type": "string", "enum": sorted(_GUIDANCE_CHANGES)},
        "estimate_revision_direction": {"type": "string", "enum": sorted(_REVISION_DIRECTIONS)},
        "margin_trend": {"type": "string", "enum": sorted(_TRENDS)},
        "earnings_quality": {"type": "string", "enum": sorted(_QUALITIES)},
        "capital_allocation_quality": {"type": "string", "enum": sorted(_QUALITIES)},
        "thesis_risks": {"type": "array", "items": {"type": "string"}},
        "catalyst_strength": {"type": "string", "enum": sorted(_STRENGTHS)},
        "uncertainty": {"type": "string", "enum": sorted(_UNCERTAINTIES)},
        "source_ids": {"type": "array", "items": {"type": "string"}},
        "source_published_at": {
            "type": "array",
            "items": {"type": "string", "description": "ISO 8601 with timezone"},
        },
        "model_id": {"type": ["string", "null"]},
        "prompt_version": {"type": ["string", "null"]},
        "generated_at": {"type": ["string", "null"], "description": "ISO 8601 with timezone"},
        "data_cutoff_at": {"type": ["string", "null"], "description": "ISO 8601 with timezone"},
        "order_creation_blocked": {"type": "boolean", "const": True},
        "content_hash": {"type": ["string", "null"]},
        "unavailable_reason": {"type": ["string", "null"]},
        "provenance_valid": {"type": "boolean"},
        "provenance_failures": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _has_tz(dt: datetime) -> bool:
    """Return True if datetime has timezone info."""
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None


def _parse_iso_dt(value: str) -> datetime | None:
    """
    Parse an ISO 8601 datetime string. Returns None if unparseable.
    Supports 'Z' suffix as UTC.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
        "provenance_valid": False,
        "provenance_failures": [],
    }


def validate_claude_shadow_output(raw: Any) -> dict[str, Any]:
    """
    Validate and normalize Claude shadow output against the V2 schema.

    Provenance validation:
      - generated_at and data_cutoff_at must be ISO 8601 with timezone
      - source_ids and source_published_at must be parallel lists (same length)
      - each source_published_at[i] must be ISO 8601 with timezone
      - generated_at >= data_cutoff_at (data must precede generation)
      - each source_published_at[i] <= data_cutoff_at (sources must precede cutoff)

    Any missing, invalid, or unknown field → "unavailable".
    Does NOT use regex as primary validation — uses explicit enum membership.
    order_creation_blocked is always True regardless of input.
    provenance_valid and provenance_failures always present in output.
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

    # ------------------------------------------------------------------
    # Provenance validation
    # ------------------------------------------------------------------
    failures: list[str] = []

    generated_at_str = validated["generated_at"]
    data_cutoff_str = validated["data_cutoff_at"]
    source_ids = validated["source_ids"]
    source_published_at = validated["source_published_at"]

    # Parse timestamps
    gen_dt = _parse_iso_dt(generated_at_str) if generated_at_str else None
    cutoff_dt = _parse_iso_dt(data_cutoff_str) if data_cutoff_str else None

    if generated_at_str is None:
        failures.append("generated_at: missing")
    elif gen_dt is None:
        failures.append(f"generated_at: unparseable ISO 8601 ({generated_at_str!r})")
    elif not _has_tz(gen_dt):
        failures.append(f"generated_at: no timezone ({generated_at_str!r})")

    if data_cutoff_str is None:
        failures.append("data_cutoff_at: missing")
    elif cutoff_dt is None:
        failures.append(f"data_cutoff_at: unparseable ISO 8601 ({data_cutoff_str!r})")
    elif not _has_tz(cutoff_dt):
        failures.append(f"data_cutoff_at: no timezone ({data_cutoff_str!r})")

    # generated_at >= data_cutoff_at
    if gen_dt is not None and cutoff_dt is not None and _has_tz(gen_dt) and _has_tz(cutoff_dt):
        gen_utc = gen_dt.astimezone(timezone.utc)
        cutoff_utc = cutoff_dt.astimezone(timezone.utc)
        if gen_utc < cutoff_utc:
            failures.append(
                f"generated_at ({generated_at_str}) is before data_cutoff_at ({data_cutoff_str})"
            )

    # Parallel list lengths
    if len(source_ids) != len(source_published_at):
        failures.append(
            f"source_ids length ({len(source_ids)}) != "
            f"source_published_at length ({len(source_published_at)})"
        )

    # Each source_published_at[i] must be timezone-aware and <= data_cutoff_at
    for i, pub_str in enumerate(source_published_at):
        pub_dt = _parse_iso_dt(pub_str)
        if pub_dt is None:
            failures.append(f"source_published_at[{i}]: unparseable ({pub_str!r})")
            continue
        if not _has_tz(pub_dt):
            failures.append(f"source_published_at[{i}]: no timezone ({pub_str!r})")
            continue
        if cutoff_dt is not None and _has_tz(cutoff_dt):
            pub_utc = pub_dt.astimezone(timezone.utc)
            cutoff_utc = cutoff_dt.astimezone(timezone.utc)
            if pub_utc > cutoff_utc:
                failures.append(
                    f"source_published_at[{i}] ({pub_str}) is after data_cutoff_at ({data_cutoff_str})"
                )

    validated["provenance_valid"] = len(failures) == 0
    validated["provenance_failures"] = failures

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
