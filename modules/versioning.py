"""
Model and portfolio versioning.

Provides:
  - VERSION constants used throughout the system
  - Config loading from config/model_versions.yml and config/portfolio_versions.yml
  - config_hash validation: SHA-256 of all config fields except config_hash itself
  - Cross-validation of YAML config against live code (scoring.py)

ModelVersionMismatchError is raised when:
  1. The YAML config_hash field does not match the computed hash, OR
  2. Key config values in the YAML do not match the live code

Run as CLI to compute and update the hash:
  python3 -m modules.versioning --compute-hash

Usage in code:
  from modules.versioning import MODEL_VERSION, PORTFOLIO_VERSION, EXECUTION_VERSION
  from modules.versioning import get_model_config, validate_all
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).parent.parent / "config"
_MODEL_VERSIONS_FILE = _CONFIG_DIR / "model_versions.yml"
_PORTFOLIO_VERSIONS_FILE = _CONFIG_DIR / "portfolio_versions.yml"

# ---------------------------------------------------------------------------
# Version constants — single source of truth for all callers
# ---------------------------------------------------------------------------

MODEL_VERSION = "quant_baseline_v1"
PORTFOLIO_VERSION = "risk_parity_pyramid_v1"
EXECUTION_VERSION = "exec_next_open_v1"

# Prompt versions (per AI module)
SENTIMENT_PROMPT_VERSION = "sentiment_v1"
EARNINGS_PROMPT_VERSION = "earnings_v1"
WEEKLY_PROMPT_VERSION = "weekly_v1"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ModelVersionMismatchError(Exception):
    """
    Raised when the frozen model config in YAML diverges from the live code,
    or when config_hash does not match the computed hash.
    Requires explicit version bump (e.g. quant_baseline_v2) to resolve.
    """


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc


def get_model_registry() -> dict:
    return _load_yaml(_MODEL_VERSIONS_FILE)


def get_portfolio_registry() -> dict:
    return _load_yaml(_PORTFOLIO_VERSIONS_FILE)


def get_model_config(model_version: str = MODEL_VERSION) -> dict:
    """Return the 'config' sub-dict for the given model version."""
    registry = get_model_registry()
    entry = registry.get(model_version)
    if entry is None:
        raise KeyError(f"Model version '{model_version}' not found in {_MODEL_VERSIONS_FILE}")
    return entry.get("config", {})


# ---------------------------------------------------------------------------
# config_hash computation
# ---------------------------------------------------------------------------

def _canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, None → null."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_config_hash(model_version: str = MODEL_VERSION) -> str:
    """
    Compute SHA-256 of all fields in the model version entry EXCEPT config_hash.
    Returns hex string (64 chars).
    """
    registry = get_model_registry()
    entry = registry.get(model_version)
    if entry is None:
        raise KeyError(f"Model version '{model_version}' not found")

    # Deep copy and remove config_hash before hashing
    entry_copy = copy.deepcopy(entry)
    entry_copy.pop("config_hash", None)

    canonical = _canonical_json(entry_copy)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_config_hash(model_version: str = MODEL_VERSION) -> None:
    """
    Verify that config_hash in the YAML matches the computed hash.
    Raises ModelVersionMismatchError on mismatch.
    """
    registry = get_model_registry()
    entry = registry.get(model_version)
    if entry is None:
        raise KeyError(f"Model version '{model_version}' not found")

    stored_hash = entry.get("config_hash", "")
    if stored_hash in ("", "PLACEHOLDER", None):
        raise ModelVersionMismatchError(
            f"config_hash for '{model_version}' is not set. "
            f"Run: python3 -m modules.versioning --compute-hash"
        )

    computed = compute_config_hash(model_version)
    if stored_hash != computed:
        raise ModelVersionMismatchError(
            f"config_hash mismatch for '{model_version}'.\n"
            f"  Stored:   {stored_hash}\n"
            f"  Computed: {computed}\n"
            f"The YAML was modified without updating config_hash, or "
            f"config_hash was corrupted. Run --compute-hash to update."
        )


# ---------------------------------------------------------------------------
# Cross-validation: YAML config vs live scoring code
# ---------------------------------------------------------------------------

def _get_live_strategies() -> dict:
    """Import STRATEGIES from scoring.py without side effects."""
    from modules.scoring import STRATEGIES  # noqa: PLC0415
    return STRATEGIES


def _get_live_regime_weights() -> dict:
    from modules.scoring import REGIME_WEIGHTS  # noqa: PLC0415
    return REGIME_WEIGHTS


def validate_strategies_match_code(model_version: str = MODEL_VERSION) -> None:
    """
    Cross-validate per-strategy config in YAML against live modules/scoring.py.
    Checks base_weights, stop_loss, trailing_stop, and key sizing parameters.
    Raises ModelVersionMismatchError on any mismatch.
    """
    config = get_model_config(model_version)
    yaml_strategies = config.get("strategies", {})
    live_strategies = _get_live_strategies()

    errors: list[str] = []

    for strat_name, yaml_cfg in yaml_strategies.items():
        live_cfg = live_strategies.get(strat_name)
        if live_cfg is None:
            errors.append(f"Strategy '{strat_name}' in YAML but missing from scoring.STRATEGIES")
            continue

        # Check base_weights
        yaml_weights = yaml_cfg.get("base_weights", {})
        live_weights = live_cfg.get("weights", {})
        for factor in ("momentum", "quality", "value", "sentiment"):
            y = round(float(yaml_weights.get(factor, -1)), 4)
            l = round(float(live_weights.get(factor, -1)), 4)
            if y != l:
                errors.append(
                    f"{strat_name}.base_weights.{factor}: "
                    f"YAML={y} vs code={l}"
                )

        # Check scalar thresholds
        scalar_fields = [
            ("stop_loss", "stop_loss"),
            ("trailing_stop", "trailing_stop"),
            ("max_positions", "max_positions"),
            ("max_position_weight", "max_position_weight"),
            ("max_new_buys_per_week", "max_new_buys_per_week"),
            ("buy_top_n", "buy_top_n"),
            ("min_score_percentile", "min_score_percentile"),
            ("sell_rank_threshold", "sell_rank_threshold"),
            ("buyback_cooldown_days", "buyback_cooldown_days"),
        ]
        for yaml_key, code_key in scalar_fields:
            y = yaml_cfg.get(yaml_key)
            l = live_cfg.get(code_key)
            if y is not None and l is not None and round(float(y), 6) != round(float(l), 6):
                errors.append(
                    f"{strat_name}.{yaml_key}: YAML={y} vs code={l}"
                )

        # Check exposure by regime
        yaml_exposure = yaml_cfg.get("exposure", {})
        live_exposure = live_cfg.get("exposure", {})
        for regime, y_val in yaml_exposure.items():
            l_val = live_exposure.get(regime)
            if l_val is not None and round(float(y_val), 4) != round(float(l_val), 4):
                errors.append(
                    f"{strat_name}.exposure.{regime}: YAML={y_val} vs code={l_val}"
                )

    # Check for strategies in code not in YAML
    for strat_name in live_strategies:
        if strat_name not in yaml_strategies:
            errors.append(
                f"Strategy '{strat_name}' in scoring.STRATEGIES but missing from YAML"
            )

    if errors:
        raise ModelVersionMismatchError(
            f"Model '{model_version}' config does not match live code "
            f"({len(errors)} mismatch(es)):\n"
            + "\n".join(f"  • {e}" for e in errors)
            + "\n\nEither update the YAML to match the code, or bump the model version."
        )


def validate_regime_weights_match_code(model_version: str = MODEL_VERSION) -> None:
    """Cross-validate regime_weights in YAML against scoring.REGIME_WEIGHTS."""
    config = get_model_config(model_version)
    yaml_rw = config.get("regime_weights", {})
    live_rw = _get_live_regime_weights()

    errors: list[str] = []
    for regime, yaml_weights in yaml_rw.items():
        live_weights = live_rw.get(regime)
        if live_weights is None:
            errors.append(f"Regime '{regime}' in YAML but missing from REGIME_WEIGHTS")
            continue
        for factor, y_val in yaml_weights.items():
            l_val = live_weights.get(factor)
            if l_val is not None and round(float(y_val), 4) != round(float(l_val), 4):
                errors.append(
                    f"regime_weights.{regime}.{factor}: YAML={y_val} vs code={l_val}"
                )

    if errors:
        raise ModelVersionMismatchError(
            f"Regime weights for '{model_version}' do not match scoring.REGIME_WEIGHTS:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )


def validate_all(model_version: str = MODEL_VERSION) -> None:
    """
    Full validation: config_hash + code cross-validation.
    Call this at system startup to verify the frozen model is intact.
    """
    validate_config_hash(model_version)
    validate_strategies_match_code(model_version)
    validate_regime_weights_match_code(model_version)


# ---------------------------------------------------------------------------
# CLI: --compute-hash
# ---------------------------------------------------------------------------

def _update_config_hash(model_version: str) -> str:
    """Compute hash and write it back into the YAML file. Returns the hash."""
    registry = get_model_registry()
    if model_version not in registry:
        raise KeyError(f"Model version '{model_version}' not found")

    new_hash = compute_config_hash(model_version)

    # Read raw YAML text and replace the PLACEHOLDER line
    raw = _MODEL_VERSIONS_FILE.read_text()
    import re  # noqa: PLC0415
    pattern = rf'(config_hash:\s*")[^"]*(")'
    replacement = rf'\g<1>{new_hash}\g<2>'
    updated = re.sub(pattern, replacement, raw, count=1)

    if updated == raw:
        # Try without quotes (PLACEHOLDER without quotes)
        pattern2 = r'(config_hash:\s*)PLACEHOLDER'
        updated = re.sub(pattern2, rf'\g<1>"{new_hash}"', raw, count=1)

    _MODEL_VERSIONS_FILE.write_text(updated)
    return new_hash


if __name__ == "__main__":
    if "--compute-hash" in sys.argv:
        version = MODEL_VERSION
        for arg in sys.argv[1:]:
            if not arg.startswith("--"):
                version = arg
                break
        try:
            new_hash = _update_config_hash(version)
            print(f"config_hash for '{version}' computed and written:")
            print(f"  {new_hash}")
            print(f"  File: {_MODEL_VERSIONS_FILE}")
            print()
            print("Verifying...")
            validate_config_hash(version)
            print("Hash verification: OK")
            validate_strategies_match_code(version)
            print("Code cross-validation: OK")
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Usage: python3 -m modules.versioning --compute-hash [version]")
        sys.exit(0)
