"""
Model, portfolio, and execution versioning.

Three separate YAML registries, each with its own hash:
  config/model_versions.yml     → model_config_hash
  config/portfolio_versions.yml → portfolio_config_hash
  config/execution_versions.yml → execution_config_hash

A change in portfolio rules (stop-loss, pyramid) requires a new portfolio_version
but does NOT require a new model_version.

A change in signal weights requires a new model_version but NOT a new portfolio_version.

A change in fill methodology requires a new execution_version only.

ModelVersionMismatchError is raised when:
  1. A stored hash does not match the computed hash, OR
  2. Key config values in the YAML do not match the live code.

CLI usage:
  python3 -m modules.versioning --compute-hash
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).parent.parent / "config"
_MODEL_VERSIONS_FILE      = _CONFIG_DIR / "model_versions.yml"
_PORTFOLIO_VERSIONS_FILE  = _CONFIG_DIR / "portfolio_versions.yml"
_EXECUTION_VERSIONS_FILE  = _CONFIG_DIR / "execution_versions.yml"

# ---------------------------------------------------------------------------
# Version constants — single source of truth for all callers
# ---------------------------------------------------------------------------

MODEL_VERSION     = "quant_baseline_v1"
PORTFOLIO_VERSION = "risk_parity_pyramid_v1"
EXECUTION_VERSION = "exec_next_open_v1"

SENTIMENT_PROMPT_VERSION = "sentiment_v1"
EARNINGS_PROMPT_VERSION  = "earnings_v1"
WEEKLY_PROMPT_VERSION    = "weekly_v1"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ModelVersionMismatchError(Exception):
    """
    Raised when a frozen config diverges from the live code, or when a hash
    does not match. Requires an explicit version bump to resolve.
    """


# ---------------------------------------------------------------------------
# YAML loading
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


def get_execution_registry() -> dict:
    return _load_yaml(_EXECUTION_VERSIONS_FILE)


def get_model_config(model_version: str = MODEL_VERSION) -> dict:
    registry = get_model_registry()
    entry = registry.get(model_version)
    if entry is None:
        raise KeyError(f"Model version '{model_version}' not found in {_MODEL_VERSIONS_FILE}")
    return entry.get("config", {})


def get_portfolio_config(portfolio_version: str = PORTFOLIO_VERSION) -> dict:
    registry = get_portfolio_registry()
    entry = registry.get(portfolio_version)
    if entry is None:
        raise KeyError(f"Portfolio version '{portfolio_version}' not found in {_PORTFOLIO_VERSIONS_FILE}")
    return entry.get("config", {})


def get_execution_config(execution_version: str = EXECUTION_VERSION) -> dict:
    registry = get_execution_registry()
    entry = registry.get(execution_version)
    if entry is None:
        raise KeyError(f"Execution version '{execution_version}' not found in {_EXECUTION_VERSIONS_FILE}")
    return entry.get("config", {})


# ---------------------------------------------------------------------------
# Hash helpers — for embedding in ledger records
# ---------------------------------------------------------------------------

def get_model_config_hash(model_version: str = MODEL_VERSION) -> str:
    """Return the stored model_config_hash (not recomputed at runtime)."""
    registry = get_model_registry()
    return registry.get(model_version, {}).get("model_config_hash", "")


def get_portfolio_config_hash(portfolio_version: str = PORTFOLIO_VERSION) -> str:
    registry = get_portfolio_registry()
    return registry.get(portfolio_version, {}).get("portfolio_config_hash", "")


def get_execution_config_hash(execution_version: str = EXECUTION_VERSION) -> str:
    registry = get_execution_registry()
    return registry.get(execution_version, {}).get("execution_config_hash", "")


# ---------------------------------------------------------------------------
# Canonical JSON and hash computation
# ---------------------------------------------------------------------------

def _canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, Python None → JSON null."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _compute_entry_hash(registry: dict, version: str, hash_key: str) -> str:
    """
    Compute SHA-256 of all fields in registry[version] except hash_key.
    Returns 64-char hex string.
    """
    entry = registry.get(version)
    if entry is None:
        raise KeyError(f"Version '{version}' not found in registry")
    entry_copy = copy.deepcopy(entry)
    entry_copy.pop(hash_key, None)
    return hashlib.sha256(_canonical_json(entry_copy).encode("utf-8")).hexdigest()


def compute_model_config_hash(model_version: str = MODEL_VERSION) -> str:
    return _compute_entry_hash(get_model_registry(), model_version, "model_config_hash")


def compute_portfolio_config_hash(portfolio_version: str = PORTFOLIO_VERSION) -> str:
    return _compute_entry_hash(get_portfolio_registry(), portfolio_version, "portfolio_config_hash")


def compute_execution_config_hash(execution_version: str = EXECUTION_VERSION) -> str:
    return _compute_entry_hash(get_execution_registry(), execution_version, "execution_config_hash")


# ---------------------------------------------------------------------------
# Hash validation
# ---------------------------------------------------------------------------

def _validate_hash(stored: str, computed: str, version: str, hash_key: str) -> None:
    if stored in ("", "PLACEHOLDER", None, "n/a"):
        raise ModelVersionMismatchError(
            f"{hash_key} for '{version}' is not set. "
            f"Run: python3 -m modules.versioning --compute-hash"
        )
    if stored != computed:
        raise ModelVersionMismatchError(
            f"{hash_key} mismatch for '{version}'.\n"
            f"  Stored:   {stored}\n"
            f"  Computed: {computed}\n"
            "The config was modified without updating the hash, or the hash was corrupted.\n"
            "Run: python3 -m modules.versioning --compute-hash"
        )


def validate_model_config_hash(model_version: str = MODEL_VERSION) -> None:
    registry = get_model_registry()
    stored = registry.get(model_version, {}).get("model_config_hash", "")
    computed = compute_model_config_hash(model_version)
    _validate_hash(stored, computed, model_version, "model_config_hash")


def validate_portfolio_config_hash(portfolio_version: str = PORTFOLIO_VERSION) -> None:
    registry = get_portfolio_registry()
    stored = registry.get(portfolio_version, {}).get("portfolio_config_hash", "")
    computed = compute_portfolio_config_hash(portfolio_version)
    _validate_hash(stored, computed, portfolio_version, "portfolio_config_hash")


def validate_execution_config_hash(execution_version: str = EXECUTION_VERSION) -> None:
    registry = get_execution_registry()
    stored = registry.get(execution_version, {}).get("execution_config_hash", "")
    computed = compute_execution_config_hash(execution_version)
    _validate_hash(stored, computed, execution_version, "execution_config_hash")


# ---------------------------------------------------------------------------
# Code cross-validation: YAML config vs live scoring.py
# ---------------------------------------------------------------------------

def _live_strategies() -> dict:
    from modules.scoring import STRATEGIES  # noqa: PLC0415
    return STRATEGIES


def _live_regime_weights() -> dict:
    from modules.scoring import REGIME_WEIGHTS  # noqa: PLC0415
    return REGIME_WEIGHTS


def validate_model_strategies_match_code(model_version: str = MODEL_VERSION) -> None:
    """
    Verify that base_weights and score_column for each strategy in model_versions.yml
    match the live scoring.STRATEGIES. Only model-side fields are checked here.
    """
    config = get_model_config(model_version)
    yaml_strategies = config.get("strategies", {})
    live_strategies = _live_strategies()
    errors: list[str] = []

    for name, yaml_cfg in yaml_strategies.items():
        live_cfg = live_strategies.get(name)
        if live_cfg is None:
            errors.append(f"'{name}' in YAML but missing from scoring.STRATEGIES")
            continue
        # Check score_column
        if yaml_cfg.get("score_column") != live_cfg.get("score_column"):
            errors.append(
                f"{name}.score_column: YAML={yaml_cfg.get('score_column')} "
                f"vs code={live_cfg.get('score_column')}"
            )
        # Check base_weights
        yaml_w = yaml_cfg.get("base_weights", {})
        live_w = live_cfg.get("weights", {})
        for factor in ("momentum", "quality", "value", "sentiment"):
            y = round(float(yaml_w.get(factor, -1)), 4)
            l = round(float(live_w.get(factor, -1)), 4)
            if y != l:
                errors.append(f"{name}.base_weights.{factor}: YAML={y} vs code={l}")

    for name in live_strategies:
        if name not in yaml_strategies:
            errors.append(f"'{name}' in scoring.STRATEGIES but missing from YAML")

    if errors:
        raise ModelVersionMismatchError(
            f"Model '{model_version}' strategies do not match live code "
            f"({len(errors)} error(s)):\n"
            + "\n".join(f"  • {e}" for e in errors)
            + "\n\nUpdate YAML to match code, or bump model_version."
        )


def validate_regime_weights_match_code(model_version: str = MODEL_VERSION) -> None:
    config = get_model_config(model_version)
    yaml_rw = config.get("regime_weights", {})
    live_rw = _live_regime_weights()
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


def validate_portfolio_strategies_match_code(portfolio_version: str = PORTFOLIO_VERSION) -> None:
    """
    Verify that portfolio-side strategy params (stop_loss, trailing_stop, etc.)
    in portfolio_versions.yml match the live scoring.STRATEGIES.
    These live in scoring.STRATEGIES pending a future code refactor that
    separates signal config from portfolio config.
    """
    config = get_portfolio_config(portfolio_version)
    yaml_strategies = config.get("strategies", {})
    live_strategies = _live_strategies()
    errors: list[str] = []

    portfolio_fields = [
        ("stop_loss",              "stop_loss"),
        ("trailing_stop",          "trailing_stop"),
        ("max_positions",          "max_positions"),
        ("max_position_weight",    "max_position_weight"),
        ("max_new_buys_per_week",  "max_new_buys_per_week"),
        ("buy_top_n",              "buy_top_n"),
        ("min_score_percentile",   "min_score_percentile"),
        ("sell_rank_threshold",    "sell_rank_threshold"),
        ("buyback_cooldown_days",  "buyback_cooldown_days"),
    ]

    for name, yaml_cfg in yaml_strategies.items():
        live_cfg = live_strategies.get(name)
        if live_cfg is None:
            errors.append(f"'{name}' in portfolio YAML but missing from scoring.STRATEGIES")
            continue
        for yaml_key, code_key in portfolio_fields:
            y = yaml_cfg.get(yaml_key)
            l = live_cfg.get(code_key)
            if y is not None and l is not None and round(float(y), 6) != round(float(l), 6):
                errors.append(f"{name}.{yaml_key}: YAML={y} vs code={l}")
        # exposure per regime
        for regime, y_val in yaml_cfg.get("exposure", {}).items():
            l_val = live_cfg.get("exposure", {}).get(regime)
            if l_val is not None and round(float(y_val), 4) != round(float(l_val), 4):
                errors.append(f"{name}.exposure.{regime}: YAML={y_val} vs code={l_val}")

    if errors:
        raise ModelVersionMismatchError(
            f"Portfolio '{portfolio_version}' strategies do not match live code "
            f"({len(errors)} error(s)):\n"
            + "\n".join(f"  • {e}" for e in errors)
            + "\n\nUpdate YAML to match code, or bump portfolio_version."
        )


def validate_execution_constants_match_code(execution_version: str = EXECUTION_VERSION) -> None:
    """
    Verify that commission/fee constants in execution_versions.yml match
    the live modules/portfolio.py estimate_trade_cost() constants.
    """
    from modules.portfolio import estimate_trade_cost  # noqa: PLC0415

    config = get_execution_config(execution_version)
    costs = config.get("transaction_costs", {})

    yaml_commission = float(costs.get("commission_rate", -1))
    yaml_sec_fee    = float(costs.get("sec_fee_rate_on_sells", -1))

    # Probe the live function with known values to infer the constants.
    # estimate_trade_cost(value, "BUY")  = value * 0.001
    # estimate_trade_cost(value, "SELL") = value * 0.001 + value * 0.00003
    probe = 10_000.0
    live_buy_cost  = estimate_trade_cost(probe, "BUY")
    live_sell_cost = estimate_trade_cost(probe, "SELL")

    inferred_commission = live_buy_cost / probe
    inferred_sec_fee    = (live_sell_cost - live_buy_cost) / probe

    errors: list[str] = []
    if abs(yaml_commission - inferred_commission) > 1e-7:
        errors.append(
            f"commission_rate: YAML={yaml_commission} vs inferred={inferred_commission:.6f}"
        )
    if abs(yaml_sec_fee - inferred_sec_fee) > 1e-7:
        errors.append(
            f"sec_fee_rate_on_sells: YAML={yaml_sec_fee} vs inferred={inferred_sec_fee:.6f}"
        )

    if errors:
        raise ModelVersionMismatchError(
            f"Execution '{execution_version}' constants do not match portfolio.estimate_trade_cost:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# validate_all
# ---------------------------------------------------------------------------

def validate_all(
    model_version:     str = MODEL_VERSION,
    portfolio_version: str = PORTFOLIO_VERSION,
    execution_version: str = EXECUTION_VERSION,
) -> None:
    """
    Full validation: hashes + code cross-validation for all three configs.
    Call at system startup to verify all frozen configs are intact.
    """
    validate_model_config_hash(model_version)
    validate_portfolio_config_hash(portfolio_version)
    validate_execution_config_hash(execution_version)
    validate_model_strategies_match_code(model_version)
    validate_regime_weights_match_code(model_version)
    validate_portfolio_strategies_match_code(portfolio_version)
    validate_execution_constants_match_code(execution_version)


# ---------------------------------------------------------------------------
# CLI: --compute-hash  (writes to all three YAML files)
# ---------------------------------------------------------------------------

def _write_hash_to_file(path: Path, version_name: str, hash_key: str, new_hash: str) -> None:
    """
    Replace hash_key within version_name's YAML block only.
    Uses the version name to scope the replacement, so a deprecated version's
    hash key in the same file is never accidentally modified.
    """
    raw = path.read_text()
    lines = raw.splitlines(keepends=True)

    # Find the line where this version's top-level block starts.
    version_line = -1
    for i, line in enumerate(lines):
        if re.match(rf'^{re.escape(version_name)}\s*:', line):
            version_line = i
            break
    if version_line < 0:
        raise KeyError(f"Version '{version_name}' not found in {path}")

    # Find where the block ends: the next non-indented, non-empty, non-comment line.
    block_end = len(lines)
    for i in range(version_line + 1, len(lines)):
        line = lines[i]
        if line and line[0] not in (" ", "\t", "#", "\n", "\r"):
            block_end = i
            break

    prefix  = "".join(lines[:version_line])
    block   = "".join(lines[version_line:block_end])
    suffix  = "".join(lines[block_end:])

    # Replace the hash value within the block.
    # Use re.search to check which form is present before substituting,
    # so identical replacement text can't cause the fallback branch to also run.
    quoted_pattern   = rf'({re.escape(hash_key)}:\s*)"[^"]*"'
    unquoted_pattern = rf'({re.escape(hash_key)}:\s*)([^\n\r"]+)'

    if re.search(quoted_pattern, block):
        new_block = re.sub(quoted_pattern, rf'\g<1>"{new_hash}"', block, count=1)
    elif re.search(unquoted_pattern, block):
        new_block = re.sub(unquoted_pattern, rf'\g<1>"{new_hash}"', block, count=1)
    else:
        raise ValueError(f"{hash_key} not found in {version_name} block of {path}")

    path.write_text(prefix + new_block + suffix)


if __name__ == "__main__":
    if "--compute-hash" not in sys.argv:
        print("Usage: python3 -m modules.versioning --compute-hash")
        sys.exit(0)

    print("Computing and writing config hashes...\n")

    try:
        h_model = compute_model_config_hash(MODEL_VERSION)
        _write_hash_to_file(_MODEL_VERSIONS_FILE, MODEL_VERSION, "model_config_hash", h_model)
        print(f"model_config_hash ({MODEL_VERSION}): {h_model}")

        h_portfolio = compute_portfolio_config_hash(PORTFOLIO_VERSION)
        _write_hash_to_file(_PORTFOLIO_VERSIONS_FILE, PORTFOLIO_VERSION, "portfolio_config_hash", h_portfolio)
        print(f"portfolio_config_hash ({PORTFOLIO_VERSION}): {h_portfolio}")

        h_execution = compute_execution_config_hash(EXECUTION_VERSION)
        _write_hash_to_file(_EXECUTION_VERSIONS_FILE, EXECUTION_VERSION, "execution_config_hash", h_execution)
        print(f"execution_config_hash ({EXECUTION_VERSION}): {h_execution}")

        print("\nVerifying all hashes...")
        validate_model_config_hash(MODEL_VERSION)
        validate_portfolio_config_hash(PORTFOLIO_VERSION)
        validate_execution_config_hash(EXECUTION_VERSION)
        print("Hash verification: OK")

        print("\nCross-validating against live code...")
        validate_model_strategies_match_code(MODEL_VERSION)
        validate_regime_weights_match_code(MODEL_VERSION)
        validate_portfolio_strategies_match_code(PORTFOLIO_VERSION)
        validate_execution_constants_match_code(EXECUTION_VERSION)
        print("Code cross-validation: OK\n")
        print("All config hashes written and verified.")

    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
