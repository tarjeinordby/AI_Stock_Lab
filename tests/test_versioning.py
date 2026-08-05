"""
Tests for modules/versioning.py

Verifies:
  - config_hash is deterministic and validates correctly
  - A YAML change without hash update triggers ModelVersionMismatchError
  - A code change (strategy weight change) without new model_version triggers error
  - validate_all() passes for the current frozen config
  - Version constants are set correctly
"""

import copy
import hashlib
import json
from unittest.mock import patch

import pytest
import yaml

from modules.versioning import (
    MODEL_VERSION,
    PORTFOLIO_VERSION,
    EXECUTION_VERSION,
    ModelVersionMismatchError,
    compute_config_hash,
    get_model_config,
    get_model_registry,
    validate_all,
    validate_config_hash,
    validate_regime_weights_match_code,
    validate_strategies_match_code,
    _canonical_json,
    _MODEL_VERSIONS_FILE,
)


# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

class TestVersionConstants:
    def test_model_version_is_baseline(self):
        assert MODEL_VERSION == "quant_baseline_v1"

    def test_portfolio_version_is_descriptive(self):
        assert "pyramid" in PORTFOLIO_VERSION or "managed" in PORTFOLIO_VERSION or "risk" in PORTFOLIO_VERSION

    def test_execution_version_is_next_open(self):
        assert "next_open" in EXECUTION_VERSION


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_model_registry_loads(self):
        registry = get_model_registry()
        assert "quant_baseline_v1" in registry

    def test_model_config_has_required_sections(self):
        config = get_model_config("quant_baseline_v1")
        required = [
            "strategies", "regime_weights", "quality_filter",
            "momentum_factor", "quality_factor", "value_factor",
            "sentiment_factor", "score_adjustments", "portfolio_execution",
        ]
        for section in required:
            assert section in config, f"Missing config section: {section}"

    def test_all_six_strategies_present(self):
        config = get_model_config("quant_baseline_v1")
        strategies = config["strategies"]
        expected = {
            "Max_Return_AI", "Quality_Momentum_AI", "Balanced_AI",
            "Low_Risk_AI", "MegaCap_AI", "AI_Sentiment_AI"
        }
        assert set(strategies.keys()) == expected

    def test_unknown_model_version_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_model_config("nonexistent_v99")


# ---------------------------------------------------------------------------
# config_hash — determinism and validation
# ---------------------------------------------------------------------------

class TestConfigHash:
    def test_hash_is_deterministic(self):
        h1 = compute_config_hash("quant_baseline_v1")
        h2 = compute_config_hash("quant_baseline_v1")
        assert h1 == h2

    def test_hash_is_64_hex_chars(self):
        h = compute_config_hash("quant_baseline_v1")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_stored_hash_matches_computed(self):
        """validate_config_hash must pass for the current YAML."""
        # This is the core "is the frozen model intact?" check.
        validate_config_hash("quant_baseline_v1")

    def test_tampered_yaml_hash_fails(self, tmp_path, monkeypatch):
        """
        Simulate someone editing the YAML config_hash field to a wrong value.
        Must raise ModelVersionMismatchError.
        """
        registry = get_model_registry()
        tampered = copy.deepcopy(registry)
        tampered["quant_baseline_v1"]["config_hash"] = "deadbeef" * 8

        monkeypatch.setattr(
            "modules.versioning.get_model_registry",
            lambda: tampered
        )
        with pytest.raises(ModelVersionMismatchError, match="config_hash mismatch"):
            validate_config_hash("quant_baseline_v1")

    def test_placeholder_hash_fails(self, monkeypatch):
        """
        A PLACEHOLDER config_hash (not yet computed) must fail.
        """
        registry = get_model_registry()
        incomplete = copy.deepcopy(registry)
        incomplete["quant_baseline_v1"]["config_hash"] = "PLACEHOLDER"

        monkeypatch.setattr(
            "modules.versioning.get_model_registry",
            lambda: incomplete
        )
        with pytest.raises(ModelVersionMismatchError, match="not set"):
            validate_config_hash("quant_baseline_v1")

    def test_modifying_weight_changes_hash(self):
        """
        Changing a strategy weight in the YAML content must produce a different hash.
        This proves the hash actually covers config values.
        """
        registry = get_model_registry()
        entry = registry["quant_baseline_v1"]

        # Compute hash of original
        entry_copy = copy.deepcopy(entry)
        entry_copy.pop("config_hash", None)
        original_hash = hashlib.sha256(
            _canonical_json(entry_copy).encode()
        ).hexdigest()

        # Mutate a weight value
        entry_copy["config"]["strategies"]["Max_Return_AI"]["base_weights"]["momentum"] = 0.99
        mutated_hash = hashlib.sha256(
            _canonical_json(entry_copy).encode()
        ).hexdigest()

        assert original_hash != mutated_hash, (
            "Mutating a strategy weight must change the config_hash"
        )


# ---------------------------------------------------------------------------
# Code cross-validation
# ---------------------------------------------------------------------------

class TestCodeCrossValidation:
    def test_strategies_match_live_code(self):
        """The frozen YAML must match the current scoring.STRATEGIES exactly."""
        validate_strategies_match_code("quant_baseline_v1")

    def test_regime_weights_match_live_code(self):
        """The frozen YAML must match the current scoring.REGIME_WEIGHTS exactly."""
        validate_regime_weights_match_code("quant_baseline_v1")

    def test_changed_weight_in_code_triggers_error(self, monkeypatch):
        """
        Simulate a developer changing a strategy weight in scoring.py
        without creating a new model version.
        Must raise ModelVersionMismatchError — not silently pass.
        """
        from modules import scoring

        # Save original and mutate
        original_strategies = copy.deepcopy(scoring.STRATEGIES)
        mutated = copy.deepcopy(original_strategies)
        mutated["Max_Return_AI"]["weights"]["momentum"] = 0.99  # was 0.45

        with patch.object(scoring, "STRATEGIES", mutated):
            with pytest.raises(ModelVersionMismatchError) as exc_info:
                validate_strategies_match_code("quant_baseline_v1")

        assert "Max_Return_AI" in str(exc_info.value)
        assert "momentum" in str(exc_info.value)
        assert "0.99" in str(exc_info.value)

    def test_changed_regime_weight_in_code_triggers_error(self, monkeypatch):
        """
        Simulate a developer changing a regime weight in scoring.py.
        Must raise ModelVersionMismatchError.
        """
        from modules import scoring

        original_regime = copy.deepcopy(scoring.REGIME_WEIGHTS)
        mutated = copy.deepcopy(original_regime)
        mutated["bullish"]["momentum"] = 0.99  # was 0.35

        with patch.object(scoring, "REGIME_WEIGHTS", mutated):
            with pytest.raises(ModelVersionMismatchError) as exc_info:
                validate_regime_weights_match_code("quant_baseline_v1")

        assert "bullish" in str(exc_info.value)
        assert "momentum" in str(exc_info.value)

    def test_missing_strategy_in_code_triggers_error(self):
        """
        If a strategy in YAML is removed from code, validation must fail.
        """
        from modules import scoring

        mutated = {k: v for k, v in scoring.STRATEGIES.items() if k != "MegaCap_AI"}

        with patch.object(scoring, "STRATEGIES", mutated):
            with pytest.raises(ModelVersionMismatchError) as exc_info:
                validate_strategies_match_code("quant_baseline_v1")

        assert "MegaCap_AI" in str(exc_info.value)

    def test_extra_strategy_in_code_triggers_error(self):
        """
        If a new strategy is added to code without updating YAML and model_version,
        validation must fail.
        """
        from modules import scoring

        mutated = copy.deepcopy(scoring.STRATEGIES)
        mutated["New_Strategy_AI"] = {
            "description": "Undocumented new strategy",
            "score_column": "score_new",
            "weights": {"momentum": 0.25, "quality": 0.25, "value": 0.25, "sentiment": 0.25},
            "max_positions": 10, "max_position_weight": 0.10,
            "max_new_buys_per_week": 3, "buy_top_n": 20,
            "min_score_percentile": 0.70, "sell_rank_threshold": 100,
            "stop_loss": -0.15, "trailing_stop": -0.20,
            "buyback_cooldown_days": 7,
            "exposure": {"explosive": 0.80, "bullish": 0.75, "neutral": 0.50,
                         "defensive": 0.25, "unknown": 0.45},
        }

        with patch.object(scoring, "STRATEGIES", mutated):
            with pytest.raises(ModelVersionMismatchError) as exc_info:
                validate_strategies_match_code("quant_baseline_v1")

        assert "New_Strategy_AI" in str(exc_info.value)


# ---------------------------------------------------------------------------
# validate_all
# ---------------------------------------------------------------------------

class TestValidateAll:
    def test_validate_all_passes_for_current_state(self):
        """
        Full end-to-end validation. This test fails if:
          - YAML config_hash is wrong
          - Any strategy weight in code differs from YAML
          - Any regime weight in code differs from YAML
        """
        validate_all("quant_baseline_v1")

    def test_validate_all_fails_if_hash_and_code_both_wrong(self, monkeypatch):
        """
        Even if hash fails (short-circuits), the error is clear.
        """
        registry = get_model_registry()
        bad = copy.deepcopy(registry)
        bad["quant_baseline_v1"]["config_hash"] = "0" * 64

        monkeypatch.setattr("modules.versioning.get_model_registry", lambda: bad)
        with pytest.raises(ModelVersionMismatchError):
            validate_all("quant_baseline_v1")


# ---------------------------------------------------------------------------
# canonical_json determinism
# ---------------------------------------------------------------------------

class TestCanonicalJson:
    def test_key_order_does_not_affect_hash(self):
        d1 = {"b": 2, "a": 1, "c": {"z": 99, "y": 88}}
        d2 = {"a": 1, "c": {"y": 88, "z": 99}, "b": 2}
        assert _canonical_json(d1) == _canonical_json(d2)

    def test_float_equality_preserved(self):
        j = _canonical_json({"momentum": 0.45})
        assert "0.45" in j
