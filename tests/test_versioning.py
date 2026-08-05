"""
Tests for modules/versioning.py

Three separate config registries, each with its own hash:
  model_config_hash      (model_versions.yml)
  portfolio_config_hash  (portfolio_versions.yml)
  execution_config_hash  (execution_versions.yml)

Key invariants verified:
  - Each hash is deterministic and validates correctly
  - A change to one registry does not invalidate another registry's hash
  - YAML tamper detection works for all three hashes
  - Code cross-validation (YAML vs live scoring.py) catches divergence
  - validate_all() passes for the current frozen config state
"""

import copy
import hashlib
from unittest.mock import patch

import pytest
import yaml

from modules.versioning import (
    MODEL_VERSION,
    PORTFOLIO_VERSION,
    EXECUTION_VERSION,
    SENTIMENT_PROMPT_VERSION,
    EARNINGS_PROMPT_VERSION,
    WEEKLY_PROMPT_VERSION,
    ModelVersionMismatchError,
    # Compute
    compute_model_config_hash,
    compute_portfolio_config_hash,
    compute_execution_config_hash,
    # Validate hashes
    validate_model_config_hash,
    validate_portfolio_config_hash,
    validate_execution_config_hash,
    # Get config
    get_model_config,
    get_portfolio_config,
    get_execution_config,
    # Get registry
    get_model_registry,
    get_portfolio_registry,
    get_execution_registry,
    # Stored hash accessors
    get_model_config_hash,
    get_portfolio_config_hash,
    get_execution_config_hash,
    # Code cross-validation
    validate_model_strategies_match_code,
    validate_regime_weights_match_code,
    validate_portfolio_strategies_match_code,
    validate_execution_constants_match_code,
    # validate_all
    validate_all,
    # Helpers
    _canonical_json,
    _MODEL_VERSIONS_FILE,
    _PORTFOLIO_VERSIONS_FILE,
    _EXECUTION_VERSIONS_FILE,
)


# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

class TestVersionConstants:
    def test_model_version_is_baseline(self):
        assert MODEL_VERSION == "quant_baseline_v1"

    def test_portfolio_version_has_descriptive_name(self):
        assert "pyramid" in PORTFOLIO_VERSION or "risk" in PORTFOLIO_VERSION

    def test_execution_version_is_next_open(self):
        assert "next_open" in EXECUTION_VERSION

    def test_prompt_versions_set(self):
        assert SENTIMENT_PROMPT_VERSION == "sentiment_v1"
        assert EARNINGS_PROMPT_VERSION == "earnings_v1"
        assert WEEKLY_PROMPT_VERSION == "weekly_v1"


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_model_registry_loads(self):
        registry = get_model_registry()
        assert "quant_baseline_v1" in registry

    def test_portfolio_registry_loads(self):
        registry = get_portfolio_registry()
        assert "risk_parity_pyramid_v1" in registry

    def test_execution_registry_loads(self):
        registry = get_execution_registry()
        assert "exec_next_open_v1" in registry

    def test_model_config_has_required_sections(self):
        config = get_model_config("quant_baseline_v1")
        required = [
            "strategies", "regime_weights", "quality_filter",
            "momentum_factor", "quality_factor", "value_factor",
            "sentiment_factor", "score_adjustments",
        ]
        for section in required:
            assert section in config, f"Missing model config section: {section}"

    def test_model_config_does_not_contain_portfolio_fields(self):
        """Strict scope check: model config must not bleed into portfolio territory."""
        config = get_model_config("quant_baseline_v1")
        forbidden_in_model = [
            "stop_loss", "trailing_stop", "pyramid", "sector_cap",
            "max_positions", "drawdown_protection", "transaction_costs",
        ]
        for key in forbidden_in_model:
            assert key not in config, (
                f"'{key}' found in model config — it belongs in portfolio_versions.yml"
            )

    def test_portfolio_config_has_required_sections(self):
        config = get_portfolio_config("risk_parity_pyramid_v1")
        required = ["strategies", "general", "pyramid", "drawdown_protection",
                    "macro_portfolio_exposure"]
        for section in required:
            assert section in config, f"Missing portfolio config section: {section}"

    def test_execution_config_has_required_sections(self):
        config = get_execution_config("exec_next_open_v1")
        required = ["fail_closed", "transaction_costs", "gap_filters"]
        for section in required:
            assert section in config, f"Missing execution config section: {section}"

    def test_all_six_strategies_in_model(self):
        config = get_model_config("quant_baseline_v1")
        expected = {
            "Max_Return_AI", "Quality_Momentum_AI", "Balanced_AI",
            "Low_Risk_AI", "MegaCap_AI", "AI_Sentiment_AI"
        }
        assert set(config["strategies"].keys()) == expected

    def test_all_six_strategies_in_portfolio(self):
        config = get_portfolio_config("risk_parity_pyramid_v1")
        expected = {
            "Max_Return_AI", "Quality_Momentum_AI", "Balanced_AI",
            "Low_Risk_AI", "MegaCap_AI", "AI_Sentiment_AI"
        }
        assert set(config["strategies"].keys()) == expected

    def test_unknown_model_version_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_model_config("nonexistent_v99")

    def test_unknown_portfolio_version_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_portfolio_config("nonexistent_v99")

    def test_unknown_execution_version_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_execution_config("nonexistent_v99")


# ---------------------------------------------------------------------------
# model_config_hash
# ---------------------------------------------------------------------------

class TestModelConfigHash:
    def test_hash_is_deterministic(self):
        h1 = compute_model_config_hash("quant_baseline_v1")
        h2 = compute_model_config_hash("quant_baseline_v1")
        assert h1 == h2

    def test_hash_is_64_hex_chars(self):
        h = compute_model_config_hash("quant_baseline_v1")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_stored_hash_matches_computed(self):
        validate_model_config_hash("quant_baseline_v1")

    def test_tampered_model_hash_fails(self, monkeypatch):
        registry = get_model_registry()
        tampered = copy.deepcopy(registry)
        tampered["quant_baseline_v1"]["model_config_hash"] = "deadbeef" * 8
        monkeypatch.setattr("modules.versioning.get_model_registry", lambda: tampered)
        with pytest.raises(ModelVersionMismatchError, match="model_config_hash mismatch"):
            validate_model_config_hash("quant_baseline_v1")

    def test_placeholder_model_hash_fails(self, monkeypatch):
        registry = get_model_registry()
        incomplete = copy.deepcopy(registry)
        incomplete["quant_baseline_v1"]["model_config_hash"] = "PLACEHOLDER"
        monkeypatch.setattr("modules.versioning.get_model_registry", lambda: incomplete)
        with pytest.raises(ModelVersionMismatchError, match="not set"):
            validate_model_config_hash("quant_baseline_v1")

    def test_modifying_strategy_weight_changes_model_hash(self):
        registry = get_model_registry()
        entry = registry["quant_baseline_v1"]
        entry_copy = copy.deepcopy(entry)
        entry_copy.pop("model_config_hash", None)
        original_hash = hashlib.sha256(_canonical_json(entry_copy).encode()).hexdigest()

        entry_copy["config"]["strategies"]["Max_Return_AI"]["base_weights"]["momentum"] = 0.99
        mutated_hash = hashlib.sha256(_canonical_json(entry_copy).encode()).hexdigest()

        assert original_hash != mutated_hash

    def test_get_model_config_hash_returns_stored_value(self):
        stored = get_model_config_hash("quant_baseline_v1")
        computed = compute_model_config_hash("quant_baseline_v1")
        assert stored == computed


# ---------------------------------------------------------------------------
# portfolio_config_hash
# ---------------------------------------------------------------------------

class TestPortfolioConfigHash:
    def test_hash_is_deterministic(self):
        h1 = compute_portfolio_config_hash("risk_parity_pyramid_v1")
        h2 = compute_portfolio_config_hash("risk_parity_pyramid_v1")
        assert h1 == h2

    def test_hash_is_64_hex_chars(self):
        h = compute_portfolio_config_hash("risk_parity_pyramid_v1")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_stored_hash_matches_computed(self):
        validate_portfolio_config_hash("risk_parity_pyramid_v1")

    def test_tampered_portfolio_hash_fails(self, monkeypatch):
        registry = get_portfolio_registry()
        tampered = copy.deepcopy(registry)
        tampered["risk_parity_pyramid_v1"]["portfolio_config_hash"] = "cafebabe" * 8
        monkeypatch.setattr("modules.versioning.get_portfolio_registry", lambda: tampered)
        with pytest.raises(ModelVersionMismatchError, match="portfolio_config_hash mismatch"):
            validate_portfolio_config_hash("risk_parity_pyramid_v1")

    def test_placeholder_portfolio_hash_fails(self, monkeypatch):
        registry = get_portfolio_registry()
        incomplete = copy.deepcopy(registry)
        incomplete["risk_parity_pyramid_v1"]["portfolio_config_hash"] = "PLACEHOLDER"
        monkeypatch.setattr("modules.versioning.get_portfolio_registry", lambda: incomplete)
        with pytest.raises(ModelVersionMismatchError, match="not set"):
            validate_portfolio_config_hash("risk_parity_pyramid_v1")

    def test_get_portfolio_config_hash_returns_stored_value(self):
        stored = get_portfolio_config_hash("risk_parity_pyramid_v1")
        computed = compute_portfolio_config_hash("risk_parity_pyramid_v1")
        assert stored == computed


# ---------------------------------------------------------------------------
# execution_config_hash
# ---------------------------------------------------------------------------

class TestExecutionConfigHash:
    def test_hash_is_deterministic(self):
        h1 = compute_execution_config_hash("exec_next_open_v1")
        h2 = compute_execution_config_hash("exec_next_open_v1")
        assert h1 == h2

    def test_hash_is_64_hex_chars(self):
        h = compute_execution_config_hash("exec_next_open_v1")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_stored_hash_matches_computed(self):
        validate_execution_config_hash("exec_next_open_v1")

    def test_tampered_execution_hash_fails(self, monkeypatch):
        registry = get_execution_registry()
        tampered = copy.deepcopy(registry)
        tampered["exec_next_open_v1"]["execution_config_hash"] = "baddcafe" * 8
        monkeypatch.setattr("modules.versioning.get_execution_registry", lambda: tampered)
        with pytest.raises(ModelVersionMismatchError, match="execution_config_hash mismatch"):
            validate_execution_config_hash("exec_next_open_v1")

    def test_get_execution_config_hash_returns_stored_value(self):
        stored = get_execution_config_hash("exec_next_open_v1")
        computed = compute_execution_config_hash("exec_next_open_v1")
        assert stored == computed


# ---------------------------------------------------------------------------
# Cross-hash independence
# ---------------------------------------------------------------------------

class TestCrossHashIndependence:
    """Changing one registry must not invalidate the other registry's hash."""

    def test_portfolio_change_does_not_invalidate_model_hash(self, monkeypatch):
        """A stop_loss change in portfolio registry must not affect model hash."""
        p_registry = get_portfolio_registry()
        tampered_p = copy.deepcopy(p_registry)
        tampered_p["risk_parity_pyramid_v1"]["config"]["strategies"]["Max_Return_AI"][
            "stop_loss"
        ] = -0.99
        monkeypatch.setattr("modules.versioning.get_portfolio_registry", lambda: tampered_p)
        # Model hash validation must still pass (reads model registry, not portfolio)
        validate_model_config_hash("quant_baseline_v1")

    def test_model_change_does_not_invalidate_portfolio_hash(self, monkeypatch):
        """A momentum weight change in model registry must not affect portfolio hash."""
        m_registry = get_model_registry()
        tampered_m = copy.deepcopy(m_registry)
        tampered_m["quant_baseline_v1"]["config"]["strategies"]["Max_Return_AI"][
            "base_weights"
        ]["momentum"] = 0.99
        monkeypatch.setattr("modules.versioning.get_model_registry", lambda: tampered_m)
        # Portfolio hash validation must still pass
        validate_portfolio_config_hash("risk_parity_pyramid_v1")

    def test_execution_change_does_not_invalidate_model_hash(self, monkeypatch):
        """A commission change in execution registry must not affect model hash."""
        e_registry = get_execution_registry()
        tampered_e = copy.deepcopy(e_registry)
        tampered_e["exec_next_open_v1"]["config"]["transaction_costs"]["commission_rate"] = 0.005
        monkeypatch.setattr("modules.versioning.get_execution_registry", lambda: tampered_e)
        # Model hash validation must still pass
        validate_model_config_hash("quant_baseline_v1")


# ---------------------------------------------------------------------------
# Code cross-validation
# ---------------------------------------------------------------------------

class TestCodeCrossValidation:
    def test_model_strategies_match_live_code(self):
        validate_model_strategies_match_code("quant_baseline_v1")

    def test_regime_weights_match_live_code(self):
        validate_regime_weights_match_code("quant_baseline_v1")

    def test_portfolio_strategies_match_live_code(self):
        validate_portfolio_strategies_match_code("risk_parity_pyramid_v1")

    def test_execution_constants_match_live_code(self):
        validate_execution_constants_match_code("exec_next_open_v1")

    def test_changed_weight_in_code_triggers_error(self):
        from modules import scoring
        mutated = copy.deepcopy(scoring.STRATEGIES)
        mutated["Max_Return_AI"]["weights"]["momentum"] = 0.99
        with patch.object(scoring, "STRATEGIES", mutated):
            with pytest.raises(ModelVersionMismatchError) as exc_info:
                validate_model_strategies_match_code("quant_baseline_v1")
        assert "Max_Return_AI" in str(exc_info.value)
        assert "momentum" in str(exc_info.value)

    def test_changed_regime_weight_in_code_triggers_error(self):
        from modules import scoring
        mutated = copy.deepcopy(scoring.REGIME_WEIGHTS)
        mutated["bullish"]["momentum"] = 0.99
        with patch.object(scoring, "REGIME_WEIGHTS", mutated):
            with pytest.raises(ModelVersionMismatchError) as exc_info:
                validate_regime_weights_match_code("quant_baseline_v1")
        assert "bullish" in str(exc_info.value)
        assert "momentum" in str(exc_info.value)

    def test_changed_stop_loss_in_code_triggers_portfolio_error(self):
        from modules import scoring
        mutated = copy.deepcopy(scoring.STRATEGIES)
        mutated["Max_Return_AI"]["stop_loss"] = -0.99
        with patch.object(scoring, "STRATEGIES", mutated):
            with pytest.raises(ModelVersionMismatchError) as exc_info:
                validate_portfolio_strategies_match_code("risk_parity_pyramid_v1")
        assert "Max_Return_AI" in str(exc_info.value)
        assert "stop_loss" in str(exc_info.value)

    def test_missing_strategy_in_code_triggers_error(self):
        from modules import scoring
        mutated = {k: v for k, v in scoring.STRATEGIES.items() if k != "MegaCap_AI"}
        with patch.object(scoring, "STRATEGIES", mutated):
            with pytest.raises(ModelVersionMismatchError) as exc_info:
                validate_model_strategies_match_code("quant_baseline_v1")
        assert "MegaCap_AI" in str(exc_info.value)

    def test_extra_strategy_in_code_triggers_error(self):
        from modules import scoring
        mutated = copy.deepcopy(scoring.STRATEGIES)
        mutated["Ghost_Strategy_AI"] = {
            "description": "Undocumented",
            "score_column": "score_ghost",
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
                validate_model_strategies_match_code("quant_baseline_v1")
        assert "Ghost_Strategy_AI" in str(exc_info.value)


# ---------------------------------------------------------------------------
# validate_all
# ---------------------------------------------------------------------------

class TestValidateAll:
    def test_validate_all_passes_for_current_state(self):
        """Full end-to-end validation with default versions."""
        validate_all()

    def test_validate_all_fails_if_model_hash_wrong(self, monkeypatch):
        registry = get_model_registry()
        bad = copy.deepcopy(registry)
        bad["quant_baseline_v1"]["model_config_hash"] = "0" * 64
        monkeypatch.setattr("modules.versioning.get_model_registry", lambda: bad)
        with pytest.raises(ModelVersionMismatchError):
            validate_all()

    def test_validate_all_fails_if_portfolio_hash_wrong(self, monkeypatch):
        registry = get_portfolio_registry()
        bad = copy.deepcopy(registry)
        bad["risk_parity_pyramid_v1"]["portfolio_config_hash"] = "f" * 64
        monkeypatch.setattr("modules.versioning.get_portfolio_registry", lambda: bad)
        with pytest.raises(ModelVersionMismatchError):
            validate_all()

    def test_validate_all_fails_if_execution_hash_wrong(self, monkeypatch):
        registry = get_execution_registry()
        bad = copy.deepcopy(registry)
        bad["exec_next_open_v1"]["execution_config_hash"] = "a" * 64
        monkeypatch.setattr("modules.versioning.get_execution_registry", lambda: bad)
        with pytest.raises(ModelVersionMismatchError):
            validate_all()


# ---------------------------------------------------------------------------
# Fail-closed rules are documented in execution config
# ---------------------------------------------------------------------------

class TestExecutionFailClosedConfig:
    def test_no_fallback_to_signal_price(self):
        config = get_execution_config("exec_next_open_v1")
        assert config["fail_closed"]["no_fallback_to_signal_price"] is True

    def test_no_fallback_to_previous_close(self):
        config = get_execution_config("exec_next_open_v1")
        assert config["fail_closed"]["no_fallback_to_previous_close"] is True

    def test_no_fallback_to_premarket(self):
        config = get_execution_config("exec_next_open_v1")
        assert config["fail_closed"]["no_fallback_to_premarket_price"] is True

    def test_price_method_is_first_bar(self):
        config = get_execution_config("exec_next_open_v1")
        assert config["price_method"] == "first_bar_after_regular_session_open"


# ---------------------------------------------------------------------------
# _canonical_json determinism
# ---------------------------------------------------------------------------

class TestCanonicalJson:
    def test_key_order_does_not_affect_hash(self):
        d1 = {"b": 2, "a": 1, "c": {"z": 99, "y": 88}}
        d2 = {"a": 1, "c": {"y": 88, "z": 99}, "b": 2}
        assert _canonical_json(d1) == _canonical_json(d2)

    def test_float_equality_preserved(self):
        j = _canonical_json({"momentum": 0.45})
        assert "0.45" in j
