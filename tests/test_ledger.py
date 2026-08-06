"""
Tests for modules/ledger.py

Covers:
  - Retry-stable signal_run_id
  - Deterministic feature_snapshot_id and signal_id
  - Single feature_snapshot shared across 6 strategies
  - Signal ID independence from portfolio/execution version
  - Atomic append with idempotency
  - Partial last-line healing
  - Corrupt line detection (integrity check)
  - Signal-run status flow
  - Ledger write failure prevents signal file publication
  - Record structure and content_hash
"""

import copy
import json
import os
from pathlib import Path

import pytest

from modules.ledger import (
    SCHEMA_VERSION,
    LedgerWriteError,
    _append_batch,
    _get_last_close,
    _heal_partial_last_line,
    _jsonl_path,
    _index_path,
    _make_content_hash,
    _needs_index_rebuild,
    _rebuild_index_from_jsonl,
    build_all_ledger_records,
    build_signal_run_record,
    check_ledger_file_sizes,
    check_ledger_integrity,
    compute_input_hash,
    compute_input_manifest,
    compute_universe_hash,
    get_signal_run_status,
    make_feature_snapshot_id,
    make_signal_id,
    make_signal_run_id,
    record_id_exists,
    update_signal_run_status,
    write_feature_snapshots,
    write_signal_records,
    write_signal_run,
    LEDGER_DIR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    """Redirect LEDGER_DIR to a temp directory for every test."""
    import modules.ledger as ledger_mod
    monkeypatch.setattr(ledger_mod, "LEDGER_DIR", tmp_path)
    yield tmp_path


YEAR_MONTH = "2026-08"
SESSION = "2026-08-05"
MODEL = "quant_baseline_v1"
HASH8 = "d303b9a1"  # first 8 chars of model_config_hash

FULL_MODEL_HASH = HASH8 + "2ce0f055" + "2eeaddee" + "6f9625db" + "d713d1e0" + "8d3f527f"
DATA_CUTOFF = "2026-08-05"          # canonical_data_cutoff = intended_session for standard runs
UNIVERSE_HASH = compute_universe_hash(["AAPL", "MSFT", "NVDA", "GOOGL", "META"])
INPUT_MANIFEST_HASH = "a1b2c3d4e5f6abcd"   # test placeholder, 16 hex chars


def _make_run_record(session=SESSION, attempt=1):
    rid = make_signal_run_id(
        session, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH
    )
    return build_signal_run_record(
        signal_run_id=rid,
        intended_session=session,
        model_version=MODEL,
        model_config_hash=FULL_MODEL_HASH,
        universe_hash=UNIVERSE_HASH,
        canonical_data_cutoff=DATA_CUTOFF,
        input_manifest_hash=INPUT_MANIFEST_HASH,
        universe_count=520,
        analyzed_count=148,
        attempt_number=attempt,
    )


def _make_snapshot(signal_run_id, ticker, features=None):
    f = features or {"price": 100.0, "mom_12_1": 0.20, "regime": "bullish"}
    ih = compute_input_hash(f)
    sid = make_feature_snapshot_id(signal_run_id, ticker, ih)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "feature_snapshot",
        "content_hash": "",
        "feature_snapshot_id": sid,
        "signal_run_id": signal_run_id,
        "ticker": ticker,
        "input_hash": ih,
        "features": f,
        "data_quality": {"fundamentals_complete": True},
    }


def _make_signal(signal_run_id, feature_snapshot_id, strategy_id, ticker):
    sid = make_signal_id(signal_run_id, MODEL, strategy_id, ticker, feature_snapshot_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "signal_record",
        "content_hash": "",
        "signal_id": sid,
        "signal_run_id": signal_run_id,
        "feature_snapshot_id": feature_snapshot_id,
        "ticker": ticker,
        "strategy_id": strategy_id,
        "model_version": MODEL,
        "model_config_hash": FULL_MODEL_HASH,
        "score": 75.0,
        "rank": 5,
    }


# ---------------------------------------------------------------------------
# 1. signal_run_id retry stability
# ---------------------------------------------------------------------------

class TestSignalRunId:
    def test_same_inputs_give_same_id(self):
        id1 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        id2 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        assert id1 == id2

    def test_same_day_retry_is_same_id(self):
        """A retry on the same day with same model must produce the same signal_run_id."""
        id1 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH, "scheduled")
        id2 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH, "scheduled")
        assert id1 == id2

    def test_new_session_gives_new_id(self):
        id1 = make_signal_run_id("2026-08-05", MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        id2 = make_signal_run_id("2026-08-06", MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        assert id1 != id2

    def test_new_model_version_gives_new_id(self):
        id1 = make_signal_run_id(SESSION, "quant_baseline_v1", FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        id2 = make_signal_run_id(SESSION, "quant_baseline_v2", FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        assert id1 != id2

    def test_id_format(self):
        rid = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        assert rid.startswith(f"run-{SESSION}-")
        assert len(rid) == len(f"run-{SESSION}-") + 12


# ---------------------------------------------------------------------------
# 2. feature_snapshot_id determinism
# ---------------------------------------------------------------------------

class TestFeatureSnapshotId:
    def test_same_input_gives_same_id(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        f = {"price": 100.0, "mom_12_1": 0.20}
        ih = compute_input_hash(f)
        id1 = make_feature_snapshot_id(run_id, "AAPL", ih)
        id2 = make_feature_snapshot_id(run_id, "AAPL", ih)
        assert id1 == id2

    def test_changed_price_gives_new_id(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        ih1 = compute_input_hash({"price": 100.0, "mom_12_1": 0.20})
        ih2 = compute_input_hash({"price": 101.0, "mom_12_1": 0.20})
        id1 = make_feature_snapshot_id(run_id, "AAPL", ih1)
        id2 = make_feature_snapshot_id(run_id, "AAPL", ih2)
        assert id1 != id2

    def test_different_tickers_give_different_ids(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        ih = compute_input_hash({"price": 100.0})
        id_aapl = make_feature_snapshot_id(run_id, "AAPL", ih)
        id_msft = make_feature_snapshot_id(run_id, "MSFT", ih)
        assert id_aapl != id_msft

    def test_id_format(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        ih = compute_input_hash({"x": 1})
        fid = make_feature_snapshot_id(run_id, "AAPL", ih)
        assert fid.startswith("fs-")
        assert len(fid) == 15  # "fs-" + 12


# ---------------------------------------------------------------------------
# 3. Six strategies share one feature_snapshot per ticker
# ---------------------------------------------------------------------------

class TestFeatureSnapshotSharing:
    def test_same_ticker_same_snapshot_across_strategies(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        features = {"price": 100.0, "mom_12_1": 0.20, "regime": "bullish"}
        ih = compute_input_hash(features)

        snapshot_id = make_feature_snapshot_id(run_id, "AAPL", ih)
        strategies = ["Max_Return_AI", "Quality_Momentum_AI", "Balanced_AI",
                      "Low_Risk_AI", "MegaCap_AI", "AI_Sentiment_AI"]

        # All 6 strategies for same ticker reference the same snapshot_id
        for strat in strategies:
            signal_id = make_signal_id(run_id, MODEL, strat, "AAPL", snapshot_id)
            assert signal_id.startswith("sig-"), f"Bad signal_id for {strat}"

        # The snapshot_id itself is the same for all strategies
        for strat in strategies:
            assert snapshot_id == make_feature_snapshot_id(run_id, "AAPL", ih)

    def test_strategies_get_different_signal_ids(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        ih = compute_input_hash({"price": 100.0})
        snapshot_id = make_feature_snapshot_id(run_id, "AAPL", ih)

        strategies = ["Max_Return_AI", "Quality_Momentum_AI", "Balanced_AI",
                      "Low_Risk_AI", "MegaCap_AI", "AI_Sentiment_AI"]
        signal_ids = [make_signal_id(run_id, MODEL, s, "AAPL", snapshot_id) for s in strategies]

        assert len(set(signal_ids)) == len(strategies), "Each strategy must give a unique signal_id"

    def test_signal_record_references_snapshot(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        ih = compute_input_hash({"price": 100.0})
        snapshot_id = make_feature_snapshot_id(run_id, "AAPL", ih)
        signal_id = make_signal_id(run_id, MODEL, "Max_Return_AI", "AAPL", snapshot_id)

        # signal_id must be derived from snapshot_id (same snapshot → same signal_id on retry)
        signal_id2 = make_signal_id(run_id, MODEL, "Max_Return_AI", "AAPL", snapshot_id)
        assert signal_id == signal_id2


# ---------------------------------------------------------------------------
# 4. Signal ID excludes portfolio/execution version
# ---------------------------------------------------------------------------

class TestSignalIdIndependence:
    def test_portfolio_version_does_not_affect_signal_id(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        ih = compute_input_hash({"price": 100.0})
        snapshot_id = make_feature_snapshot_id(run_id, "AAPL", ih)

        sid = make_signal_id(run_id, MODEL, "Max_Return_AI", "AAPL", snapshot_id)
        # portfolio_version is not passed → signal_id is the same regardless
        sid2 = make_signal_id(run_id, MODEL, "Max_Return_AI", "AAPL", snapshot_id)
        assert sid == sid2

    def test_same_signal_consumable_by_multiple_portfolios(self):
        """
        Two different portfolio versions consuming the same signal_id
        is valid by design — signal_id does not encode portfolio context.
        """
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        ih = compute_input_hash({"price": 100.0})
        snapshot_id = make_feature_snapshot_id(run_id, "AAPL", ih)
        signal_id = make_signal_id(run_id, MODEL, "Max_Return_AI", "AAPL", snapshot_id)

        # Both portfolio versions reference the same signal_id
        order_v1 = {"signal_id": signal_id, "portfolio_version": "risk_parity_pyramid_v1"}
        order_v2 = {"signal_id": signal_id, "portfolio_version": "equal_weight_top10_v1"}
        assert order_v1["signal_id"] == order_v2["signal_id"]


# ---------------------------------------------------------------------------
# 5. Append: idempotency and duplicate rejection
# ---------------------------------------------------------------------------

class TestAppend:
    def test_fresh_append_returns_count(self, isolated_ledger):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snap = _make_snapshot(run_id, "AAPL")
        n = write_feature_snapshots([snap], YEAR_MONTH)
        assert n == 1

    def test_full_duplicate_returns_zero(self, isolated_ledger):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snap = _make_snapshot(run_id, "AAPL")
        write_feature_snapshots([snap], YEAR_MONTH)

        # Second call with same records → idempotent, no error, returns 0
        snap2 = _make_snapshot(run_id, "AAPL")  # same input → same snapshot_id
        n = write_feature_snapshots([snap2], YEAR_MONTH)
        assert n == 0

    def test_partial_duplicate_raises_error(self, isolated_ledger):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snap_aapl = _make_snapshot(run_id, "AAPL")
        write_feature_snapshots([snap_aapl], YEAR_MONTH)

        # Try to write AAPL (already exists) + MSFT (new) in one batch
        snap_msft = _make_snapshot(run_id, "MSFT")
        with pytest.raises(LedgerWriteError, match="Partial duplicate"):
            write_feature_snapshots([snap_aapl, snap_msft], YEAR_MONTH)

    def test_different_run_same_ticker_allowed(self, isolated_ledger):
        run1 = make_signal_run_id("2026-08-05", MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        run2 = make_signal_run_id("2026-08-06", MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snap1 = _make_snapshot(run1, "AAPL")
        snap2 = _make_snapshot(run2, "AAPL")  # different run → different snapshot_id

        n1 = write_feature_snapshots([snap1], YEAR_MONTH)
        n2 = write_feature_snapshots([snap2], YEAR_MONTH)
        assert n1 == 1
        assert n2 == 1

    def test_content_hash_written_on_append(self, isolated_ledger):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snap = _make_snapshot(run_id, "AAPL")
        assert snap["content_hash"] == ""

        write_feature_snapshots([snap], YEAR_MONTH)

        jpath = _jsonl_path("feature_snapshot", YEAR_MONTH)
        with open(jpath) as f:
            stored = json.loads(f.readline())
        assert stored["content_hash"] != ""
        assert len(stored["content_hash"]) == 64

    def test_record_id_exists_after_write(self, isolated_ledger):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snap = _make_snapshot(run_id, "AAPL")
        write_feature_snapshots([snap], YEAR_MONTH)

        assert record_id_exists(snap["feature_snapshot_id"], "feature_snapshot", YEAR_MONTH)
        assert not record_id_exists("fs-nonexistent000", "feature_snapshot", YEAR_MONTH)


# ---------------------------------------------------------------------------
# 6. Partial last-line healing
# ---------------------------------------------------------------------------

class TestPartialLineHealing:
    def test_clean_file_unchanged(self, isolated_ledger, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_bytes(b'{"a":1}\n{"b":2}\n')
        removed = _heal_partial_last_line(p)
        assert removed == 0
        assert p.read_bytes() == b'{"a":1}\n{"b":2}\n'

    def test_partial_last_line_truncated(self, isolated_ledger, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_bytes(b'{"a":1}\n{"b":2')  # partial last line, no trailing \n
        removed = _heal_partial_last_line(p)
        assert removed == 6  # len('{"b":2') = 6
        assert p.read_bytes() == b'{"a":1}\n'

    def test_no_newline_at_all_truncates_to_empty(self, isolated_ledger, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_bytes(b'{"a":1}')  # no newline at all
        removed = _heal_partial_last_line(p)
        assert removed == 7
        assert p.read_bytes() == b''

    def test_empty_file_unchanged(self, isolated_ledger, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_bytes(b'')
        removed = _heal_partial_last_line(p)
        assert removed == 0

    def test_corrupt_last_line_healed_and_valid_lines_preserved(self, isolated_ledger):
        """Append a partial line to the JSONL, then a fresh write should heal it first."""
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snap = _make_snapshot(run_id, "AAPL")
        write_feature_snapshots([snap], YEAR_MONTH)

        # Corrupt the file by appending a partial JSON line
        jpath = _jsonl_path("feature_snapshot", YEAR_MONTH)
        with open(jpath, "ab") as f:
            f.write(b'{"feature_snapshot_id":"fs-corrupted')  # no closing brace or \n

        # Writing a new valid record should trigger healing of the partial line
        snap2 = _make_snapshot(run_id, "MSFT")
        n = write_feature_snapshots([snap2], YEAR_MONTH)
        assert n == 1

        ok, bad, tampered = check_ledger_integrity("feature_snapshot", YEAR_MONTH)
        assert bad == 0
        assert tampered == 0
        assert ok == 2


# ---------------------------------------------------------------------------
# 7. Integrity check
# ---------------------------------------------------------------------------

class TestIntegrityCheck:
    def test_all_valid_records(self, isolated_ledger):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snaps = [_make_snapshot(run_id, t) for t in ["AAPL", "MSFT", "NVDA"]]
        write_feature_snapshots(snaps, YEAR_MONTH)

        ok, bad, tampered = check_ledger_integrity("feature_snapshot", YEAR_MONTH)
        assert ok == 3
        assert bad == 0
        assert tampered == 0

    def test_missing_file_returns_zeros(self, isolated_ledger):
        ok, bad, tampered = check_ledger_integrity("feature_snapshot", "2099-01")
        assert ok == 0
        assert bad == 0
        assert tampered == 0

    def test_corrupt_line_counted(self, isolated_ledger):
        jpath = _jsonl_path("feature_snapshot", YEAR_MONTH)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text('{"feature_snapshot_id":"fs-abc"}\n{bad json\n')

        ok, bad, tampered = check_ledger_integrity("feature_snapshot", YEAR_MONTH)
        assert ok == 1
        assert bad == 1
        assert tampered == 0

    def test_tampered_record_detected(self, isolated_ledger):
        """A record whose content_hash was altered after writing is reported as tampered."""
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snap = _make_snapshot(run_id, "AAPL")
        write_feature_snapshots([snap], YEAR_MONTH)

        # Corrupt the content_hash in the stored record
        jpath = _jsonl_path("feature_snapshot", YEAR_MONTH)
        stored = json.loads(jpath.read_text())
        stored["content_hash"] = "a" * 64  # wrong hash
        jpath.write_text(json.dumps(stored, sort_keys=True) + "\n")

        ok, bad, tampered = check_ledger_integrity("feature_snapshot", YEAR_MONTH)
        assert tampered == 1
        assert ok == 0
        assert bad == 0


# ---------------------------------------------------------------------------
# 8. Signal run status flow
# ---------------------------------------------------------------------------

class TestSignalRunStatus:
    def test_initial_status_is_none(self, isolated_ledger):
        assert get_signal_run_status("run-nonexistent", YEAR_MONTH) is None

    def test_pending_to_completed(self, isolated_ledger):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        update_signal_run_status(run_id, "pending", YEAR_MONTH)

        status = get_signal_run_status(run_id, YEAR_MONTH)
        assert status["status"] == "pending"

        update_signal_run_status(run_id, "completed", YEAR_MONTH,
                                  n_feature_snapshots=148, n_signal_records=888)
        status = get_signal_run_status(run_id, YEAR_MONTH)
        assert status["status"] == "completed"
        assert status["n_feature_snapshots"] == 148

    def test_failed_status_includes_error(self, isolated_ledger):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        update_signal_run_status(run_id, "failed", YEAR_MONTH, error="I/O error: disk full")
        status = get_signal_run_status(run_id, YEAR_MONTH)
        assert status["status"] == "failed"
        assert "disk full" in status["error"]

    def test_write_signal_run_idempotent(self, isolated_ledger):
        rec = _make_run_record()
        n1 = write_signal_run(rec, YEAR_MONTH)
        n2 = write_signal_run(copy.deepcopy(rec), YEAR_MONTH)
        assert n1 == 1
        assert n2 == 0  # idempotent


# ---------------------------------------------------------------------------
# 9. build_all_ledger_records structure
# ---------------------------------------------------------------------------

class TestBuildAllLedgerRecords:
    """
    Verify that build_all_ledger_records produces correctly structured records
    from minimal in-memory data (no real yfinance/scoring needed).
    """

    _STRATEGIES_CFG = {
        "Max_Return_AI": {
            "score_column": "score_max_return",
            "weights": {"momentum": 0.45, "quality": 0.30, "value": 0.20, "sentiment": 0.05},
            "buy_top_n": 20,
            "min_score_percentile": 0.80,
        },
        "Low_Risk_AI": {
            "score_column": "score_low_risk",
            "weights": {"momentum": 0.25, "quality": 0.45, "value": 0.25, "sentiment": 0.05},
            "buy_top_n": 25,
            "min_score_percentile": 0.70,
        },
    }

    def _make_analyzed_ticker(self, ticker, price=100.0):
        return {
            "ticker": ticker,
            "price": price,
            "sma200": 95.0,
            "above_sma200": True,
            "vol60": 0.22,
            "rsi": 58.0,
            "overbought": False,
            "very_weak_rsi": False,
            "is_megacap": False,
            "mom_12_1": 0.20,
            "mom_6_1": 0.15,
            "mom_3_1": 0.08,
            "ret_1m": 0.03,
            "relative_strength_3m": 0.10,
            "momentum_score": 65.0,
            "quality_score": 72.0,
            "value_score": 55.0,
            "sentiment_score": 60.0,
            "insider_buying": False,
            "score_max_return": 68.0,
            "score_low_risk": 65.0,
        }

    def _make_candidate(self, ticker, strategy_cfg, rank=5, score=68.0):
        col = strategy_cfg["score_column"]
        row = self._make_analyzed_ticker(ticker)
        row["rank"] = rank
        row["strategy_score"] = score
        row["score_percentile"] = 0.85
        row[col] = score
        return row

    def test_one_snapshot_per_ticker(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        tickers = ["AAPL", "MSFT", "NVDA"]
        analyzed = [self._make_analyzed_ticker(t) for t in tickers]
        candidates_max = [self._make_candidate(t, self._STRATEGIES_CFG["Max_Return_AI"]) for t in tickers]
        candidates_low = [self._make_candidate(t, self._STRATEGIES_CFG["Low_Risk_AI"]) for t in tickers]

        strategies_payload = {
            "Max_Return_AI": {"candidates": candidates_max},
            "Low_Risk_AI": {"candidates": candidates_low},
        }
        macro = {"inverted": False, "spread": 0.15}

        snapshots, signals = build_all_ledger_records(
            run_id, analyzed, strategies_payload,
            fundamentals={}, sentiment_scores={}, full_earnings={},
            macro=macro, regime="bullish",
            model_version=MODEL, model_config_hash=FULL_MODEL_HASH,
            strategies_config=self._STRATEGIES_CFG,
        )

        # One snapshot per ticker
        assert len(snapshots) == 3
        tickers_in_snapshots = {s["ticker"] for s in snapshots}
        assert tickers_in_snapshots == set(tickers)

        # 2 strategies × 3 tickers = 6 signal records
        assert len(signals) == 6

    def test_snapshots_have_required_fields(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        analyzed = [self._make_analyzed_ticker("AAPL")]
        strategies_payload = {
            "Max_Return_AI": {
                "candidates": [self._make_candidate("AAPL", self._STRATEGIES_CFG["Max_Return_AI"])]
            }
        }
        macro = {"inverted": False, "spread": 0.10}

        snapshots, _ = build_all_ledger_records(
            run_id, analyzed, strategies_payload,
            fundamentals={}, sentiment_scores={}, full_earnings={},
            macro=macro, regime="bullish",
            model_version=MODEL, model_config_hash=FULL_MODEL_HASH,
            strategies_config=self._STRATEGIES_CFG,
        )

        snap = snapshots[0]
        for field in ("feature_snapshot_id", "signal_run_id", "ticker",
                      "input_hash", "features", "data_quality"):
            assert field in snap, f"Missing field: {field}"
        assert snap["record_type"] == "feature_snapshot"
        assert snap["schema_version"] == SCHEMA_VERSION

    def test_signal_records_have_required_fields(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        analyzed = [self._make_analyzed_ticker("AAPL")]
        strategies_payload = {
            "Max_Return_AI": {
                "candidates": [self._make_candidate("AAPL", self._STRATEGIES_CFG["Max_Return_AI"])]
            }
        }
        macro = {"inverted": False, "spread": 0.10}

        _, signals = build_all_ledger_records(
            run_id, analyzed, strategies_payload,
            fundamentals={}, sentiment_scores={}, full_earnings={},
            macro=macro, regime="bullish",
            model_version=MODEL, model_config_hash=FULL_MODEL_HASH,
            strategies_config=self._STRATEGIES_CFG,
        )

        sig = signals[0]
        for field in ("signal_id", "signal_run_id", "feature_snapshot_id", "ticker",
                      "strategy_id", "model_version", "model_config_hash",
                      "score", "rank", "recommendation",
                      "component_scores", "penalties_applied"):
            assert field in sig, f"Missing field: {field}"
        assert sig["record_type"] == "signal_record"
        # No portfolio or execution version in signal_record
        assert "portfolio_version" not in sig
        assert "execution_version" not in sig

    def test_same_ticker_shares_snapshot_id_across_strategies(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        analyzed = [self._make_analyzed_ticker("AAPL")]
        strategies_payload = {
            "Max_Return_AI": {
                "candidates": [self._make_candidate("AAPL", self._STRATEGIES_CFG["Max_Return_AI"])]
            },
            "Low_Risk_AI": {
                "candidates": [self._make_candidate("AAPL", self._STRATEGIES_CFG["Low_Risk_AI"])]
            },
        }
        macro = {"inverted": False, "spread": 0.10}

        snapshots, signals = build_all_ledger_records(
            run_id, analyzed, strategies_payload,
            fundamentals={}, sentiment_scores={}, full_earnings={},
            macro=macro, regime="bullish",
            model_version=MODEL, model_config_hash=FULL_MODEL_HASH,
            strategies_config=self._STRATEGIES_CFG,
        )

        assert len(snapshots) == 1  # one snapshot for AAPL
        snapshot_id = snapshots[0]["feature_snapshot_id"]

        # Both signal records reference the same snapshot_id
        sig_snapshots = {s["feature_snapshot_id"] for s in signals}
        assert sig_snapshots == {snapshot_id}, "Both strategies must reference the same feature_snapshot"

    def test_strategies_produce_different_signal_ids(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        analyzed = [self._make_analyzed_ticker("AAPL")]
        strategies_payload = {
            "Max_Return_AI": {
                "candidates": [self._make_candidate("AAPL", self._STRATEGIES_CFG["Max_Return_AI"])]
            },
            "Low_Risk_AI": {
                "candidates": [self._make_candidate("AAPL", self._STRATEGIES_CFG["Low_Risk_AI"])]
            },
        }
        macro = {"inverted": False, "spread": 0.10}

        _, signals = build_all_ledger_records(
            run_id, analyzed, strategies_payload,
            fundamentals={}, sentiment_scores={}, full_earnings={},
            macro=macro, regime="bullish",
            model_version=MODEL, model_config_hash=FULL_MODEL_HASH,
            strategies_config=self._STRATEGIES_CFG,
        )

        signal_ids = [s["signal_id"] for s in signals]
        assert len(set(signal_ids)) == 2, "Different strategies must give different signal_ids"

    def test_changed_input_gives_new_snapshot_and_signal_id(self):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)

        def get_ids(price):
            analyzed = [self._make_analyzed_ticker("AAPL", price=price)]
            strat = self._STRATEGIES_CFG["Max_Return_AI"]
            candidate = self._make_candidate("AAPL", strat)
            candidate["price"] = price
            strategies_payload = {"Max_Return_AI": {"candidates": [candidate]}}
            macro = {"inverted": False, "spread": 0.10}
            snapshots, signals = build_all_ledger_records(
                run_id, analyzed, strategies_payload,
                fundamentals={}, sentiment_scores={}, full_earnings={},
                macro=macro, regime="bullish",
                model_version=MODEL, model_config_hash=FULL_MODEL_HASH,
                strategies_config=self._STRATEGIES_CFG,
            )
            return snapshots[0]["feature_snapshot_id"], signals[0]["signal_id"]

        snap_id1, sig_id1 = get_ids(100.0)
        snap_id2, sig_id2 = get_ids(101.0)
        assert snap_id1 != snap_id2
        assert sig_id1 != sig_id2


# ---------------------------------------------------------------------------
# 10. signal_run_id includes data_cutoff and universe_hash
# ---------------------------------------------------------------------------

class TestSignalRunIdV2:
    """Verify that signal_run_id is sensitive to all 6 key components."""

    def test_identical_retry_gives_same_id(self):
        """All inputs identical → same ID (safe to retry)."""
        id1 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        id2 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        assert id1 == id2

    def test_new_canonical_data_cutoff_gives_new_id(self):
        """Different canonical_data_cutoff (planned session date) → new run ID."""
        id1 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, "2026-08-04", INPUT_MANIFEST_HASH)
        id2 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, "2026-08-05", INPUT_MANIFEST_HASH)
        assert id1 != id2

    def test_new_input_manifest_hash_gives_new_id(self):
        """
        Same session, same cutoff, but different data (price revision, new ticker,
        staleness change) → different input_manifest_hash → new run ID.
        """
        mh1 = "a1b2c3d4e5f6abcd"
        mh2 = "b2c3d4e5f6a7bcde"
        id1 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, mh1)
        id2 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, mh2)
        assert id1 != id2

    def test_new_model_config_hash_gives_new_id(self):
        """Config change detected by hash → new run ID."""
        hash_a = "a" * 8 + FULL_MODEL_HASH[8:]
        hash_b = "b" * 8 + FULL_MODEL_HASH[8:]
        id1 = make_signal_run_id(SESSION, MODEL, hash_a, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        id2 = make_signal_run_id(SESSION, MODEL, hash_b, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        assert id1 != id2

    def test_new_universe_hash_gives_new_id(self):
        """Different ticker universe → new universe_hash → new run ID."""
        uh1 = compute_universe_hash(["AAPL", "MSFT", "NVDA"])
        uh2 = compute_universe_hash(["AAPL", "MSFT", "NVDA", "TSLA"])
        id1 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, uh1, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        id2 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, uh2, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        assert id1 != id2

    def test_attempt_number_does_not_affect_id(self):
        """
        attempt_number is NOT a parameter to make_signal_run_id —
        retries of the same logical run always produce the same ID.
        """
        id1 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH, "scheduled")
        id2 = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH, "scheduled")
        assert id1 == id2

    def test_universe_hash_order_independent(self):
        """compute_universe_hash sorts input so ticker order does not matter."""
        uh1 = compute_universe_hash(["AAPL", "MSFT", "NVDA"])
        uh2 = compute_universe_hash(["NVDA", "AAPL", "MSFT"])
        assert uh1 == uh2

    def test_universe_hash_detects_added_ticker(self):
        """Adding one ticker to the universe changes the hash."""
        uh1 = compute_universe_hash(["AAPL", "MSFT"])
        uh2 = compute_universe_hash(["AAPL", "MSFT", "GOOGL"])
        assert uh1 != uh2


# ---------------------------------------------------------------------------
# 11. Index rebuild validates content_hash
# ---------------------------------------------------------------------------

class TestIndexRebuildValidation:
    """_rebuild_index_from_jsonl must validate content_hash and flag tampered records."""

    def test_valid_records_not_flagged_as_tampered(self, isolated_ledger):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snaps = [_make_snapshot(run_id, t) for t in ["AAPL", "MSFT"]]
        write_feature_snapshots(snaps, YEAR_MONTH)

        jpath = _jsonl_path("feature_snapshot", YEAR_MONTH)
        index = _rebuild_index_from_jsonl(jpath, "feature_snapshot", YEAR_MONTH, "feature_snapshot_id")
        assert index["tampered_ids"] == []
        assert index["count"] == 2

    def test_tampered_record_flagged_in_rebuild(self, isolated_ledger):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snap = _make_snapshot(run_id, "AAPL")
        write_feature_snapshots([snap], YEAR_MONTH)

        # Corrupt content_hash in the written record
        jpath = _jsonl_path("feature_snapshot", YEAR_MONTH)
        rec = json.loads(jpath.read_text())
        original_id = rec["feature_snapshot_id"]
        rec["content_hash"] = "b" * 64
        jpath.write_text(json.dumps(rec, sort_keys=True) + "\n")

        index = _rebuild_index_from_jsonl(jpath, "feature_snapshot", YEAR_MONTH, "feature_snapshot_id")
        assert original_id in index["tampered_ids"]
        assert original_id in index["ids"]  # still indexed so it's not re-appended

    def test_tampered_record_still_indexed(self, isolated_ledger):
        """Tampered records go into both tampered_ids and ids to prevent re-append."""
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snap = _make_snapshot(run_id, "AAPL")
        write_feature_snapshots([snap], YEAR_MONTH)

        jpath = _jsonl_path("feature_snapshot", YEAR_MONTH)
        rec = json.loads(jpath.read_text())
        rec["content_hash"] = "c" * 64
        jpath.write_text(json.dumps(rec, sort_keys=True) + "\n")

        index = _rebuild_index_from_jsonl(jpath, "feature_snapshot", YEAR_MONTH, "feature_snapshot_id")
        assert len(index["ids"]) == 1

    def test_corrupt_json_skipped_in_rebuild(self, isolated_ledger):
        """Lines that cannot be parsed as JSON are silently skipped."""
        jpath = _jsonl_path("feature_snapshot", YEAR_MONTH)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text('{"feature_snapshot_id":"fs-good123","content_hash":""}\n{bad json\n')

        index = _rebuild_index_from_jsonl(jpath, "feature_snapshot", YEAR_MONTH, "feature_snapshot_id")
        assert "fs-good123" in index["ids"]
        assert index["count"] == 1


# ---------------------------------------------------------------------------
# 12. Input manifest hash — compute_input_manifest
# ---------------------------------------------------------------------------

import pandas as pd
from datetime import date as date_type


def _make_test_df(last_date: str, close: float, n_rows: int = 100) -> "pd.DataFrame":
    """Build a minimal DataFrame that looks like a yfinance OHLCV result."""
    dates = pd.date_range(end=last_date, periods=n_rows, freq="B")
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000},
        index=dates,
    )


class TestInputManifest:
    """compute_input_manifest produces a deterministic hash of actual data inputs."""

    def _make_market_data(self, tickers_with_prices: dict, last_date: str) -> dict:
        return {t: _make_test_df(last_date, price) for t, price in tickers_with_prices.items()}

    def test_identical_inputs_give_same_hash(self):
        md = self._make_market_data({"AAPL": 210.0, "MSFT": 420.0}, SESSION)
        m1 = compute_input_manifest(md, {}, SESSION)
        m2 = compute_input_manifest(md, {}, SESSION)
        assert m1["input_manifest_hash"] == m2["input_manifest_hash"]

    def test_changed_price_gives_new_hash(self):
        """Same timestamp, different close price → different manifest hash."""
        md1 = self._make_market_data({"AAPL": 210.0, "MSFT": 420.0}, SESSION)
        md2 = self._make_market_data({"AAPL": 211.0, "MSFT": 420.0}, SESSION)  # AAPL changed
        m1 = compute_input_manifest(md1, {}, SESSION)
        m2 = compute_input_manifest(md2, {}, SESSION)
        assert m1["input_manifest_hash"] != m2["input_manifest_hash"]

    def test_changed_input_for_one_ticker_changes_hash(self):
        """Changing one ticker's data changes the manifest even if all timestamps are equal."""
        md1 = self._make_market_data({"AAPL": 200.0, "MSFT": 400.0, "NVDA": 800.0}, SESSION)
        md2 = self._make_market_data({"AAPL": 200.0, "MSFT": 400.0, "NVDA": 801.0}, SESSION)  # NVDA +1
        m1 = compute_input_manifest(md1, {}, SESSION)
        m2 = compute_input_manifest(md2, {}, SESSION)
        assert m1["input_manifest_hash"] != m2["input_manifest_hash"]

    def test_ticker_order_does_not_affect_hash(self):
        """Manifest hash is order-independent (sorted internally)."""
        tickers = {"AAPL": 210.0, "MSFT": 420.0, "NVDA": 800.0}
        # Python dicts maintain insertion order; pass same values, different order
        md1 = {t: _make_test_df(SESSION, p) for t, p in tickers.items()}
        reversed_items = list(tickers.items())[::-1]
        md2 = {t: _make_test_df(SESSION, p) for t, p in reversed_items}

        m1 = compute_input_manifest(md1, {}, SESSION)
        m2 = compute_input_manifest(md2, {}, SESSION)
        assert m1["input_manifest_hash"] == m2["input_manifest_hash"]

    def test_stale_ticker_appears_in_manifest_status(self):
        """A ticker whose last date is >5 days before the session is marked stale."""
        stale_date = "2026-07-25"  # 11 days before SESSION (2026-08-05)
        md = {
            "AAPL": _make_test_df(SESSION, 210.0),
            "STALE": _make_test_df(stale_date, 50.0),
        }
        manifest = compute_input_manifest(md, {}, SESSION)
        assert manifest["stale_ticker_count"] == 1
        assert manifest["valid_ticker_count"] == 1

    def test_excluded_ticker_changes_hash(self):
        """A ticker removed by quality check changes the manifest hash."""
        md_with = self._make_market_data({"AAPL": 210.0, "DEAD": 5.0}, SESSION)
        md_without = self._make_market_data({"AAPL": 210.0}, SESSION)
        qr_with_exclusion = {"volume_excluded": [{"ticker": "DEAD", "zero_days": 5}]}

        m1 = compute_input_manifest(md_with, {}, SESSION)
        m2 = compute_input_manifest(md_without, qr_with_exclusion, SESSION)
        # Both have DEAD in the manifest (as excluded or as present), so hashes differ
        assert m1["input_manifest_hash"] != m2["input_manifest_hash"]

    def test_excluded_count_from_quality_report(self):
        """volume_excluded tickers are counted in excluded_ticker_count."""
        md = self._make_market_data({"AAPL": 210.0}, SESSION)
        qr = {"volume_excluded": [{"ticker": "DEAD1", "zero_days": 5}, {"ticker": "DEAD2", "zero_days": 4}]}
        manifest = compute_input_manifest(md, qr, SESSION)
        assert manifest["excluded_ticker_count"] == 2

    def test_manifest_has_required_keys(self):
        md = self._make_market_data({"AAPL": 210.0}, SESSION)
        manifest = compute_input_manifest(md, {}, SESSION)
        for key in (
            "input_manifest_hash",
            "canonical_data_cutoff",
            "observed_min_data_timestamp",
            "observed_max_data_timestamp",
            "ticker_count",
            "valid_ticker_count",
            "stale_ticker_count",
            "excluded_ticker_count",
        ):
            assert key in manifest, f"Missing key: {key}"

    def test_canonical_data_cutoff_equals_intended_session(self):
        """canonical_data_cutoff is always the intended_session for standard runs."""
        md = self._make_market_data({"AAPL": 210.0}, SESSION)
        manifest = compute_input_manifest(md, {}, SESSION)
        assert manifest["canonical_data_cutoff"] == SESSION

    def test_input_manifest_drives_signal_run_id(self):
        """
        Same session + same timestamp, but different actual data → different signal_run_id.
        This is the core invariant: observed_max_data_timestamp alone is insufficient.
        """
        md1 = self._make_market_data({"AAPL": 210.0, "MSFT": 420.0}, SESSION)
        md2 = self._make_market_data({"AAPL": 211.0, "MSFT": 420.0}, SESSION)  # price revised
        m1 = compute_input_manifest(md1, {}, SESSION)
        m2 = compute_input_manifest(md2, {}, SESSION)

        # Both have the same observed_max_data_timestamp (same SESSION date)
        assert m1["observed_max_data_timestamp"] == m2["observed_max_data_timestamp"]
        # But the manifest hashes differ (different actual data)
        assert m1["input_manifest_hash"] != m2["input_manifest_hash"]
        # And therefore signal_run_ids differ
        run_id1 = make_signal_run_id(
            SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH,
            m1["canonical_data_cutoff"], m1["input_manifest_hash"]
        )
        run_id2 = make_signal_run_id(
            SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH,
            m2["canonical_data_cutoff"], m2["input_manifest_hash"]
        )
        assert run_id1 != run_id2


# ---------------------------------------------------------------------------
# 13. Ledger file size warnings
# ---------------------------------------------------------------------------

class TestLedgerFileSizes:
    """check_ledger_file_sizes returns files that exceed the threshold."""

    def test_no_files_returns_empty(self, isolated_ledger):
        result = check_ledger_file_sizes(YEAR_MONTH)
        assert result == {}

    def test_small_files_not_warned(self, isolated_ledger):
        run_id = make_signal_run_id(SESSION, MODEL, FULL_MODEL_HASH, UNIVERSE_HASH, DATA_CUTOFF, INPUT_MANIFEST_HASH)
        snaps = [_make_snapshot(run_id, t) for t in ["AAPL", "MSFT"]]
        write_feature_snapshots(snaps, YEAR_MONTH)

        result = check_ledger_file_sizes(YEAR_MONTH, warn_threshold_mb=50.0)
        assert result == {}

    def test_large_file_detected(self, isolated_ledger):
        """Files at or above the threshold are returned with their size."""
        jpath = _jsonl_path("feature_snapshot", YEAR_MONTH)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        # Write 51 MB of data
        jpath.write_bytes(b"x" * (51 * 1024 * 1024))

        result = check_ledger_file_sizes(YEAR_MONTH, warn_threshold_mb=50.0)
        assert jpath.name in result
        assert result[jpath.name] >= 50.0

    def test_threshold_exactly_at_boundary_triggers_warning(self, isolated_ledger):
        """File exactly at the threshold (≥) triggers the warning."""
        jpath = _jsonl_path("signal_record", YEAR_MONTH)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_bytes(b"x" * (1024 * 1024))  # exactly 1 MB

        result = check_ledger_file_sizes(YEAR_MONTH, warn_threshold_mb=1.0)
        assert jpath.name in result


# ---------------------------------------------------------------------------
# 14. Workflow persistence verification (structural checks on workflow files)
# ---------------------------------------------------------------------------

class TestWorkflowPersistenceGuarantees:
    """
    Verify that the GitHub Actions workflow files enforce the correct
    persistence order and downstream conclusion check.
    These are static checks on the workflow YAML content.
    """

    def _read_workflow(self, name: str) -> str:
        p = Path(__file__).parent.parent / ".github" / "workflows" / f"{name}.yml"
        return p.read_text(encoding="utf-8")

    def test_signal_workflow_exits_nonzero_on_push_failure(self):
        """signal.yml must exit 1 after all push retries fail (prevents false success)."""
        content = self._read_workflow("signal")
        assert "exit 1" in content, "signal.yml must exit 1 on push failure"

    def test_signal_workflow_commit_step_always_runs(self):
        """signal.yml commit step must use 'if: always()' to commit partial state on failure."""
        content = self._read_workflow("signal")
        assert "if: always()" in content

    def test_execute_workflow_requires_signal_conclusion_success(self):
        """execute.yml must check conclusion == 'success' before running."""
        content = self._read_workflow("execute")
        assert "conclusion == 'success'" in content, (
            "execute.yml must guard on workflow_run conclusion to prevent "
            "running after a failed signal push"
        )

    def test_premarket_workflow_has_cancel_in_progress_false(self):
        """premarket.yml must not cancel in-progress runs (state writes must complete)."""
        content = self._read_workflow("premarket")
        assert "cancel-in-progress: false" in content

    def test_execute_workflow_has_cancel_in_progress_false(self):
        """execute.yml must not cancel in-progress runs (state writes must complete)."""
        content = self._read_workflow("execute")
        assert "cancel-in-progress: false" in content
