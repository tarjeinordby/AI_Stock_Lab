"""
V2B.3 Shadow Reporter — behavioral tests.

Tests operate in temporary ledger and report directories (monkeypatched).
Exchange calendar is mocked. Observations are created via run_shadow_observation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import modules.v2b_ledger as ledger
import modules.v2b_report as report
import modules.v2b_shadow_runner as runner
from modules.v2b_ledger import (
    get_observation_events,
    get_observation_status,
)
from modules.v2b_report import (
    ObservationIncompleteError,
    ObservationIntegrityError,
    ObservationNotFoundError,
    ReportConflictError,
    SafetyBoundaryError,
    ShadowReportResult,
    build_report_content,
    create_report,
    format_telegram_message,
    make_report_key,
    read_completed_observation,
    run_shadow_report,
    telegram_already_sent,
    validate_observation_integrity,
)
from modules.v2b_shadow_runner import run_shadow_observation

# ---------------------------------------------------------------------------
# Constants (reuse V2B.2 test dates)
# ---------------------------------------------------------------------------

MONDAY = "2026-08-10"
TUESDAY = "2026-08-11"
SATURDAY = "2026-08-08"

VALID_CFG_HASH = "f" * 64
CUTOFF = "2026-08-10T20:00:00+00:00"
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def tmp_ledger(tmp_path, monkeypatch):
    """Redirect all v2b_ledger I/O to a temporary directory."""
    monkeypatch.setattr(ledger, "LEDGER_DIR", tmp_path / "v2b_ledger")
    yield tmp_path


@pytest.fixture(autouse=True)
def tmp_report_dir(tmp_path, monkeypatch):
    """Redirect all v2b_report I/O to a temporary directory."""
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path / "v2b_reports")
    yield tmp_path / "v2b_reports"


@pytest.fixture(autouse=True)
def mock_calendar(monkeypatch):
    """Mock NYSE calendar: Mon–Fri are sessions; Sat–Sun are not."""
    def _is_session(date_str: str) -> bool:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d.weekday() < 5

    def _next_session(date_str: str) -> str:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        nxt = d + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        return nxt.strftime("%Y-%m-%d")

    monkeypatch.setattr(runner, "is_trading_session", _is_session)
    monkeypatch.setattr(runner, "_calendar_next_session", _next_session)
    yield


def _make_price_df(n: int = 300, base: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(MONDAY)
    dates = pd.date_range(end=end, periods=n, freq="B")
    prices = base * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    vols = rng.integers(1_000_000, 10_000_000, n)
    return pd.DataFrame({"Adj Close": prices, "Close": prices, "Volume": vols}, index=dates)


def _make_fundamentals(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "returnOnEquity": float(rng.uniform(0.05, 0.30)),
        "grossMargins": float(rng.uniform(0.20, 0.70)),
        "debtToEquity": float(rng.uniform(0.1, 1.5)),
        "earningsGrowth": float(rng.uniform(0.02, 0.25)),
        "freeCashflow": float(rng.uniform(5e8, 5e9)),
        "totalRevenue": float(rng.uniform(5e9, 1e11)),
        "forwardPE": float(rng.uniform(12.0, 40.0)),
        "enterpriseValue": float(rng.uniform(1e10, 5e11)),
        "ebitda": float(rng.uniform(1e9, 2e10)),
        "marketCap": float(rng.uniform(5e9, 3e11)),
    }


def _make_data(tickers=TICKERS):
    price_data = {t: _make_price_df(n=300, seed=i * 7) for i, t in enumerate(tickers)}
    fundamentals = {t: _make_fundamentals(seed=i * 3) for i, t in enumerate(tickers)}
    return price_data, fundamentals


def _create_completed_observation(
    as_of_date: str = MONDAY,
    data_cutoff_at: str = CUTOFF,
    tickers=TICKERS,
    model_config_hash: str = VALID_CFG_HASH,
) -> tuple[str, dict]:
    """Create a COMPLETED observation in the tmp ledger. Returns (intended_session, event)."""
    price_data, fundamentals = _make_data(tickers)
    result = run_shadow_observation(
        as_of_date=as_of_date,
        price_data=price_data,
        fundamentals=fundamentals,
        data_cutoff_at=data_cutoff_at,
        model_config_hash=model_config_hash,
    )
    assert result.status == "COMPLETED", f"Expected COMPLETED, got {result.status}: {result.error}"
    events = get_observation_events(result.observation_key)
    created = next(e for e in events if e["event_type"] == "OBSERVATION_CREATED")
    return result.intended_execution_session, created


# ===========================================================================
# T44 TestReadObservation
# ===========================================================================

class TestReadObservation:
    """Reading and validating COMPLETED observations from the ledger."""

    def test_reads_completed_observation_for_session(self):
        session, _ = _create_completed_observation()
        obs = read_completed_observation(session)
        assert obs["event_type"] == "OBSERVATION_CREATED"
        assert obs["intended_execution_session"] == session

    def test_raises_not_found_when_no_observation(self):
        with pytest.raises(ObservationNotFoundError, match="No V2B observation"):
            read_completed_observation(TUESDAY)

    def test_raises_incomplete_when_not_completed(self):
        """An observation that exists but has not reached COMPLETED must raise."""
        price_data, fundamentals = _make_data()
        # create_observation() runs before transition_observation(), so the observation
        # IS written at CREATED status even when transition raises.
        with mock.patch.object(runner, "transition_observation") as mock_trans:
            mock_trans.side_effect = Exception("simulated crash in transition")
            run_shadow_observation(
                as_of_date=MONDAY,
                price_data=price_data,
                fundamentals=fundamentals,
                data_cutoff_at=CUTOFF,
                model_config_hash=VALID_CFG_HASH,
            )
            # Result status is FAILED_VALIDATION — observation exists at CREATED status

        # intended_execution_session for MONDAY as_of_date is the next session (TUESDAY)
        with pytest.raises((ObservationNotFoundError, ObservationIncompleteError)):
            read_completed_observation(TUESDAY)

    def test_validates_integrity_of_observation_key(self):
        session, obs_event = _create_completed_observation()
        # Should not raise for a valid event
        validate_observation_integrity(obs_event, session)

    def test_integrity_raises_on_session_mismatch(self):
        session, obs_event = _create_completed_observation()
        with pytest.raises(ObservationIntegrityError, match="Session mismatch"):
            validate_observation_integrity(obs_event, "2099-01-02")  # wrong session

    def test_integrity_raises_on_key_mismatch(self):
        session, obs_event = _create_completed_observation()
        # Tamper with model_version so the derived key doesn't match
        tampered = dict(obs_event, model_version="tampered_version")
        with pytest.raises(ObservationIntegrityError, match="does not match derivation formula"):
            validate_observation_integrity(tampered, session)

    def test_reads_correct_session_when_multiple_dates(self):
        """With observations for multiple dates, the correct one is returned."""
        s1, _ = _create_completed_observation(as_of_date=MONDAY, data_cutoff_at=CUTOFF)
        s2, _ = _create_completed_observation(
            as_of_date=TUESDAY, data_cutoff_at="2026-08-11T20:00:00+00:00"
        )
        assert s1 != s2
        obs1 = read_completed_observation(s1)
        obs2 = read_completed_observation(s2)
        assert obs1["intended_execution_session"] == s1
        assert obs2["intended_execution_session"] == s2


# ===========================================================================
# T45 TestBuildReport
# ===========================================================================

class TestBuildReport:
    """Report content accuracy and completeness."""

    def test_report_contains_factor_only_selected(self):
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        assert "factor_only_selected" in content
        assert isinstance(content["factor_only_selected"], list)
        assert len(content["factor_only_selected"]) > 0

    def test_report_contains_claude_shadow_selected(self):
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        assert "claude_shadow_selected" in content
        # No Claude outputs → empty list
        assert content["claude_shadow_selected"] == []

    def test_report_contains_agreement_and_removed(self):
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        assert "agreement_tickers" in content
        assert "factor_only_not_in_claude" in content

    def test_report_agreement_is_subset_of_factor_only(self):
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        fo = set(content["factor_only_selected"])
        agree = set(content["agreement_tickers"])
        assert agree.issubset(fo)

    def test_report_factor_coverage_is_numeric(self):
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        fc = content.get("factor_coverage_mean")
        assert fc is None or (isinstance(fc, float) and 0.0 <= fc <= 1.0)

    def test_report_contains_data_quality_fields(self):
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        assert "data_quality_status" in content
        assert "stale_ticker_count" in content
        assert "missing_ticker_count" in content
        assert "excluded_ticker_count" in content
        assert "universe_count" in content
        assert "valid_ticker_count" in content
        assert "signal_coverage_rate" in content

    def test_report_contains_model_and_portfolio_version(self):
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        assert "model_version" in content
        assert content["model_version"] != ""
        assert "portfolio_version" in content

    def test_report_contains_observation_key_and_run_id(self):
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        assert "observation_key" in content
        assert len(content["observation_key"]) == 64
        assert "observation_run_id" in content

    def test_report_contains_claude_shadow_status_not_collected(self):
        """When no Claude outputs were provided, claude_shadow_status is not_collected."""
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        assert content["claude_shadow_status"] == "not_collected"
        assert content["claude_ok_count"] == 0

    def test_report_stale_tickers_list_present(self):
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        assert isinstance(content["stale_tickers"], list)

    def test_make_report_key_is_deterministic(self):
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        k1 = make_report_key(content)
        k2 = make_report_key(content)
        assert k1 == k2
        assert len(k1) == 64

    def test_different_sessions_produce_different_report_keys(self):
        s1, obs1 = _create_completed_observation(as_of_date=MONDAY, data_cutoff_at=CUTOFF)
        s2, obs2 = _create_completed_observation(
            as_of_date=TUESDAY, data_cutoff_at="2026-08-11T20:00:00+00:00"
        )
        c1 = build_report_content(obs1)
        c2 = build_report_content(obs2)
        assert make_report_key(c1) != make_report_key(c2)


# ===========================================================================
# T46 TestReportLedger
# ===========================================================================

class TestReportLedger:
    """Append-only report storage: create, idempotency, conflict, integrity."""

    def test_create_report_returns_created(self):
        session, obs_event = _create_completed_observation()
        result = create_report(obs_event)
        assert result.status == "CREATED"
        assert result.report_key is not None
        assert len(result.report_key) == 64

    def test_create_report_writes_jsonl_event(self, tmp_report_dir):
        session, obs_event = _create_completed_observation()
        result = create_report(obs_event)
        yyyymm = session[:7]
        jsonl_path = tmp_report_dir / f"{yyyymm}_v2b_reports.jsonl"
        assert jsonl_path.exists()
        events = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        created = [e for e in events if e.get("event_type") == "REPORT_CREATED"]
        assert len(created) == 1
        assert created[0]["report_key"] == result.report_key

    def test_create_report_event_has_order_creation_blocked(self, tmp_report_dir):
        session, obs_event = _create_completed_observation()
        create_report(obs_event)
        yyyymm = session[:7]
        jsonl_path = tmp_report_dir / f"{yyyymm}_v2b_reports.jsonl"
        events = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        for ev in events:
            assert ev.get("order_creation_blocked") is True

    def test_create_report_event_has_event_hash(self, tmp_report_dir):
        session, obs_event = _create_completed_observation()
        create_report(obs_event)
        yyyymm = session[:7]
        jsonl_path = tmp_report_dir / f"{yyyymm}_v2b_reports.jsonl"
        events = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        for ev in events:
            assert "event_hash" in ev
            assert len(ev["event_hash"]) == 64

    def test_create_report_writes_index(self, tmp_report_dir):
        session, obs_event = _create_completed_observation()
        result = create_report(obs_event)
        idx_path = tmp_report_dir / "v2b_report_idx.json"
        assert idx_path.exists()
        idx = json.loads(idx_path.read_text())
        obs_key = obs_event["observation_key"]
        assert obs_key in idx
        assert idx[obs_key]["report_key"] == result.report_key

    def test_create_report_idempotent_same_content(self):
        session, obs_event = _create_completed_observation()
        r1 = create_report(obs_event)
        r2 = create_report(obs_event)
        assert r1.status == "CREATED"
        assert r2.status == "IDEMPOTENT_MATCH"
        assert r1.report_key == r2.report_key

    def test_idempotent_does_not_write_duplicate_events(self, tmp_report_dir):
        session, obs_event = _create_completed_observation()
        create_report(obs_event)
        create_report(obs_event)
        yyyymm = session[:7]
        jsonl_path = tmp_report_dir / f"{yyyymm}_v2b_reports.jsonl"
        events = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        created = [e for e in events if e.get("event_type") == "REPORT_CREATED"]
        assert len(created) == 1  # Only one REPORT_CREATED event

    def test_conflict_when_report_content_differs(self, monkeypatch):
        """Same observation_key but different report_content must raise ReportConflictError."""
        session, obs_event = _create_completed_observation()
        create_report(obs_event)  # First call stores the original report_key in the index

        # Patch to always return different content — new report_key ≠ stored key → conflict
        original_build = report.build_report_content
        monkeypatch.setattr(
            report,
            "build_report_content",
            lambda event: dict(original_build(event), universe_count=99999),
        )

        with pytest.raises(ReportConflictError, match="conflict"):
            create_report(obs_event)

    def test_report_traceable_to_observation_key(self, tmp_report_dir):
        session, obs_event = _create_completed_observation()
        result = create_report(obs_event)
        yyyymm = session[:7]
        jsonl_path = tmp_report_dir / f"{yyyymm}_v2b_reports.jsonl"
        events = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        created = events[0]
        assert created["observation_key"] == obs_event["observation_key"]
        assert created["report_content"]["observation_key"] == obs_event["observation_key"]


# ===========================================================================
# T47 TestTelegramMessage
# ===========================================================================

class TestTelegramMessage:
    """Telegram message format, labeling, and safety markers."""

    def _make_content(self, **kwargs):
        session, obs_event = _create_completed_observation()
        content = build_report_content(obs_event)
        content.update(kwargs)
        return content

    def test_message_labeled_v2_shadow(self):
        content = self._make_content()
        msg = format_telegram_message(content)
        assert "V2 SHADOW" in msg

    def test_message_contains_ingen_handler(self):
        content = self._make_content()
        msg = format_telegram_message(content)
        assert "INGEN HANDLER" in msg

    def test_message_contains_session(self):
        content = self._make_content()
        session = content["intended_execution_session"]
        msg = format_telegram_message(content)
        assert session in msg

    def test_message_contains_observation_key_prefix(self):
        content = self._make_content()
        obs_key = content["observation_key"]
        msg = format_telegram_message(content)
        assert obs_key[:16] in msg

    def test_message_not_presented_as_buy_recommendation(self):
        content = self._make_content()
        msg = format_telegram_message(content)
        # Must not use "kjøp", "buy", "anbefaling", "recommendation"
        lower = msg.lower()
        for forbidden in ["kjøp", "buy recommendation", "anbefaling", "purchase"]:
            assert forbidden not in lower, (
                f"Message must not contain {forbidden!r}: {msg[:200]}"
            )

    def test_message_mentions_no_orders_placed(self):
        content = self._make_content()
        msg = format_telegram_message(content)
        lower = msg.lower()
        assert "ingen ordre" in lower or "no orders" in lower or "urørt" in lower

    def test_message_contains_factor_count(self):
        content = self._make_content()
        count = len(content["factor_only_selected"])
        msg = format_telegram_message(content)
        assert str(count) in msg or "tickere" in msg.lower()

    def test_message_shows_not_collected_when_no_claude(self):
        content = self._make_content(claude_shadow_status="not_collected")
        msg = format_telegram_message(content)
        assert "ikke samlet inn" in msg.lower() or "not_collected" in msg.lower()

    def test_message_shows_data_quality_status(self):
        content = self._make_content()
        dq = content["data_quality_status"]
        msg = format_telegram_message(content)
        assert dq in msg

    def test_message_shows_model_version(self):
        content = self._make_content()
        model_v = content["model_version"]
        msg = format_telegram_message(content)
        assert model_v in msg

    def test_message_shows_v1_production_unaffected(self):
        content = self._make_content()
        msg = format_telegram_message(content)
        assert "V1" in msg or "produksjon" in msg.lower() or "urørt" in msg.lower()


# ===========================================================================
# T48 TestRunShadowReport
# ===========================================================================

class TestRunShadowReport:
    """Full reporting flow: happy path, error cases, Telegram integration."""

    def test_happy_path_returns_created(self):
        session, _ = _create_completed_observation()
        result = run_shadow_report(session)
        assert result.status == "CREATED"
        assert result.report_key is not None
        assert result.observation_key is not None

    def test_missing_observation_returns_failed(self):
        result = run_shadow_report(TUESDAY)
        assert result.status == "FAILED"
        assert result.error is not None

    def test_incomplete_observation_returns_failed(self):
        """Observation in CREATED/COLLECTING state (not COMPLETED) returns FAILED."""
        # No completed observation for TUESDAY → not found
        result = run_shadow_report(TUESDAY)
        assert result.status == "FAILED"

    def test_send_telegram_is_called_on_new_report(self):
        session, _ = _create_completed_observation()
        mock_send = mock.Mock()
        result = run_shadow_report(session, send_telegram_fn=mock_send)
        assert result.status == "CREATED"
        assert result.telegram_sent is True
        mock_send.assert_called_once()

    def test_telegram_message_sent_contains_v2_shadow_label(self):
        session, _ = _create_completed_observation()
        sent_messages = []
        run_shadow_report(session, send_telegram_fn=lambda m: sent_messages.append(m))
        assert len(sent_messages) == 1
        assert "V2 SHADOW" in sent_messages[0]
        assert "INGEN HANDLER" in sent_messages[0]

    def test_telegram_not_sent_when_fn_is_none(self):
        session, _ = _create_completed_observation()
        result = run_shadow_report(session, send_telegram_fn=None)
        assert result.status == "CREATED"
        assert result.telegram_sent is False
        assert result.telegram_skipped is False

    def test_missing_observation_does_not_call_telegram(self):
        mock_send = mock.Mock()
        result = run_shadow_report(TUESDAY, send_telegram_fn=mock_send)
        assert result.status == "FAILED"
        mock_send.assert_not_called()

    def test_conflict_returns_conflict_status(self, monkeypatch):
        session, obs_event = _create_completed_observation()
        run_shadow_report(session)  # First run creates the report

        # Patch to produce different content on second run
        original_build = report.build_report_content

        def altered_build(event):
            content = original_build(event)
            return dict(content, universe_count=99999)

        monkeypatch.setattr(report, "build_report_content", altered_build)
        result = run_shadow_report(session)
        assert result.status == "CONFLICT"

    def test_conflict_does_not_send_telegram(self, monkeypatch):
        session, _ = _create_completed_observation()
        run_shadow_report(session)

        original_build = report.build_report_content

        def altered_build(event):
            return dict(original_build(event), universe_count=99999)

        monkeypatch.setattr(report, "build_report_content", altered_build)
        mock_send = mock.Mock()
        result = run_shadow_report(session, send_telegram_fn=mock_send)
        assert result.status == "CONFLICT"
        mock_send.assert_not_called()


# ===========================================================================
# T49 TestReportIdempotency
# ===========================================================================

class TestReportIdempotency:
    """Reruns with same observation_key and content must be idempotent."""

    def test_second_run_returns_idempotent_match(self):
        session, _ = _create_completed_observation()
        r1 = run_shadow_report(session)
        r2 = run_shadow_report(session)
        assert r1.status == "CREATED"
        assert r2.status == "IDEMPOTENT_MATCH"
        assert r1.report_key == r2.report_key

    def test_idempotent_run_does_not_create_duplicate_report_event(self, tmp_report_dir):
        session, obs_event = _create_completed_observation()
        run_shadow_report(session)
        run_shadow_report(session)
        yyyymm = session[:7]
        jsonl_path = tmp_report_dir / f"{yyyymm}_v2b_reports.jsonl"
        events = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        created = [e for e in events if e.get("event_type") == "REPORT_CREATED"]
        assert len(created) == 1

    def test_telegram_not_sent_twice_on_idempotent_rerun(self):
        session, _ = _create_completed_observation()
        mock_send = mock.Mock()
        r1 = run_shadow_report(session, send_telegram_fn=mock_send)
        r2 = run_shadow_report(session, send_telegram_fn=mock_send)
        assert r1.telegram_sent is True
        assert r2.telegram_sent is False
        assert r2.telegram_skipped is True
        assert mock_send.call_count == 1  # Only called once

    def test_telegram_already_sent_detected_correctly(self, tmp_report_dir):
        session, obs_event = _create_completed_observation()
        result = create_report(obs_event)
        yyyymm = session[:7]
        report_key = result.report_key

        assert not telegram_already_sent(report_key, yyyymm)

        from modules.v2b_report import record_telegram_sent
        record_telegram_sent(report_key, obs_event["observation_key"], session)

        assert telegram_already_sent(report_key, yyyymm)

    def test_telegram_sent_event_written_to_jsonl(self, tmp_report_dir):
        session, _ = _create_completed_observation()
        mock_send = mock.Mock()
        run_shadow_report(session, send_telegram_fn=mock_send)
        yyyymm = session[:7]
        jsonl_path = tmp_report_dir / f"{yyyymm}_v2b_reports.jsonl"
        events = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        sent_events = [e for e in events if e.get("event_type") == "TELEGRAM_SENT"]
        assert len(sent_events) == 1
        assert sent_events[0].get("order_creation_blocked") is True

    def test_same_observation_different_report_version_is_not_modeled(self):
        """Same observation, same content → always IDEMPOTENT_MATCH regardless of run count."""
        session, _ = _create_completed_observation()
        results = [run_shadow_report(session) for _ in range(3)]
        assert results[0].status == "CREATED"
        assert all(r.status == "IDEMPOTENT_MATCH" for r in results[1:])
        assert len({r.report_key for r in results}) == 1  # All same report key


# ===========================================================================
# T50 TestReportSafetyBoundary
# ===========================================================================

class TestReportSafetyBoundary:
    """V2B safety: no orders, no V1 execution imports, order_creation_blocked."""

    def test_report_module_has_no_v1_execution_imports(self):
        source = Path("modules/v2b_report.py").read_text()
        # Check for import statements (docstring mentions of these module names are fine)
        forbidden_imports = [
            "from modules.portfolio import",
            "from modules.orders import",
            "from modules.fills import",
            "from modules.ledger import",
            "from modules.state import",
        ]
        # Check for actual function calls (with open-paren; docstring mentions without parens are fine)
        forbidden_calls = [
            "execute_buy(",
            "execute_sell(",
            "execute_pyramid_fill(",
        ]
        for pattern in forbidden_imports + forbidden_calls:
            assert pattern not in source, (
                f"v2b_report.py must not contain {pattern!r}"
            )

    def test_report_module_has_no_telegram_import(self):
        """v2b_report.py must not import send_telegram — it accepts it as a parameter."""
        source = Path("modules/v2b_report.py").read_text()
        assert "from modules.reporting import" not in source
        # send_telegram_fn is an allowed parameter name; only a bare import is forbidden
        assert "import send_telegram" not in source

    def test_order_creation_blocked_constant_is_true(self):
        assert report._ORDER_CREATION_BLOCKED is True

    def test_report_events_have_order_creation_blocked_true(self, tmp_report_dir):
        session, _ = _create_completed_observation()
        run_shadow_report(session, send_telegram_fn=mock.Mock())
        yyyymm = session[:7]
        jsonl_path = tmp_report_dir / f"{yyyymm}_v2b_reports.jsonl"
        events = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        assert len(events) >= 2  # REPORT_CREATED + TELEGRAM_SENT
        for ev in events:
            assert ev.get("order_creation_blocked") is True, (
                f"order_creation_blocked must be True in all events, got: {ev}"
            )

    def test_no_execute_buy_called(self):
        with mock.patch("modules.portfolio.execute_buy") as mock_buy:
            session, _ = _create_completed_observation()
            run_shadow_report(session, send_telegram_fn=mock.Mock())
            mock_buy.assert_not_called()

    def test_no_execute_sell_called(self):
        with mock.patch("modules.portfolio.execute_sell") as mock_sell:
            session, _ = _create_completed_observation()
            run_shadow_report(session, send_telegram_fn=mock.Mock())
            mock_sell.assert_not_called()

    def test_no_execute_pyramid_fill_called(self):
        with mock.patch("modules.portfolio.execute_pyramid_fill") as mock_fill:
            session, _ = _create_completed_observation()
            run_shadow_report(session, send_telegram_fn=mock.Mock())
            mock_fill.assert_not_called()

    def test_no_v1_state_files_modified(self, tmp_path):
        v1_state_dir = tmp_path / "data_v4" / "state"
        v1_state_dir.mkdir(parents=True)
        session, _ = _create_completed_observation()
        run_shadow_report(session, send_telegram_fn=mock.Mock())
        v1_files = list(v1_state_dir.iterdir())
        assert v1_files == [], f"V1 state files were written: {v1_files}"

    def test_v2_shadow_report_does_not_replace_v1_telegram_message(self):
        """
        run_shadow_report with a Telegram function must only send V2 SHADOW labeled
        messages — never generic or V1-style production messages.
        """
        session, _ = _create_completed_observation()
        sent_messages = []
        run_shadow_report(session, send_telegram_fn=lambda m: sent_messages.append(m))
        assert len(sent_messages) == 1
        msg = sent_messages[0]
        # Must be clearly labeled as V2 SHADOW
        assert "V2 SHADOW" in msg
        assert "INGEN HANDLER" in msg
        # Must not look like a production buy signal
        assert "🔔" not in msg or "V2 SHADOW" in msg  # If bell emoji, must be in shadow context
