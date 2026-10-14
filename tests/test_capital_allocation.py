"""Tests for modules/capital_allocation.allocate_capital (not yet used live)."""

import pytest

from modules.capital_allocation import allocate_capital


def _approx(d):
    return {k: pytest.approx(v) for k, v in d.items()}


class TestFullInvestment:
    def test_scores_90_8_2_with_40pct_cap_is_fully_invested(self):
        r = allocate_capital({"A": 90, "B": 8, "C": 2}, capital=10_000, max_weight=0.40)
        assert r["invested_fraction"] == pytest.approx(1.0)
        assert r["weights"] == _approx({"A": 0.40, "B": 0.40, "C": 0.20})
        assert r["cash"] == pytest.approx(0.0)

    def test_uncapped_is_score_proportional(self):
        r = allocate_capital({"A": 3, "B": 1}, capital=1000, max_weight=1.0)
        assert r["weights"] == _approx({"A": 0.75, "B": 0.25})
        assert r["amounts"] == _approx({"A": 750, "B": 250})

    def test_amounts_sum_to_capital(self):
        scores = {f"T{i}": 100 - i * 7 for i in range(12)}
        r = allocate_capital(scores, capital=12_345, max_weight=0.15)
        assert sum(r["amounts"].values()) == pytest.approx(12_345)

    def test_no_weight_exceeds_cap(self):
        scores = {f"T{i}": 2 ** (10 - i) for i in range(10)}
        r = allocate_capital(scores, capital=1, max_weight=0.12)
        assert max(r["weights"].values()) <= 0.12 + 1e-9
        assert r["invested_fraction"] == pytest.approx(1.0)


class TestLimitsLeaveCashOnlyWhenUnavoidable:
    def test_too_few_names_for_cap(self):
        r = allocate_capital({"A": 50, "B": 50}, capital=1000, max_weight=0.40)
        assert r["invested_fraction"] == pytest.approx(0.80)
        assert r["cash"] == pytest.approx(200)

    def test_empty_scores_all_cash(self):
        r = allocate_capital({}, capital=500, max_weight=0.4)
        assert r["weights"] == {}
        assert r["cash"] == 500

    def test_non_positive_scores_ignored(self):
        r = allocate_capital({"A": 10, "B": 0, "C": -5, "D": None}, capital=100, max_weight=1.0)
        assert r["weights"] == _approx({"A": 1.0})


class TestSectorCap:
    def test_sector_cap_is_share_of_total_capital(self):
        scores = {"N1": 40, "N2": 40, "E1": 10, "E2": 10}
        sectors = {"N1": "Tech", "N2": "Tech", "E1": "Energy", "E2": "Health"}
        r = allocate_capital(scores, capital=1000, max_weight=0.5,
                             sectors=sectors, sector_cap=0.40)
        w = r["weights"]
        assert w["N1"] + w["N2"] == pytest.approx(0.40)
        assert r["invested_fraction"] == pytest.approx(1.0)
        assert w["E1"] == pytest.approx(0.30)
        assert w["E2"] == pytest.approx(0.30)

    def test_sector_and_stock_caps_together(self):
        scores = {"T1": 50, "T2": 30, "T3": 20, "H1": 5, "F1": 5}
        sectors = {"T1": "Tech", "T2": "Tech", "T3": "Tech", "H1": "Health", "F1": "Fin"}
        r = allocate_capital(scores, capital=1, max_weight=0.35,
                             sectors=sectors, sector_cap=0.50)
        w = r["weights"]
        assert w["T1"] + w["T2"] + w["T3"] == pytest.approx(0.50)
        assert max(w.values()) <= 0.35 + 1e-9
        # Health and Fin can take 0.35 each but only 0.50 is left
        assert w["H1"] == pytest.approx(0.25)
        assert w["F1"] == pytest.approx(0.25)
        assert r["invested_fraction"] == pytest.approx(1.0)
        # inside Tech the 0.50 is split 50:30:20
        assert w["T1"] == pytest.approx(0.25)
        assert w["T2"] == pytest.approx(0.15)
        assert w["T3"] == pytest.approx(0.10)

    def test_stock_capped_names_never_breach_sector_cap(self):
        # Both Tech names would hit the 20 % stock cap (40 % together) but the
        # sector may only hold 30 %.
        scores = {"T1": 50, "T2": 50, "A": 1, "B": 1, "C": 1, "D": 1}
        sectors = {"T1": "Tech", "T2": "Tech", "A": "a", "B": "b", "C": "c", "D": "d"}
        r = allocate_capital(scores, capital=1, max_weight=0.20,
                             sectors=sectors, sector_cap=0.30)
        w = r["weights"]
        assert w["T1"] + w["T2"] == pytest.approx(0.30)
        assert w["T1"] == pytest.approx(0.15)
        assert all(w[t] == pytest.approx(0.175) for t in "ABCD")
        assert r["invested_fraction"] == pytest.approx(1.0)

    def test_single_sector_limited_by_sector_cap(self):
        scores = {"A": 1, "B": 1, "C": 1}
        sectors = {"A": "Tech", "B": "Tech", "C": "Tech"}
        r = allocate_capital(scores, capital=900, max_weight=0.5,
                             sectors=sectors, sector_cap=0.60)
        assert r["invested_fraction"] == pytest.approx(0.60)
        assert r["amounts"] == _approx({"A": 180, "B": 180, "C": 180})

    def test_missing_sector_counts_as_unknown(self):
        r = allocate_capital({"A": 1, "B": 1}, capital=1, max_weight=1.0,
                             sectors={}, sector_cap=0.5)
        assert r["invested_fraction"] == pytest.approx(0.5)


class TestMinPositionValue:
    def test_drops_small_positions_and_reinvests(self):
        r = allocate_capital({"A": 90, "B": 9, "C": 1}, capital=1000,
                             max_weight=1.0, min_position_value=50)
        assert "C" not in r["weights"]
        assert r["invested_fraction"] == pytest.approx(1.0)
        assert r["weights"] == _approx({"A": 0.90 / 0.99, "B": 0.09 / 0.99})

    def test_all_too_small_gives_cash(self):
        r = allocate_capital({"A": 1, "B": 1}, capital=100, max_weight=1.0,
                             min_position_value=500)
        assert r["weights"] == {}
        assert r["cash"] == 100


class TestValidation:
    @pytest.mark.parametrize("kwargs", [
        {"max_weight": 0}, {"max_weight": 1.5},
        {"max_weight": 0.5, "sector_cap": 0}, {"max_weight": 0.5, "capital": -1},
    ])
    def test_bad_arguments_raise(self, kwargs):
        args = {"scores": {"A": 1}, "capital": 100}
        args.update(kwargs)
        with pytest.raises(ValueError):
            allocate_capital(**args)


class TestRandomizedInvariants:
    def test_limits_hold_and_capital_is_used(self):
        import random
        rng = random.Random(42)
        for _ in range(300):
            n = rng.randint(1, 25)
            scores = {f"T{i}": rng.uniform(0.1, 100) for i in range(n)}
            sectors = {t: rng.choice("ABCDE") for t in scores}
            max_w = rng.uniform(0.05, 1.0)
            sec_cap = rng.choice([None, rng.uniform(0.1, 1.0)])
            r = allocate_capital(scores, 1.0, max_w, sectors, sec_cap)
            w = r["weights"]
            assert max(w.values()) <= max_w + 1e-9
            by_sector = {}
            for t, x in w.items():
                by_sector[sectors[t]] = by_sector.get(sectors[t], 0) + x
            if sec_cap is not None:
                assert max(by_sector.values()) <= sec_cap + 1e-9
            # Most the limits allow: per sector min(cap, n_names * max_w)
            cap = sec_cap if sec_cap is not None else float("inf")
            names_per_sector = {}
            for t in scores:
                names_per_sector[sectors[t]] = names_per_sector.get(sectors[t], 0) + 1
            feasible = sum(min(cap, k * max_w) for k in names_per_sector.values())
            assert r["invested_fraction"] == pytest.approx(min(1.0, feasible), abs=1e-9)
