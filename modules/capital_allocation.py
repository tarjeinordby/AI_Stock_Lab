"""
Score-proportional capital allocation that invests everything the limits allow.

NOT wired into the live code yet (natt-steg1, awaiting approval).

Rule:
  1. Every candidate with a positive score gets a weight proportional to its score.
  2. No single position may exceed ``max_weight`` of total capital.
  3. No sector may exceed ``sector_cap`` of total capital.
  4. Whatever a capped name or sector cannot take is redistributed to the
     remaining names, again in proportion to score (water-filling), until either
     all capital is placed or every name is at a limit. Inside a capped sector
     the names share the sector cap in proportion to score (still respecting
     ``max_weight``).
  5. Optionally, positions smaller than ``min_position_value`` are dropped
     (lowest score first) and the allocation is recomputed.

Cash is only left over when the limits make full investment impossible,
e.g. 2 names with ``max_weight=0.40`` can hold at most 80 %.
"""

from __future__ import annotations

_EPS = 1e-12
_BISECT_STEPS = 200


def _solve_level(f, target: float, hi: float) -> float:
    """Smallest level x in [0, hi] with f(x) >= target (f non-decreasing)."""
    lo = 0.0
    for _ in range(_BISECT_STEPS):
        mid = (lo + hi) / 2
        if f(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def _capped(scores: dict[str, float], level: float, max_weight: float) -> dict[str, float]:
    return {t: min(max_weight, level * s) for t, s in scores.items()}


def _water_fill(scores: dict[str, float], sectors: dict[str, str],
                max_weight: float, sector_cap: float | None) -> dict[str, float]:
    """
    Each name gets ``level * score`` capped at ``max_weight``; a sector whose
    names would exceed ``sector_cap`` gets its own lower level so it lands
    exactly on the cap. The common level is raised until all capital is placed
    or every name/sector is at its limit.
    """
    groups: dict[str, dict[str, float]] = {}
    for t, s in scores.items():
        groups.setdefault(sectors.get(t, "Unknown"), {})[t] = s
    cap = sector_cap if sector_cap is not None else float("inf")

    def group_weight(g: dict[str, float], level: float) -> float:
        return min(cap, sum(_capped(g, level, max_weight).values()))

    def total(level: float) -> float:
        return sum(group_weight(g, level) for g in groups.values())

    # At hi every name sits at max_weight, so total(hi) is the most we can invest.
    hi = max_weight / min(scores.values())
    level = _solve_level(total, min(1.0, total(hi)), hi)

    weights: dict[str, float] = {}
    for g in groups.values():
        w = _capped(g, level, max_weight)
        if sum(w.values()) > cap + _EPS:
            sector_level = _solve_level(
                lambda x: sum(_capped(g, x, max_weight).values()), cap, level)
            w = _capped(g, sector_level, max_weight)
        weights.update(w)
    return weights


def allocate_capital(
    scores: dict[str, float],
    capital: float,
    max_weight: float,
    sectors: dict[str, str] | None = None,
    sector_cap: float | None = None,
    min_position_value: float = 0.0,
) -> dict:
    """
    Split ``capital`` across ``scores`` (ticker -> score) under the limits.

    max_weight  — max share of total capital in one ticker (0 < max_weight <= 1)
    sectors     — ticker -> sector; tickers without a sector count as "Unknown"
    sector_cap  — max share of total capital in one sector, or None for no cap
    min_position_value — drop positions smaller than this (in currency units)

    Returns {
        "weights":  {ticker: share of capital},
        "amounts":  {ticker: currency amount},
        "cash_weight", "cash", "invested_fraction",
    }
    """
    if capital < 0:
        raise ValueError("capital must be >= 0")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if sector_cap is not None and not 0 < sector_cap <= 1:
        raise ValueError("sector_cap must be in (0, 1] or None")

    sectors = sectors or {}
    candidates = {t: float(s) for t, s in scores.items() if s is not None and float(s) > 0}

    while True:
        weights = _water_fill(candidates, sectors, max_weight, sector_cap) if candidates else {}
        too_small = [t for t, w in weights.items()
                     if w > 0 and w * capital < min_position_value - _EPS]
        if not too_small:
            break
        # Drop the weakest undersized name and recompute.
        del candidates[min(too_small, key=lambda t: (candidates[t], t))]

    weights = {t: w for t, w in weights.items() if w > _EPS}
    invested = min(sum(weights.values()), 1.0)
    return {
        "weights": weights,
        "amounts": {t: w * capital for t, w in weights.items()},
        "cash_weight": 1.0 - invested,
        "cash": capital * (1.0 - invested),
        "invested_fraction": invested,
    }
