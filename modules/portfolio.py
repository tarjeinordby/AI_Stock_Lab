import pandas as pd

from modules.market_data import get_price_cached
from modules.state import now_utc, safe_float, safe_round, today_str, trading_days_between

EXECUTION_PRICE_SOURCE = "next_session_daily_open_v1"


MIN_POSITION_VALUE = 500
SECTOR_CAP = 0.25


def estimate_trade_cost(value, side):
    cost = abs(value) * 0.001
    if side == "SELL":
        cost += abs(value) * 0.00003
    return cost


def current_portfolio_value(state, candidates_by_ticker, price_cache):
    positions_value = 0.0
    for ticker, pos in state["positions"].items():
        fallback = pos.get("last_price", pos.get("avg_price"))
        candidate_price = candidates_by_ticker.get(ticker, {}).get("price", fallback)
        price = get_price_cached(ticker, candidate_price, price_cache)
        if price is None:
            price = fallback
        shares = float(pos["shares"])
        value = shares * price
        pos["last_price"] = price
        pos["market_value"] = value
        pos["highest_price"] = max(float(pos.get("highest_price", price)), price)
        positions_value += value

    total = float(state["cash"]) + positions_value
    return total, positions_value, state


def build_target_weights(candidates, config, regime_name, portfolio_drawdown=0.0, macro_mult=1.0):
    if not candidates:
        return {}

    df = pd.DataFrame(candidates).copy()
    df = df[df["above_sma200"] == True].copy()
    df = df[df["strategy_score"] > 0].copy()

    if df.empty:
        return {}

    df = df.sort_values("strategy_score", ascending=False).head(config["buy_top_n"])
    df = df[df["score_percentile"] >= config["min_score_percentile"]].copy()

    if df.empty:
        return {}

    # Risk parity: score² / vol60
    df["vol60"] = df["vol60"].clip(lower=0.15, upper=1.50)
    df["raw_weight"] = (df["strategy_score"] ** 2) / df["vol60"]

    if df["raw_weight"].sum() <= 0:
        return {}

    df["target_weight"] = df["raw_weight"] / df["raw_weight"].sum()
    df["target_weight"] = df["target_weight"].clip(upper=config["max_position_weight"])

    # Sector cap at 25%
    if "sector" in df.columns:
        sector_totals = df.groupby("sector")["target_weight"].sum()
        for sector, total in sector_totals.items():
            if total > SECTOR_CAP:
                scale = SECTOR_CAP / total
                df.loc[df["sector"] == sector, "target_weight"] *= scale

    # Re-normalize to 1.0
    total_w = df["target_weight"].sum()
    if total_w > 0:
        df["target_weight"] = df["target_weight"] / total_w

    # Apply regime exposure (overridden by portfolio drawdown protection + macro mult)
    base_exposure = config["exposure"].get(regime_name, config["exposure"].get("unknown", 0.50))
    effective_exposure = _get_effective_exposure(base_exposure, portfolio_drawdown) * max(0.0, min(1.0, macro_mult))
    df["target_weight"] *= effective_exposure

    return dict(zip(df["ticker"], df["target_weight"]))


def _get_effective_exposure(base_exposure, drawdown):
    if drawdown <= -0.30:
        return min(base_exposure, 0.20)
    if drawdown <= -0.20:
        return min(base_exposure, 0.50)
    return base_exposure


def execute_sell(
    state, trades_df, strategy_name, ticker, reason, execution_price_cache
):
    import pandas as pd

    pos = state["positions"].get(ticker)
    if not pos:
        return state, trades_df, None

    if ticker not in execution_price_cache:
        print(f"  {ticker}: ingen gyldig åpningspris for session — SELG utsatt")
        return state, trades_df, None
    price = execution_price_cache[ticker]

    shares = float(pos["shares"])
    value = shares * price
    cost = estimate_trade_cost(value, "SELL")
    net_value = value - cost

    state["cash"] += net_value

    trade = {
        "date": today_str(),
        "timestamp_utc": now_utc().isoformat(),
        "strategy": strategy_name,
        "action": "SELL",
        "ticker": ticker,
        "shares": round(shares, 6),
        "price": round(price, 2),
        "value": round(value, 2),
        "cost": round(cost, 2),
        "reason": reason,
        "execution_session": today_str(),
        "execution_price_source": EXECUTION_PRICE_SOURCE,
        "order_status": "filled",
    }

    trades_df = pd.concat([trades_df, pd.DataFrame([trade])], ignore_index=True)
    del state["positions"][ticker]
    state["cooldowns"][ticker] = today_str()

    return state, trades_df, trade


def execute_buy(
    state, trades_df, strategy_name, ticker, target_value, reason, candidate,
    execution_price_cache, premarket_move=None, pyramid_remaining=0.0,
):
    import pandas as pd

    if ticker not in execution_price_cache:
        print(f"  {ticker}: ingen gyldig åpningspris for session — KJØP utsatt")
        return state, trades_df, None
    price = execution_price_cache[ticker]

    signal_price = candidate.get("price")

    # Pre-market filter: halve size if moved >4% since yesterday's close
    if premarket_move is not None and abs(premarket_move) > 0.04:
        target_value *= 0.50
        arrow = "↑" if premarket_move > 0 else "↓"
        reason = f"{reason} [PM {premarket_move:+.1%}{arrow} → 50%]"

    # Gap check: open price vs yesterday's close (overnight gap)
    if signal_price and signal_price > 0:
        gap = (price / signal_price) - 1
        if gap > 0.05:
            print(f"  {ticker}: åpnings-gap {gap:.1%} fra signal — hopper over")
            return state, trades_df, None
        if gap > 0.025:
            target_value *= 0.50

    target_value = min(target_value, state["cash"])
    if target_value < MIN_POSITION_VALUE:
        return state, trades_df, None

    cost = estimate_trade_cost(target_value, "BUY")
    total_needed = target_value + cost

    if total_needed > state["cash"]:
        target_value = state["cash"] * 0.995
        cost = estimate_trade_cost(target_value, "BUY")
        total_needed = target_value + cost

    if target_value < MIN_POSITION_VALUE:
        return state, trades_df, None

    shares = target_value / price
    if shares <= 0:
        return state, trades_df, None

    state["cash"] -= total_needed
    is_partial = pyramid_remaining > 0
    state["positions"][ticker] = {
        "shares": shares,
        "avg_price": price,
        "last_price": price,
        "highest_price": price,
        "market_value": shares * price,
        "buy_date": today_str(),
        "buy_score": round(safe_float(candidate.get("strategy_score"), 0.0), 1),
        "is_partial": is_partial,
        "pyramid_remaining_value": round(pyramid_remaining, 2) if is_partial else 0.0,
        "pyramid_min_price": round(price * 1.02, 4) if is_partial else 0.0,
    }
    state["weekly_meta"]["buys_this_week"] += 1

    trade = {
        "date": today_str(),
        "timestamp_utc": now_utc().isoformat(),
        "strategy": strategy_name,
        "action": "BUY",
        "ticker": ticker,
        "shares": round(shares, 6),
        "price": round(price, 2),
        "value": round(target_value, 2),
        "cost": round(cost, 2),
        "reason": reason,
        "execution_session": today_str(),
        "execution_price_source": EXECUTION_PRICE_SOURCE,
        "order_status": "filled",
    }

    trades_df = pd.concat([trades_df, pd.DataFrame([trade])], ignore_index=True)
    return state, trades_df, trade


def execute_pyramid_fill(state, trades_df, strategy_name, ticker, execution_price_cache):
    """
    Complete a partial position: buy the remaining 40% if:
    - Position is marked is_partial=True
    - At least 5 trading days since buy_date
    - Open price >= pyramid_min_price (entry × 1.02)
    - Valid execution price available for today's session
    """
    pos = state["positions"].get(ticker)
    if not pos or not pos.get("is_partial"):
        return state, trades_df, None

    days_held = trading_days_between(pos.get("buy_date"))
    if days_held < 5:
        return state, trades_df, None

    remaining = float(pos.get("pyramid_remaining_value", 0))
    if remaining < MIN_POSITION_VALUE:
        pos["is_partial"] = False
        pos["pyramid_remaining_value"] = 0.0
        return state, trades_df, None

    if ticker not in execution_price_cache:
        return state, trades_df, None
    price = execution_price_cache[ticker]

    min_price = float(pos.get("pyramid_min_price", 0))
    if price < min_price:
        return state, trades_df, None

    # Cap to available cash
    remaining = min(remaining, state["cash"] * 0.995)
    cost = estimate_trade_cost(remaining, "BUY")
    if remaining + cost > state["cash"]:
        remaining = state["cash"] * 0.995 - cost
    if remaining < MIN_POSITION_VALUE:
        return state, trades_df, None

    new_shares = remaining / price
    old_shares = float(pos["shares"])
    old_avg = float(pos["avg_price"])
    total_shares = old_shares + new_shares
    new_avg = (old_shares * old_avg + remaining) / total_shares
    entry_pct = (price / old_avg - 1) * 100

    state["cash"] -= remaining + cost
    pos["shares"] = total_shares
    pos["avg_price"] = new_avg
    pos["market_value"] = total_shares * price
    pos["last_price"] = price
    pos["highest_price"] = max(float(pos.get("highest_price", price)), price)
    pos["is_partial"] = False
    pos["pyramid_remaining_value"] = 0.0

    trade = {
        "date": today_str(),
        "timestamp_utc": now_utc().isoformat(),
        "strategy": strategy_name,
        "action": "BUY",
        "ticker": ticker,
        "shares": round(new_shares, 6),
        "price": round(price, 2),
        "value": round(remaining, 2),
        "cost": round(cost, 2),
        "reason": f"Pyramidering 40% ({entry_pct:+.1f}% fra entry, dag {days_held})",
        "execution_session": today_str(),
        "execution_price_source": EXECUTION_PRICE_SOURCE,
        "order_status": "filled",
    }

    trades_df = pd.concat([trades_df, pd.DataFrame([trade])], ignore_index=True)
    return state, trades_df, trade
