import pandas as pd

from modules.market_data import get_price_cached
from modules.state import now_utc, safe_float, safe_round, today_str


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


def build_target_weights(candidates, config, regime_name, portfolio_drawdown=0.0):
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

    # Apply regime exposure (overridden by portfolio drawdown protection)
    base_exposure = config["exposure"].get(regime_name, config["exposure"].get("unknown", 0.50))
    effective_exposure = _get_effective_exposure(base_exposure, portfolio_drawdown)
    df["target_weight"] *= effective_exposure

    return dict(zip(df["ticker"], df["target_weight"]))


def _get_effective_exposure(base_exposure, drawdown):
    if drawdown <= -0.30:
        return min(base_exposure, 0.20)
    if drawdown <= -0.20:
        return min(base_exposure, 0.50)
    return base_exposure


def execute_sell(
    state, trades_df, strategy_name, ticker, reason, candidates_by_ticker, price_cache
):
    import pandas as pd

    pos = state["positions"].get(ticker)
    if not pos:
        return state, trades_df, None

    fallback = pos.get("last_price", pos.get("avg_price"))
    signal_price = candidates_by_ticker.get(ticker, {}).get("price", fallback)
    price = get_price_cached(ticker, signal_price, price_cache)
    if price is None or price <= 0:
        return state, trades_df, None

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
    }

    trades_df = pd.concat([trades_df, pd.DataFrame([trade])], ignore_index=True)
    del state["positions"][ticker]
    state["cooldowns"][ticker] = today_str()

    return state, trades_df, trade


def execute_buy(
    state, trades_df, strategy_name, ticker, target_value, reason, candidate, price_cache,
    premarket_move=None,
):
    import pandas as pd

    signal_price = candidate.get("price")
    price = get_price_cached(ticker, signal_price, price_cache)

    if price is None or price <= 0:
        return state, trades_df, None

    # Pre-market filter: halve size if moved >4% since yesterday's close
    if premarket_move is not None and abs(premarket_move) > 0.04:
        target_value *= 0.50
        arrow = "↑" if premarket_move > 0 else "↓"
        reason = f"{reason} [PM {premarket_move:+.1%}{arrow} → 50%]"

    # Gap check vs signal price
    if signal_price and signal_price > 0:
        gap = (price / signal_price) - 1
        if gap > 0.05:
            print(f"  {ticker}: pris gapet {gap:.1%} fra signal — hopper over")
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
    state["positions"][ticker] = {
        "shares": shares,
        "avg_price": price,
        "last_price": price,
        "highest_price": price,
        "market_value": shares * price,
        "buy_date": today_str(),
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
    }

    trades_df = pd.concat([trades_df, pd.DataFrame([trade])], ignore_index=True)
    return state, trades_df, trade
