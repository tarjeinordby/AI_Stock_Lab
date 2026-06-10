from modules.state import START_CAPITAL, safe_float, safe_round


def get_portfolio_drawdown(state, total_value):
    highest = float(state.get("highest_portfolio_value", START_CAPITAL))
    if highest <= 0:
        return 0.0
    return (total_value / highest) - 1


def should_sell_position(ticker, pos, candidate, config):
    if candidate is None:
        return True, "Falt ut av universet"

    price = candidate.get("live_price", candidate.get("price", pos.get("last_price", pos.get("avg_price"))))
    avg_price = safe_float(pos.get("avg_price"), 0)
    highest_price = safe_float(pos.get("highest_price", avg_price), avg_price)

    if price is None or avg_price <= 0:
        return False, ""

    ret_from_entry = (price / avg_price) - 1
    ret_from_high = (price / highest_price) - 1 if highest_price > 0 else 0

    if ret_from_entry <= config["stop_loss"]:
        return True, f"Stop-loss {safe_round(ret_from_entry * 100, 1)}%"

    if ret_from_high <= config["trailing_stop"]:
        return True, f"Trailing stop {safe_round(ret_from_high * 100, 1)}% fra topp"

    if not candidate.get("above_sma200", True):
        return True, "Under 200-dagers snitt"

    rank = int(candidate.get("rank", 999))
    score = safe_float(candidate.get("strategy_score"), 0)
    if rank > config["sell_rank_threshold"] and score < 45:
        return True, f"Svekket ranking (#{rank}, score {safe_round(score, 1)})"

    return False, ""


def drawdown_protection_message(drawdown):
    if drawdown <= -0.30:
        return f"DRAWDOWN BESKYTTELSE: {drawdown:.1%} fra topp → maks 20% investert"
    if drawdown <= -0.20:
        return f"DRAWDOWN ADVARSEL: {drawdown:.1%} fra topp → maks 50% investert"
    return None


def spy_is_recovering(spy_candidate):
    if spy_candidate is None:
        return False
    return bool(spy_candidate.get("above_sma50", False))
