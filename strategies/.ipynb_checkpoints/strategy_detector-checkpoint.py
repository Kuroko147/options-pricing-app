import streamlit as st

def detect_strategy(legs):
    if len(legs) != 2:
        return None

    l1, l2 = legs[0], legs[1]

    # Booleans for leg 1
    l1_long_call = (l1['type'] == 'call' and l1['position'] == 1)
    l1_short_call = (l1['type'] == 'call' and l1['position'] == -1)
    l1_long_put = (l1['type'] == 'put' and l1['position'] == 1)
    l1_short_put = (l1['type'] == 'put' and l1['position'] == -1)

    # Booleans for leg 2
    l2_long_call = (l2['type'] == 'call' and l2['position'] == 1)
    l2_short_call = (l2['type'] == 'call' and l2['position'] == -1)
    l2_long_put = (l2['type'] == 'put' and l2['position'] == 1)
    l2_short_put = (l2['type'] == 'put' and l2['position'] == -1)

    # Bull Call Spread: Long call lower strike, Short call higher strike
    if (l1_long_call and l2_short_call and l1['strike'] < l2['strike']) or \
       (l2_long_call and l1_short_call and l2['strike'] < l1['strike']):
        return "Bull Call Spread"

    # Bear Put Spread: Long put higher strike, Short put lower strike
    if (l1_long_put and l2_short_put and l1['strike'] > l2['strike']) or \
       (l2_long_put and l1_short_put and l2['strike'] > l1['strike']):
        return "Bear Put Spread"

    # Long Straddle: Long call + Long put at same strike
    if ((l1_long_call and l2_long_put) or (l2_long_call and l1_long_put)) and \
       abs(l1['strike'] - l2['strike']) < 1e-5:
        return "Long Straddle"

    # Short Straddle: Short call + Short put at same strike
    if ((l1_short_call and l2_short_put) or (l2_short_call and l1_short_put)) and \
       abs(l1['strike'] - l2['strike']) < 1e-5:
        return "Short Straddle"

    # Long Strangle: Long call higher strike + Long put lower strike
    if (l1_long_call and l2_long_put and l1['strike'] > l2['strike']) or \
       (l2_long_call and l1_long_put and l2['strike'] > l1['strike']):
        return "Long Strangle"

    # Short Strangle: Short call higher strike + Short put lower strike
    if (l1_short_call and l2_short_put and l1['strike'] > l2['strike']) or \
       (l2_short_call and l1_short_put and l2['strike'] > l1['strike']):
        return "Short Strangle"

    # Covered Call: Long stock + Short call (We can't detect stock position here, so skip)

    # Protective Put: Long stock + Long put (Skip as above)

    # Calendar Spread: Same strike, different expiries (we don't have expiry here, so skip)

    # Vertical Put Spread (Bull Put Spread): Short put lower strike, Long put higher strike
    if (l1_short_put and l2_long_put and l1['strike'] < l2['strike']) or \
       (l2_short_put and l1_long_put and l2['strike'] < l1['strike']):
        return "Bull Put Spread"

    # Vertical Call Spread (Bear Call Spread): Short call lower strike, Long call higher strike
    if (l1_short_call and l2_long_call and l1['strike'] < l2['strike']) or \
       (l2_short_call and l1_long_call and l2['strike'] < l1['strike']):
        return "Bear Call Spread"

    return None


def analyze_strategy(legs):
    """
    Analyzes the given 2-leg option strategy.
    Returns a dict with net premium, max profit, max loss, breakeven(s), and strategy name,
    or None if strategy not detected or unsupported.
    """

    if len(legs) != 2:
        return None

    # Calculate net premium: positive if received, negative if paid
    net_premium = sum(leg['premium'] * -leg['position'] for leg in legs)

    strategy = detect_strategy(legs)
    if not strategy:
        return None

    result = {"strategy": strategy, "net_premium": net_premium}

    net_premium_received = max(net_premium, 0)
    net_premium_paid = -min(net_premium, 0)

    if strategy == "Bull Call Spread":
        long_call = next(leg for leg in legs if leg['position'] == 1 and leg['type'] == 'call')
        short_call = next(leg for leg in legs if leg['position'] == -1 and leg['type'] == 'call')

        max_profit = short_call['strike'] - long_call['strike'] - net_premium_paid
        max_loss = net_premium_paid
        breakeven = long_call['strike'] + net_premium_paid

        result.update({
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": [breakeven]
        })

    elif strategy == "Bear Put Spread":
        long_put = next(leg for leg in legs if leg['position'] == 1 and leg['type'] == 'put')
        short_put = next(leg for leg in legs if leg['position'] == -1 and leg['type'] == 'put')

        max_profit = long_put['strike'] - short_put['strike'] - net_premium_paid
        max_loss = net_premium_paid
        breakeven = long_put['strike'] - net_premium_paid

        result.update({
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": [breakeven]
        })

    elif strategy == "Long Straddle":
        long_call = next(leg for leg in legs if leg['position'] == 1 and leg['type'] == 'call')
        long_put = next(leg for leg in legs if leg['position'] == 1 and leg['type'] == 'put')

        strike = long_call['strike']

        breakeven_low = strike - net_premium_paid
        breakeven_high = strike + net_premium_paid
        max_loss = net_premium_paid  # limited to premium paid

        result.update({
            "max_profit": float('inf'),  # unlimited upside
            "max_loss": max_loss,
            "breakeven": [breakeven_low, breakeven_high]
        })

    elif strategy == "Short Straddle":
        short_call = next(leg for leg in legs if leg['position'] == -1 and leg['type'] == 'call')
        short_put = next(leg for leg in legs if leg['position'] == -1 and leg['type'] == 'put')

        strike = short_call['strike']

        breakeven_low = strike - net_premium_received
        breakeven_high = strike + net_premium_received
        max_profit = net_premium_received
        max_loss = float('inf')  # unlimited risk

        result.update({
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": [breakeven_low, breakeven_high]
        })

    elif strategy == "Long Strangle":
        long_call = next(leg for leg in legs if leg['position'] == 1 and leg['type'] == 'call')
        long_put = next(leg for leg in legs if leg['position'] == 1 and leg['type'] == 'put')

        breakeven_low = long_put['strike'] - net_premium_paid
        breakeven_high = long_call['strike'] + net_premium_paid
        max_loss = net_premium_paid
        max_profit = float('inf')

        result.update({
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": [breakeven_low, breakeven_high]
        })

    elif strategy == "Short Strangle":
        short_call = next(leg for leg in legs if leg['position'] == -1 and leg['type'] == 'call')
        short_put = next(leg for leg in legs if leg['position'] == -1 and leg['type'] == 'put')

        breakeven_low = short_put['strike'] - net_premium_received
        breakeven_high = short_call['strike'] + net_premium_received
        max_profit = net_premium_received
        max_loss = float('inf')

        result.update({
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": [breakeven_low, breakeven_high]
        })

    elif strategy == "Bull Put Spread":
        short_put = next(leg for leg in legs if leg['position'] == -1 and leg['type'] == 'put')
        long_put = next(leg for leg in legs if leg['position'] == 1 and leg['type'] == 'put')

        max_profit = net_premium_received
        max_loss = long_put['strike'] - short_put['strike'] - max_profit
        breakeven = short_put['strike'] + max_profit

        result.update({
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": [breakeven]
        })

    elif strategy == "Bear Call Spread":
        short_call = next(leg for leg in legs if leg['position'] == -1 and leg['type'] == 'call')
        long_call = next(leg for leg in legs if leg['position'] == 1 and leg['type'] == 'call')

        max_profit = net_premium_received
        max_loss = long_call['strike'] - short_call['strike'] - max_profit
        breakeven = short_call['strike'] + max_profit

        result.update({
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": [breakeven]
        })

    else:
        # Strategy recognized but no calculation provided yet
        result.update({
            "max_profit": None,
            "max_loss": None,
            "breakeven": []
        })

    return result

