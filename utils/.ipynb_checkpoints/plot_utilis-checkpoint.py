import matplotlib.pyplot as plt
import numpy as np
from pricing.implied_volatility import implied_volatility

def plot_volatility_smile(S, T, r, option_type, q, strikes, market_prices):
    ivs = []
    for K, market_price in zip(strikes, market_prices):
        try:
            iv = implied_volatility(market_price, S, K, T, r, option_type, q)
            ivs.append(iv)
        except Exception:
            ivs.append(np.nan)

    fig, ax = plt.subplots()
    ax.plot(strikes, ivs, marker ='o')
    ax.set_title("Implied Volatility Smile")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Implied Volatility")
    ax.grid(True)
    return fig