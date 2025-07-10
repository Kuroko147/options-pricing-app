from scipy.optimize import brentq
from scipy.stats import norm
import math

def black_scholes_price(S, K, T, r, sigma, option_type='call', q=0.0):
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def implied_volatility(market_price, S, K, T, r, option_type='call', q=0.0):
    if market_price <= 0 or T <= 0:
        return None

    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        return brentq(objective, 1e-5, 5.0, maxiter=500, xtol=1e-6)
    except:
        return None

