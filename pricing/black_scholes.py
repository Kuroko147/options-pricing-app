import math
from scipy.stats import norm

def black_scholes_price(S,K,T,r, sigma, option_type='call', q=0.0):
    if T <=0 or sigma <= 0:
        raise ValueError("Time to to maturity and volatility must be positive")

    d1 = (math.log(S / K) +(r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price