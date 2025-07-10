import math
from scipy.stats import norm

def compute_greeks(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    Compute Greeks for a European option using the Black-Scholes model with dividend yield q.
    """

    if T <= 0 or sigma <= 0:
        raise ValueError("Time to maturity and volatility must be positive.")

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    greeks = {}

    # Delta
    if option_type == 'call':
        greeks['Delta'] = math.exp(-q * T) * norm.cdf(d1)
    elif option_type == 'put':
        greeks['Delta'] = -math.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Gamma
    greeks['Gamma'] = math.exp(-q * T) * pdf_d1 / (S * sigma * math.sqrt(T))

    # Vega
    greeks['Vega'] = S * math.exp(-q * T) * pdf_d1 * math.sqrt(T) / 100  # per 1% vol change

    # Theta
    if option_type == 'call':
        theta = (-S * math.exp(-q * T) * pdf_d1 * sigma / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2)
                 + q * S * math.exp(-q * T) * norm.cdf(d1))
    else:
        theta = (-S * math.exp(-q * T) * pdf_d1 * sigma / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)
                 - q * S * math.exp(-q * T) * norm.cdf(-d1))
    greeks['Theta'] = theta / 365

    # Rho
    if option_type == 'call':
        greeks['Rho'] = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:
        greeks['Rho'] = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    return greeks
