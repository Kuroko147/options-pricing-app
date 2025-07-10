import numpy as np

def monte_carlo_option_price(S, K, T, r, sigma,  option_type='call', num_paths=10000, num_steps=100):
    """
    MONTE CARLO SIMULATION FOR EUROPEAN OPTION PRICING.

    PARAMETERS:
    - num_paths: Number of Simulated paths
    - num_steps : Number of time steps per path

    Returns:
    - Estimated Option Price (float)

    """
    dt = T / num_steps
    discount_factor = np.exp(-r * T)

    S_paths = np.zeros((num_paths, num_steps + 1))
    S_paths[:, 0] = S

    for t in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_paths)  # 1D array of random shocks
        S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    if option_type == 'call':
        payoff = np.maximum(S_paths[:, -1] - K, 0)
    else:
        payoff = np.maximum(K - S_paths[:, -1], 0)

    price = discount_factor * np.mean(payoff)
    return price