import numpy as np

def option_payoff(option_type, K, premium, position, S_range):
    """
    Compute the payoff of a single option.
    
    Parameters:
    - option_type: 'call' or 'put'
    - K: Strike price
    - premium: Option premium (cost)
    - position: +1 for long, -1 for short
    - S_range: Array of stock prices
    
    Returns:
    - Payoff array for the given S_range
    """
    if option_type == 'call':
        return position * (np.maximum(S_range - K, 0) - premium)
    elif option_type == 'put':
        return position * (np.maximum(K - S_range, 0) - premium)
    else:
        raise ValueError("Option type must be 'call' or 'put'")

def combined_strategy_payoff(legs, S_range):
    """
    Compute the total payoff of a multi-leg option strategy.
    
    Parameters:
    - legs: List of dictionaries, each with keys ['type', 'K', 'premium', 'position']
    - S_range: Array of stock prices
    
    Returns:
    - Total payoff array for the combined strategy
    """
    total_payoff = np.zeros_like(S_range, dtype=float)
    for leg in legs:
        payoff = option_payoff(
            leg['type'], leg['K'], leg['premium'], leg['position'], S_range
        )
        total_payoff += payoff
    return total_payoff  # ❗ FIXED: previously returning `payoff` instead of `total_payoff`
