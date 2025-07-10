import math

def binomial_tree_price(S,K,T,r,sigma, N=100, option_type='call', american=False, q=0.0):
    """
    PRICE A EUROPEAN USING A BINOMIAL TREE.

    PARAMATERS :: 
    S = STOCK PRICE(CURRENT)
    K = STRIKE PRICE
    T = TIME TO MATURITY (IN YEARS)
    R = RISKE FREE RATE
    SIGMA = VOLATILITY
    Q = DIVIDEND YIELD (CONTINOUS RATE)
    N - NUMBER OF THE STEPS
    OPTIONTYPE = CALL OR PUT

    RETURNS:
    --OPTION PRICE
    """
    dt = T / N ### TIME STEP
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 /u
    p = (math.exp((r - q) * dt) - d)/ (u - d)

    ## 1. GENERATE ASSET PRICES AT MATURITY
    asset_prices = [S * (u ** j) * (d ** (N - j)) for j in range(N + 1)]

    ## 2. COMPUTE OPTION VALUES AT MATURITY
    if option_type == 'call':
        option_values = [max(0, price - K) for price in asset_prices]
    elif option_type == 'put':
        option_values = [max(0, K - price) for price in asset_prices]
    else:
        raise ValueError("option_type must be 'call' or 'put'")

  # Step 3: Backward induction
    for i in range(N - 1, -1, -1):
        option_values_new = []
        for j in range(i + 1):
            hold = math.exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j])
            if american:
                # Intrinsic value at this node
                if option_type == 'call':
                    exercise = max(0, S * (u ** j) * (d ** (i - j)) - K)
                else:
                    exercise = max(0, K - S * (u ** j) * (d ** (i - j)))
                option_values_new.append(max(hold, exercise))
            else:
                option_values_new.append(hold)
        option_values = option_values_new

    return option_values[0]