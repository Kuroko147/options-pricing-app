import yfinance as yf
import datetime
import pandas as pd
from pricing.implied_volatility import implied_volatility  # Your module

# TICKERS defined here once
TICKERS = {
    "Indices & ETFs": [
        "^SPX", "^NDX", "^RUT", "^VIX", "SPY", "QQQ", "DIA", "IWM", "XSP"
    ],
    "Commodities (ETFs)": [
        "GLD", "SLV", "USO", "DBA"
    ]
}

def _safe_iv(price, spot, strike, T, r, option_type, q=0.0):
    try:
        if price < 0.05 or T <= 0:
            return None
        return implied_volatility(price, spot, strike, T, r, option_type, q)
    except:
        return None

def classify_option(row, spot, option_type):
    strike = row['strike']
    if option_type == 'call':
        if strike < spot:
            return 'ITM'
        elif strike == spot:
            return 'ATM'
        else:
            return 'OTM'
    else:
        if strike > spot:
            return 'ITM'
        elif strike == spot:
            return 'ATM'
        else:
            return 'OTM'

def fetch_option_chain_with_iv(ticker: str, expiry: str = None, r: float = 0.05, q: float = 0.0):
    import pandas as pd
    import datetime
    ticker_obj = yf.Ticker(ticker)
    expiries = ticker_obj.options
    if not expiries:
        raise ValueError(f"No option chain data available for {ticker}")

    fetch_all = expiry == "ALL" or expiry is None
    spot = ticker_obj.history(period='1d')['Close'].iloc[-1]
    all_calls = []
    all_puts = []

    expiries_to_fetch = expiries if fetch_all else [expiry]

    for exp in expiries_to_fetch:
        T = (datetime.datetime.strptime(exp, "%Y-%m-%d") - datetime.datetime.today()).days / 365.0
        T = max(T, 1e-4)

        try:
            option_chain = ticker_obj.option_chain(exp)
        except Exception as e:
            print(f"Error fetching options for {ticker} on {exp}: {e}")
            continue

        calls = option_chain.calls.copy()
        puts = option_chain.puts.copy()

        # Handle openInterest & changeInOpenInterest safely for calls
        if 'openInterest' not in calls.columns:
            calls['openInterest'] = 0
        else:
            calls['openInterest'] = calls['openInterest'].fillna(0).astype(int)

        if 'changeInOpenInterest' not in calls.columns:
            calls['changeInOpenInterest'] = 0
        else:
            calls['changeInOpenInterest'] = calls['changeInOpenInterest'].fillna(0).astype(int)

        # Same for puts
        if 'openInterest' not in puts.columns:
            puts['openInterest'] = 0
        else:
            puts['openInterest'] = puts['openInterest'].fillna(0).astype(int)

        if 'changeInOpenInterest' not in puts.columns:
            puts['changeInOpenInterest'] = 0
        else:
            puts['changeInOpenInterest'] = puts['changeInOpenInterest'].fillna(0).astype(int)

        # Calculate IV using yahoo IV if present, else fallback
        calls['impliedVol'] = calls.apply(
            lambda row: row['impliedVolatility'] if ('impliedVolatility' in row and pd.notnull(row['impliedVolatility'])) else
                        _safe_iv(row['lastPrice'], spot, row['strike'], T, r, 'call', q), axis=1)

        puts['impliedVol'] = puts.apply(
            lambda row: row['impliedVolatility'] if ('impliedVolatility' in row and pd.notnull(row['impliedVolatility'])) else
                        _safe_iv(row['lastPrice'], spot, row['strike'], T, r, 'put', q), axis=1)

        calls['moneyness'] = calls.apply(lambda row: classify_option(row, spot, 'call'), axis=1)
        puts['moneyness'] = puts.apply(lambda row: classify_option(row, spot, 'put'), axis=1)

        calls['expiry'] = exp
        puts['expiry'] = exp

        all_calls.append(calls[['strike', 'lastPrice', 'impliedVol', 'openInterest', 'changeInOpenInterest', 'moneyness', 'expiry']])
        all_puts.append(puts[['strike', 'lastPrice', 'impliedVol', 'openInterest', 'changeInOpenInterest', 'moneyness', 'expiry']])

    calls_df = pd.concat(all_calls, ignore_index=True)
    puts_df = pd.concat(all_puts, ignore_index=True)

    return {
        "spot": spot,
        "expiry": expiry if expiry and expiry != "ALL" else "ALL",
        "expiries": expiries,
        "calls": calls_df,
        "puts": puts_df
    }
