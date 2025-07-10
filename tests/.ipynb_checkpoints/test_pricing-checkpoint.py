import yfinance as yf
import datetime
import pandas as pd
from pricing.implied_volatility import implied_volatility

OPTIONABLE_TICKERS = {
    "US": [
        "AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "NFLX", "AMD", "INTC",
        "BABA", "DIS", "PYPL", "CRM", "ADBE", "CSCO", "V", "JNJ", "WMT"
    ],
    "India (ADRs)": [
        "INFY", "WIT", "HDB", "TCS", "EPL",
    ],
    "China (ADRs)": [
        "BIDU", "JD", "PDD", "TCEHY", "NIO",
    ],
    "UK (ADRs)": [
        "BP", "HSBC", "RDS.A", "RDS.B", "UL",
    ],
    "Canada (ADRs)": [
        "RY", "TD", "BNS", "ENB", "CNQ",
    ],
    "Germany (ADRs)": [
        "BAYRY", "DDAIF", "DMLRY", "SAP", "BAMXF",
    ],
    "Japan (ADRs)": [
        "TM", "SNE", "MFG", "MTU", "HMC",
    ],
    "France (ADRs)": [
        "LVMUY", "AIRYY", "ORPYF", "BNPYY",
    ],
    "Australia (ADRs)": [
        "BHP", "RIO", "WPL", "CSL", "WOW",
    ],
    "Brazil (ADRs)": [
        "VALE", "PBR", "ITUB", "BBD", "GGB",
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
    else:  # put option
        if strike > spot:
            return 'ITM'
        elif strike == spot:
            return 'ATM'
        else:
            return 'OTM'

def fetch_option_chain_with_iv(ticker: str, expiry: str = None, r: float = 0.05, q: float = 0.0):
    ticker_obj = yf.Ticker(ticker)
    expiries = ticker_obj.options
    if not expiries:
        raise ValueError(f"No option chain data available for {ticker}")

    # If user wants all expiries, we will combine them
    fetch_all = expiry == "ALL" or expiry is None

    spot = ticker_obj.history(period='1d')['Close'].iloc[-1]

    all_calls = []
    all_puts = []

    # Function to classify option
    def classify_option(row, spot, option_type):
        strike = row['strike']
        if option_type == 'call':
            if strike < spot:
                return 'ITM'
            elif strike == spot:
                return 'ATM'
            else:
                return 'OTM'
        else:  # put option
            if strike > spot:
                return 'ITM'
            elif strike == spot:
                return 'ATM'
            else:
                return 'OTM'

    expiries_to_fetch = expiries if fetch_all else [expiry]

    for exp in expiries_to_fetch:
        T = (datetime.datetime.strptime(exp, "%Y-%m-%d") - datetime.datetime.today()).days / 365.0
        T = max(T, 1e-4)

        option_chain = ticker_obj.option_chain(exp)
        calls = option_chain.calls.copy()
        puts = option_chain.puts.copy()

        # Calculate IV
        calls["impliedVol"] = calls.apply(
            lambda row: _safe_iv(row['lastPrice'], spot, row['strike'], T, r, 'call', q), axis=1
        )
        puts["impliedVol"] = puts.apply(
            lambda row: _safe_iv(row['lastPrice'], spot, row['strike'], T, r, 'put', q), axis=1
        )

        calls['moneyness'] = calls.apply(lambda row: classify_option(row, spot, 'call'), axis=1)
        puts['moneyness'] = puts.apply(lambda row: classify_option(row, spot, 'put'), axis=1)

        calls['expiry'] = exp
        puts['expiry'] = exp

        all_calls.append(calls[['strike', 'lastPrice', 'impliedVol', 'moneyness', 'expiry']])
        all_puts.append(puts[['strike', 'lastPrice', 'impliedVol', 'moneyness', 'expiry']])

    # Concatenate all expiry option chains
    calls_df = pd.concat(all_calls, ignore_index=True)
    puts_df = pd.concat(all_puts, ignore_index=True)

    return {
        "spot": spot,
        "expiry": expiry if expiry and expiry != "ALL" else "ALL",
        "expiries": expiries,
        "calls": calls_df,
        "puts": puts_df
    }



