import streamlit as st

def check_volatility(name, sigma):
    if sigma < 0.01 or sigma > 1.5:
        st.warning(f"Volatility for {name} is out of typical bounds (0.01 to 1.5). Please verify.")

def check_time_to_expiry(name, T):
    if T <= 0:
        st.error(f"Time to expiry for {name} must be positive.")

def check_market_price_vs_intrinsic(name, option_type, market_price, intrinsic):
    if market_price < intrinsic:
        st.warning(
            f"Market price for {name} ({option_type.capitalize()}) is less than intrinsic value. "
            "This is unusual and may indicate a data error or option type mismatch."
        )

def check_time_value(name, option_type, time_value):
    if time_value < 0:
        st.warning(
            f"Time value for {name} ({option_type.capitalize()}) is negative ({time_value:.4f}). "
            "This might be caused by inconsistent input parameters or option type."
        )

def run_all_checks_scenario(name, option_type, sigma, T, market_price, intrinsic, time_value):
    check_volatility(name, sigma)
    check_time_to_expiry(name, T)
    check_market_price_vs_intrinsic(name, option_type, market_price, intrinsic)
    check_time_value(name, option_type, time_value)
