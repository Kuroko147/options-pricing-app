import streamlit as st
import time
import sys
import os
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from streamlit import dataframe, expander
import numpy as np
import matplotlib.pyplot as plt
import traceback
import plotly.graph_objects as go
from pricing.black_scholes import black_scholes_price
from pricing.binomial_tree import binomial_tree_price
from pricing.greeks import compute_greeks
from pricing.implied_volatility import implied_volatility
from utils.plot_utilis import plot_volatility_smile
from strategies.payoff_composer import combined_strategy_payoff
from pricing.monte_carlo import monte_carlo_option_price
from utils.sanity_checks import run_all_checks_scenario
from tests.test_pricing import fetch_option_chain_with_iv, TICKERS 
from strategies.strategy_detector import detect_strategy, analyze_strategy


# --- CIR simulation function ---
def simulate_cir_v_paths(v0, kappa, theta, sigma, T, N, M, seed=None):
    """
    Simulate variance paths using the CIR model.
    dv_t = kappa*(theta - v_t)dt + sigma*sqrt(v_t)*dW_t
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    v_paths = np.zeros((M, N + 1))
    v_paths[:, 0] = v0

    for t in range(1, N + 1):
        vt = v_paths[:, t - 1]
        vt = np.maximum(vt, 0)
        dW = np.random.normal(0, np.sqrt(dt), size=M)
        dv = kappa * (theta - vt) * dt + sigma * np.sqrt(vt) * dW
        v_paths[:, t] = vt + dv
        v_paths[:, t] = np.maximum(v_paths[:, t], 0)

    return v_paths


# --- Plot functions ---
def plot_payoff(ax, S, K, premium, option_type, title, breakeven=None):
    S_T_range = np.linspace(0.5 * S, 1.5 * S, 100)
    if option_type == 'call':
        payoff = np.maximum(S_T_range - K, 0) - premium
    else:
        payoff = np.maximum(K - S_T_range, 0) - premium
    ax.plot(S_T_range, payoff, label="Payoff")
    ax.axhline(0, color="black", linewidth=0.5)
    if breakeven:
        ax.axvline(breakeven, linestyle="--", color="red", label=f"Break-even: {breakeven:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Underlying Price at Expiration")
    ax.set_ylabel("Profit / Loss")
    ax.legend()
    ax.grid(True)


def plot_binomial_convergence(ax, S, K, T, r, sigma, option_type, q, american, max_steps):
    steps = list(range(10, max_steps + 1, 10))
    prices = [binomial_tree_price(S, K, T, r, sigma, N=n, option_type=option_type, q=q, american=american) for n in steps]
    bs_price = black_scholes_price(S, K, T, r, sigma, option_type=option_type, q=q)
    ax.plot(steps, prices, marker='o', label="Binomial Tree")
    ax.axhline(bs_price, color="red", linestyle="--", label="Black-Scholes")
    ax.set_title("Binomial Tree Convergence")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)


# --- Streamlit UI setup ---
st.set_page_config(layout="wide")
st.title("Option Pricing Calculator with Scenario Comparison and CIR Volatility Simulation")

option_style = st.radio(
    "Option Style",
    options=['European', 'American'],
    horizontal=True,
    key="option_style"
)
american = option_style == "American"

today = datetime.date.today()

with st.sidebar.expander("Scenario 1: Input Parameters", expanded=True):
    st.markdown("### Option & Market Parameters")
    S1 = st.number_input("Stock Price (S1)", min_value=0.01, value=100.0, key='S1')
    K1 = st.number_input("Strike Price (K1)", min_value=0.01, value=100.0, key='K1')
    expiry_date_1 = st.date_input("Expiry Date (Scenario 1)", value=today + datetime.timedelta(days=365), key='expiry1')
    T1 = max((expiry_date_1 - today).days / 365, 0.0001)
    r1 = st.number_input("Risk-free Rate (r1)", min_value=0.0, value=0.05, step=0.001, key='r1')
    sigma1 = st.number_input("Volatility (σ1)", min_value=0.01, value=0.2, step=0.01, key='sigma1')
    q1 = st.number_input("Dividend Yield (q1)", min_value=0.0, value=0.0, step=0.001, key='q1')
    option_type_1 = st.selectbox("Option Type (Scenario 1)", ['call', 'put'], key='opt1')
    market_price_1 = st.number_input("Market Option Price (Scenario 1, optional)", min_value=0.0, value=0.0, step=0.01, key="market_price_1")
    N1 = st.slider("Binomial Tree Steps (N1)", min_value=10, max_value=500, value=100, key="steps_pricing1")
    
    
with st.sidebar.expander("Scenario 2: Input Parameters", expanded=False):
    st.markdown("### Option & Market Parameters")
    S2 = st.number_input("Stock Price (S2)", min_value=0.01, value=100.0, key='S2')
    K2 = st.number_input("Strike Price (K2)", min_value=0.01, value=100.0, key='K2')
    expiry_date_2 = st.date_input("Expiry Date (Scenario 2)", value=today + datetime.timedelta(days=365), key='expiry2')
    T2 = max((expiry_date_2 - today).days / 365, 0.0001)
    r2 = st.number_input("Risk-free Rate (r2)", min_value=0.0, value=0.05, step=0.001, key='r2')
    sigma2 = st.number_input("Volatility (σ2)", min_value=0.01, value=0.2, step=0.01, key='sigma2')
    q2 = st.number_input("Dividend Yield (q2)", min_value=0.0, value=0.0, step=0.001, key='q2')
    option_type_2 = st.selectbox("Option Type (Scenario 2)", ['call', 'put'], key='opt2')
    market_price_2 = st.number_input("Market Option Price (Scenario 2, optional)", min_value=0.0, value=0.0, step=0.01, key="market_price_2")
    N2 = st.slider("Binomial Tree Steps (N2)", min_value=10, max_value=500, value=100, key="steps_pricing2")
    


st.sidebar.markdown("---")
use_cir_vol = st.sidebar.checkbox("Use CIR stochastic volatility (advanced)", value=False, key="use_cir_vol")

if use_cir_vol:
    st.sidebar.markdown("### CIR Model Parameters")
    v0_cir = st.sidebar.number_input("Initial Variance (v₀)", min_value=0.0001, value=0.04, step=0.01, key="cir_v0")
    kappa = st.sidebar.number_input("Mean Reversion Speed (κ)", min_value=0.01, value=3.0, step=0.1, key="cir_kappa")
    theta = st.sidebar.number_input("Long Term Variance (θ)", min_value=0.0001, value=0.04, step=0.01, key="cir_theta")
    sigma_v = st.sidebar.number_input("Volatility of Volatility (σ)", min_value=0.001, value=0.3, step=0.01, key="cir_sigma_v")
    cir_steps = st.sidebar.slider("CIR simulation time steps", min_value=50, max_value=500, value=250, step=10, key="cir_steps")
    cir_paths = st.sidebar.slider("Number of CIR simulation paths", min_value=100, max_value=2000, value=500, step=100, key="cir_paths")

if "run_comparison" not in st.session_state:
    st.session_state.run_comparison = False

if st.button("Compare Scenarios"):
    st.session_state.run_comparison = True


if st.session_state.run_comparison:


    # Effective volatilities with CIR if enabled
    if use_cir_vol:
        v_paths_1 = simulate_cir_v_paths(v0_cir, kappa, theta, sigma_v, T1, cir_steps, cir_paths)
        avg_variance_1 = np.mean(v_paths_1)
        sigma1_used = np.sqrt(avg_variance_1)
    else:
        sigma1_used = sigma1

    if use_cir_vol:
        v_paths_2 = simulate_cir_v_paths(v0_cir, kappa, theta, sigma_v, T2, cir_steps, cir_paths)
        avg_variance_2 = np.mean(v_paths_2)
        sigma2_used = np.sqrt(avg_variance_2)
    else:
        sigma2_used = sigma2

        # Scenario 1 calculations
    price_bs_1 = black_scholes_price(S1, K1, T1, r1, sigma1_used, option_type=option_type_1, q=q1)
    price_bt_1 = binomial_tree_price(S1, K1, T1, r1, sigma1_used, N1, option_type=option_type_1, american=american, q=q1)
    greeks_1 = compute_greeks(S1, K1, T1, r1, sigma1_used, option_type=option_type_1, q=q1)
    

    
    # Scenario 2 calculations
    price_bs_2 = black_scholes_price(S2, K2, T2, r2, sigma2_used, option_type=option_type_2, q=q2)
    price_bt_2 = binomial_tree_price(S2, K2, T2, r2, sigma2_used, N2, option_type=option_type_2, american=american, q=q2)
    greeks_2 = compute_greeks(S2, K2, T2, r2, sigma2_used, option_type=option_type_2, q=q2)
    


    # Implied volatility function
    def calc_iv(mkt_price, S, K, T, r, opt_type, q):
        if mkt_price > 0 and T > 0:
            try:
                return implied_volatility(mkt_price, S, K, T, r, opt_type, q)
            except:
                return None
        return None

    iv_1 = calc_iv(market_price_1, S1, K1, T1, r1, option_type_1, q1)
    iv_2 = calc_iv(market_price_2, S2, K2, T2, r2, option_type_2, q2)

    # Display Scenario Comparison
    st.subheader("Scenario Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Scenario 1")
        st.write(f"Option Type: {option_type_1.capitalize()}")
        st.write(f"Effective Volatility (σ1): {sigma1_used:.4f} {'(from CIR)' if use_cir_vol else '(input)'}")
        st.write(f"Black-Scholes Price: {price_bs_1:.4f}")
        st.write(f"Binomial Tree Price: {price_bt_1:.4f}")
        if market_price_1 > 0:
            st.write(f"Market Price: {market_price_1:.4f}")
            if iv_1 is not None:
                st.write(f"Implied Volatility (from market): {iv_1:.4f}")
            else:
                st.warning("Could not compute implied volatility for Scenario 1.")
        else:
            st.info("Enter a market price to compute implied volatility.")

        intrinsic_1 = max(S1 - K1, 0) if option_type_1 == 'call' else max(K1 - S1, 0)
        time_value_1 = (market_price_1 if market_price_1 > 0 else price_bs_1) - intrinsic_1

        run_all_checks_scenario(
            "Scenario 1",
            option_type_1,
            sigma1_used,
            T1,
            market_price_1,
            intrinsic_1,
            time_value_1
        )

        st.write(f"Intrinsic Value: {intrinsic_1:.4f}")
        st.write(f"Time Value: {time_value_1:.4f}")
        
        st.write("Greeks:")
        for g, val in greeks_1.items():
            st.write(f"{g}: {val:.4f}")
        breakeven_1 = K1 + (market_price_1 if market_price_1 > 0 else price_bs_1) if option_type_1 == 'call' else K1 - (market_price_1 if market_price_1 > 0 else price_bs_1)
        st.write(f"Break-even Point: {breakeven_1:.2f}")

        include_mc_1 = st.checkbox("Include Monte Carlo Estimate for Scenario 1", key="mc_scenario_1_check")
        if include_mc_1:
            if 'mc_price_1' not in st.session_state:
                st.session_state.mc_price_1 = monte_carlo_option_price(
                    S1, K1, T1, r1, sigma1_used,
                    option_type=option_type_1,
                    num_paths=10000, num_steps=100
                )
            st.write(f"📊 Monte Carlo Price (Scenario 1): **{st.session_state.mc_price_1:.4f}** (using 10,000 paths)")
        else:
            # Optionally clear stored price when unchecked
            st.session_state.pop('mc_price_1', None)
    
    
    with col2:
        st.markdown("### Scenario 2")
        st.write(f"Option Type: {option_type_2.capitalize()}")
        st.write(f"Effective Volatility (σ2): {sigma2_used:.4f} {'(from CIR)' if use_cir_vol else '(input)'}")
        st.write(f"Black-Scholes Price: {price_bs_2:.4f}")
        st.write(f"Binomial Tree Price: {price_bt_2:.4f}")
        if market_price_2 > 0:
            st.write(f"Market Price: {market_price_2:.4f}")
            if iv_2 is not None:
                st.write(f"Implied Volatility (from market): {iv_2:.4f}")
            else:
                st.warning("Could not compute implied volatility for Scenario 2.")
        else:
            st.info("Enter a market price to compute implied volatility.")

        intrinsic_2 = max(S2 - K2, 0) if option_type_2 =='call' else max(K2 - S2, 0)
        time_value_2 = (market_price_2 if market_price_2 > 0 else price_bs_2) - intrinsic_2

        run_all_checks_scenario(
            "Scenario 2",
            option_type_2,
            sigma2_used,
            T2,
            market_price_2,
            intrinsic_2,
            time_value_2
        )

        st.write(f"Intrinsic Value: {intrinsic_2:.4f}")
        st.write(f"Time Value: {time_value_2:.4f}")

        
        st.write("Greeks:")
        for g, val in greeks_2.items():
            st.write(f"{g}: {val:.4f}")
        breakeven_2 = K2 + (market_price_2 if market_price_2 > 0 else price_bs_2) if option_type_2 == 'call' else K2 - (market_price_2 if market_price_2 > 0 else price_bs_2)
        st.write(f"Break-even Point: {breakeven_2:.2f}")

        include_mc_2 = st.checkbox("Include Monte Carlo Estimate for Scenario 2", key="mc_scenario_2_check")
        if include_mc_2:
            if 'mc_price_2' not in st.session_state:
                st.session_state.mc_price_2 = monte_carlo_option_price(
                    S2, K2, T2, r2, sigma2_used,
                    option_type=option_type_2,
                    num_paths=10000, num_steps=100
                )
            st.write(f"📊 Monte Carlo Price (Scenario 2): **{st.session_state.mc_price_2:.4f}** (using 10,000 paths)")
        else:
            # Optionally clear stored price when unchecked
            st.session_state.pop('mc_price_2', None)
    

    if not american:
        st.markdown("---")
        st.subheader("🔁 Put-Call Parity Check (European Options Only)")
        for i, (S, K, T, r, option_type, market_price, scenario) in enumerate([
            (S1, K1, T1, r1, option_type_1, market_price_1, "Scenario 1"),
            (S2, K2, T2, r2, option_type_2, market_price_2, "Scenario 2")]):
            
            if market_price > 0:
                st.markdown(f"**{scenario} Parity Check**")
                discounted_K = K * np.exp(-r * T)
    
                if option_type == 'call':
                    # You have call price → compute put via parity
                    call_price = market_price
                    put_parity = call_price - S + discounted_K
                    parity_diff = call_price - (put_parity + S - discounted_K)
                else:
                    # You have put price → compute call via parity
                    put_price = market_price
                    call_parity = put_price + S - discounted_K
                    parity_diff = call_parity - (put_price + S - discounted_K)
    
                st.write(f"Discounted Strike (K·e^(-rT)): {discounted_K:.4f}")
                if option_type == 'call':
                    st.write(f"Implied Put (via parity): {put_parity:.4f}")
                else:
                    st.write(f"Implied Call (via parity): {call_parity:.4f}")
                st.write(f"Put-Call Parity Diff: {parity_diff:.4f}")
    
                if abs(parity_diff) < 0.01:
                    st.success("✅ Put-Call parity holds approximately.")
                else:
                    st.warning("⚠️ Put-Call parity **does not** hold.")


    # --- Binomial Tree Convergence ---
    st.markdown("---")
    st.subheader("Binomial Tree Convergence")
    col_conv1, col_conv2 = st.columns(2)
    for i, (col, S, K, T, r, sigma, option_type, q) in enumerate([
        (col_conv1, S1, K1, T1, r1, sigma1_used, option_type_1, q1),
        (col_conv2, S2, K2, T2, r2, sigma2_used, option_type_2, q2)]):
        with col:
            st.markdown(f"**Scenario {i+1}**")
            max_steps = st.slider(f"Max Steps (Scenario {i+1})", min_value=10, max_value=500, value=100, step=10)
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_binomial_convergence(ax, S, K, T, r, sigma, option_type, q, american, max_steps)
            st.pyplot(fig)
            plt.close(fig)


    # --- Volatility Smile Plot---


    st.markdown("---")

with st.expander("Show Volatility Smile Tool"):
    st.markdown("Enter a list of strike prices and market prices to visualize the implied volatility smile.")
    
    smile_col1, smile_col2 = st.columns(2)
    with smile_col1:
        strikes_input = st.text_area("Strike Prices (comma-separated)", "80,90,100,110,120")
    with smile_col2:
        prices_input = st.text_area("Market Option Prices (comma-separated)", "24,18,12,7,4")
    
    try:
        strikes = [float(x.strip()) for x in strikes_input.split(",")]
        market_prices = [float(x.strip()) for x in prices_input.split(",")]
    
        if len(strikes) != len(market_prices):
            st.error("Please ensure both lists have the same number of values")
        elif T1 <= 0:
            st.error("Time to expiry must be positive.")
        else:
            fig = plot_volatility_smile(S1, T1, r1, option_type_1, q1, strikes, market_prices)
            st.pyplot(fig)
            plt.close(fig)
    
    except Exception as e:
        st.warning(f"Invalid input or error computing smile: {e}")



# --- Strategy Builder ---
st.markdown("---")
with st.expander("Option Payoff Builder"):
# --- Option Strategy Input Section ---
    st.subheader("🧮 Option Strategy Payoff Builder")
    
    # Strategy dropdown
    selected_strategy = st.selectbox(
        "Select Strategy to auto-fill legs (or choose Custom Strategy)",
        ["Custom Strategy", "Bull Call Spread", "Bear Put Spread", "Long Straddle",
         "Short Straddle", "Long Strangle", "Short Strangle", "Bull Put Spread", "Bear Call Spread"]
    )
    
    legs = []
    
    # --- Custom Strategy Input ---
    if selected_strategy == "Custom Strategy":
        num_legs = st.number_input("Number of Option Legs", min_value=1, max_value=5, value=2, step=1)
    
        for i in range(num_legs):
            st.markdown(f"### Leg {i+1}")
            option_type = st.selectbox(f"Type {i+1}", ["call", "put"], key=f"type_{i}")
            strike = st.number_input(f"Strike (K{i+1})", value=100.0, step=1.0, key=f"strike_{i}")
            premium = st.number_input(f"Premium (Leg {i+1})", value=5.0, step=0.5, key=f"premium_{i}")
            position_str = st.selectbox(f"Position {i+1}", ["Long", "Short"], key=f"position_{i}")
            position = 1 if position_str == "Long" else -1
    
            legs.append({
                "type": option_type,
                "strike": strike,
                "premium": premium,
                "position": position
            })
    
    # --- Predefined Strategies ---
    else:
        if selected_strategy == "Bull Call Spread":
            legs = [
                {"type": "call", "strike": 95, "premium": 7, "position": 1},
                {"type": "call", "strike": 105, "premium": 3, "position": -1}
            ]
        elif selected_strategy == "Bear Put Spread":
            legs = [
                {"type": "put", "strike": 105, "premium": 8, "position": 1},
                {"type": "put", "strike": 95, "premium": 4, "position": -1}
            ]
        elif selected_strategy == "Long Straddle":
            legs = [
                {"type": "call", "strike": 100, "premium": 6, "position": 1},
                {"type": "put", "strike": 100, "premium": 5, "position": 1}
            ]
        elif selected_strategy == "Short Straddle":
            legs = [
                {"type": "call", "strike": 100, "premium": 6, "position": -1},
                {"type": "put", "strike": 100, "premium": 5, "position": -1}
            ]
        elif selected_strategy == "Long Strangle":
            legs = [
                {"type": "put", "strike": 95, "premium": 3, "position": 1},
                {"type": "call", "strike": 105, "premium": 4, "position": 1}
            ]
        elif selected_strategy == "Short Strangle":
            legs = [
                {"type": "put", "strike": 95, "premium": 3, "position": -1},
                {"type": "call", "strike": 105, "premium": 4, "position": -1}
            ]
        elif selected_strategy == "Bull Put Spread":
            legs = [
                {"type": "put", "strike": 95, "premium": 2, "position": -1},
                {"type": "put", "strike": 105, "premium": 5, "position": 1}
            ]
        elif selected_strategy == "Bear Call Spread":
            legs = [
                {"type": "call", "strike": 95, "premium": 2, "position": -1},
                {"type": "call", "strike": 105, "premium": 5, "position": 1}
            ]
    
        # Display strategy legs
        st.markdown("### Pre-filled Legs for Selected Strategy")
        for i, leg in enumerate(legs):
            position_label = "Long" if leg["position"] == 1 else "Short"
            st.markdown(
                f"**Leg {i+1}:** {position_label} {leg['type'].title()} | "
                f"Strike: {leg['strike']} | Premium: {leg['premium']}"
            )
            
        # --- Spot Price Input ---
    spot_price = st.number_input("Spot Price (S)", value=100.0, step=1.0)
    
    # Analyze the strategy
    analysis = analyze_strategy(legs)
    
    # Calculate payoff
    x = np.linspace(spot_price * 0.5, spot_price * 1.5, 100)
    payoff = np.zeros_like(x)
    
    for leg in legs:
        if leg["type"] == "call":
            leg_payoff = np.maximum(x - leg["strike"], 0)
        else:
            leg_payoff = np.maximum(leg["strike"] - x, 0)
        leg_payoff = leg["position"] * (leg_payoff - leg["premium"])
        payoff += leg_payoff
    
    # Plot payoff chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=payoff, mode="lines", name="Strategy Payoff", line=dict(color='blue')))
    fig.add_vline(x=spot_price, line_dash="dash", line_color="gray",
                  annotation_text="Spot", annotation_position="top left")
    fig.update_layout(
        title="Options Strategy Payoff",
        xaxis_title="Spot Price at Expiry",
        yaxis_title="Net P&L",
        template="plotly_white",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show payoff table
    table_df = pd.DataFrame({
        "Spot Price": np.round(x, 2),
        "Total Payoff": np.round(payoff, 2)
    })
    st.dataframe(table_df)
    
    if analysis:
        st.markdown(f"### 🔍 Detected Strategy: {analysis['strategy']}")
    
        net_premium = analysis['net_premium']
        if net_premium >= 0:
            st.markdown(f"**Net Premium Received:** {net_premium:.2f}")
        else:
            st.markdown(f"**Net Premium Paid:** {-net_premium:.2f}")
    
        max_profit = analysis['max_profit']
        max_loss = analysis['max_loss']
        breakeven = analysis['breakeven']
    
        max_profit_str = f"{max_profit:.2f}" if max_profit is not None and max_profit != float('inf') else "Unlimited"
        max_loss_str = f"{max_loss:.2f}" if max_loss is not None and max_loss != float('inf') else "Unlimited"
    
        if len(breakeven) == 1:
            breakeven_str = f"{breakeven[0]:.2f}"
        else:
            breakeven_str = " and ".join(f"{b:.2f}" for b in breakeven)
    
        st.markdown("### 📊 Strategy Summary")
        st.markdown(f"**Max Profit (Est.):** {max_profit_str}")
        st.markdown(f"**Max Loss (Est.):** {max_loss_str}")
        st.markdown(f"**Breakeven Point(s):** {breakeven_str}")
    else:
        st.info("Strategy not recognized or unsupported. Showing general payoff only.")

        

# 🔧 UI: Category and Ticker Selection

st.markdown("---")
with st.expander("Real-Time Option Data"):

    category = st.selectbox("Select Asset Category", list(TICKERS.keys()))
    tickers = TICKERS[category]
    ticker = st.selectbox("Select Ticker", tickers)
    
    r_live = st.number_input("Risk-Free Rate (for IV)", min_value=0.0, value=0.05)
    q_live = st.number_input("Dividend Yield", min_value=0.0, value=0.0)
    
    # 🔧 Expiry Dropdown Setup
    try:
        ticker_obj = fetch_option_chain_with_iv.__globals__['yf'].Ticker(ticker)
        expiries = ticker_obj.options
    except Exception:
        expiries = []
    
    if expiries:
        expiry_choices = ["All"] + list(expiries)
        selected_expiry = st.selectbox("Select Expiry", expiry_choices)
    else:
        selected_expiry = None
        st.warning(f"No expiry data found for {ticker}.")
    
    # 🔧 Display Helper Function
    def format_and_display_options(df, option_type="Option"):
        if df.empty:
            st.info(f"No {option_type} data available.")
            return
    
        df = df.copy()
    
        # Ensure OI fields exist
        if 'openInterest' not in df.columns:
            df['openInterest'] = 0
        if 'changeInOpenInterest' not in df.columns:
            df['changeInOpenInterest'] = 0
    
        df['impliedVolFormatted'] = df['impliedVol'].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "N/A")
        df['openInterest'] = df['openInterest'].fillna(0).astype(int)
        df['changeInOpenInterest'] = df['changeInOpenInterest'].fillna(0).astype(int)
    
        # Moneyness filtering
        moneyness_options = df['moneyness'].unique().tolist()
        selected_moneyness = st.multiselect(
            f"Filter {option_type} by Moneyness", options=moneyness_options, default=moneyness_options,
            key=f"{option_type}_moneyness"
        )
        filtered_df = df[df['moneyness'].isin(selected_moneyness)]
    
        # Display dataframe with renamed columns
        display_df = filtered_df[['strike', 'lastPrice', 'impliedVolFormatted', 'openInterest', 'changeInOpenInterest', 'moneyness', 'expiry']].copy()
        display_df.columns = ['Strike', 'Last Price', 'Implied Vol', 'Open Interest', 'Change in OI', 'Moneyness', 'Expiry']
        display_df = display_df.sort_values(by='Strike')
    
        st.markdown(f"**Total Open Interest ({option_type}):** {display_df['Open Interest'].sum():,}")
    
        # Style Change in OI with colors
        def color_change(val):
            if val > 0:
                return 'color: green'
            elif val < 0:
                return 'color: red'
            return 'color: black'
    
        styled_df = display_df.style.applymap(color_change, subset=['Change in OI']) \
                                   .format({'Last Price': '${:,.2f}', 'Strike': '${:,.2f}'})
    
        st.dataframe(styled_df, use_container_width=True)
    
        # CSV download button
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {option_type} Options as CSV",
            data=csv,
            file_name=f"{option_type.lower()}_options_{ticker}.csv",
            mime='text/csv',
            key=f"{option_type}_csv"
        )
    
        # Charts
        st.markdown("📊 **Open Interest vs Strike**")
        st.bar_chart(display_df.set_index("Strike")["Open Interest"])
    
        st.markdown("📈 **Implied Volatility vs Strike**")
        iv_chart_df = display_df[display_df["Implied Vol"] != "N/A"].copy()
        iv_chart_df["IV Numeric"] = iv_chart_df["Implied Vol"].str.replace('%', '').astype(float)
        st.line_chart(iv_chart_df.set_index("Strike")["IV Numeric"])

    
    # 🔧 Fetch Button Logic
    if selected_expiry and st.button("📥 Fetch Live Option Chain"):
        try:
            expiry_param = None if selected_expiry == "All" else selected_expiry
            data = fetch_option_chain_with_iv(ticker, expiry=expiry_param, r=r_live, q=q_live)
    
            st.success(f"Data fetched for **{ticker}** at spot price **${data['spot']:.2f}**")
    
            col1, col2 = st.columns(2)
    
            with col1:
                st.markdown("### 📞 Call Options")
                format_and_display_options(data['calls'], option_type="Call")
    
            with col2:
                st.markdown("### 📉 Put Options")
                format_and_display_options(data['puts'], option_type="Put")
    
        except Exception as e:
            fallback = ticker if 'ticker' in locals() else "selected symbol"
            st.error(f"❌ Failed to fetch data for {fallback}: {e}")
            st.text(traceback.format_exc())
    else:
        st.info("Choose ticker and expiry, then click 'Fetch Live Option Chain'.")