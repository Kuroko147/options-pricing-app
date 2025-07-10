import sys
import os
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from pricing.black_scholes import black_scholes_price
from pricing.binomial_tree import binomial_tree_price
from pricing.greeks import compute_greeks
from pricing.implied_volatility import implied_volatility


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

def plot_greek_sensitivity_vs_S(axs, S, K, T, r, sigma, option_type, q):
    S_vals = np.linspace(0.5 * S, 1.5 * S, 50)
    keys = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    greek_vals = {k: [] for k in keys}
    for s_val in S_vals:
        g = compute_greeks(s_val, K, T, r, sigma, option_type, q)
        for k in keys:
            greek_vals[k].append(g[k])
    for i, k in enumerate(keys):
        axs[i].plot(S_vals, greek_vals[k])
        axs[i].set_title(f"{k} vs Stock Price")
        axs[i].grid(True)

def plot_greek_sensitivity_vs_sigma(axs, S, K, T, r, sigma, option_type, q):
    sig_vals = np.linspace(0.01, 3 * sigma, 50)
    keys = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    greek_vals = {k: [] for k in keys}
    for sig in sig_vals:
        g = compute_greeks(S, K, T, r, sig, option_type, q)
        for k in keys:
            greek_vals[k].append(g[k])
    for i, k in enumerate(keys):
        axs[i].plot(sig_vals, greek_vals[k])
        axs[i].set_title(f"{k} vs Volatility")
        axs[i].grid(True)

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



if st.button("Compare Scenarios"):

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
        st.write("Greeks:")
        for g, val in greeks_1.items():
            st.write(f"{g}: {val:.4f}")
        breakeven_1 = K1 + (market_price_1 if market_price_1 > 0 else price_bs_1) if option_type_1 == 'call' else K1 - (market_price_1 if market_price_1 > 0 else price_bs_1)
        st.write(f"Break-even Point: {breakeven_1:.2f}")

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
        st.write("Greeks:")
        for g, val in greeks_2.items():
            st.write(f"{g}: {val:.4f}")
        breakeven_2 = K2 + (market_price_2 if market_price_2 > 0 else price_bs_2) if option_type_2 == 'call' else K2 - (market_price_2 if market_price_2 > 0 else price_bs_2)
        st.write(f"Break-even Point: {breakeven_2:.2f}")

    # --- Greek Sensitivity vs Stock Price ---
    st.markdown("---")
    st.subheader("Greek Sensitivity vs Stock Price")
    col_gs1, col_gs2 = st.columns(2)
    for i, (col, S, K, T, r, sigma, option_type, q) in enumerate(
        [(col_gs1, S1, K1, T1, r1, sigma1_used, option_type_1, q1),
         (col_gs2, S2, K2, T2, r2, sigma2_used, option_type_2, q2)]):
        with col:
            st.markdown(f"**Scenario {i+1}**")
            fig, axs = plt.subplots(5, 1, figsize=(6, 12))
            plot_greek_sensitivity_vs_S(axs, S, K, T, r, sigma, option_type, q)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # --- Greek Sensitivity vs Volatility ---
    st.markdown("---")
    st.subheader("Greek Sensitivity vs Volatility")
    col_vs1, col_vs2 = st.columns(2)
    for i, (col, S, K, T, r, sigma, option_type, q) in enumerate(
        [(col_vs1, S1, K1, T1, r1, sigma1_used, option_type_1, q1),
         (col_vs2, S2, K2, T2, r2, sigma2_used, option_type_2, q2)]):
        with col:
            st.markdown(f"**Scenario {i+1}**")
            fig, axs = plt.subplots(5, 1, figsize=(6, 12))
            plot_greek_sensitivity_vs_sigma(axs, S, K, T, r, sigma, option_type, q)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # --- Put-Call Parity Check (European only) ---
    if not american:
        st.markdown("---")
        st.subheader("🔁 Put-Call Parity Check (European Options Only)")
        for i, (S, K, T, r, option_type, market_price, scenario) in enumerate([
            (S1, K1, T1, r1, option_type_1, market_price_1, "Scenario 1"),
            (S2, K2, T2, r2, option_type_2, market_price_2, "Scenario 2")]):
            if market_price > 0:
                st.markdown(f"**{scenario} Parity Check**")
                discounted_K = K * np.exp(-r * T)
                parity_call = market_price if option_type == 'call' else S - market_price + discounted_K
                parity_put = market_price if option_type == 'put' else market_price - S + discounted_K
                parity_diff = parity_call - parity_put
                st.write(f"Call (via parity): {parity_call:.4f}")
                st.write(f"Put  (via parity): {parity_put:.4f}")
                st.write(f"Call - Put Diff: {parity_diff:.4f}")
                if abs(parity_diff) < 0.01:
                    st.success("Put-Call parity holds approximately.")
                else:
                    st.warning("Put-Call parity **does not** hold.")

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
    st.subheader("📈 Volatility Smile Analysis")


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
    st.subheader("🧮 Option Strategy Payoff Builder")


    with st.expander("Build and Visualize Custom Strategy"):
        st.markdown("Add Multiple legs (long, short calls/puts) to simulate your strategy.")

        num_legs = st.number_input("Number of Option Legs", min_value=1, max_value=5, value=2, step=1)

        legs = []
        for i in range(num_legs):       
            st.markdown(f"### Leg {i + 1}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                opt_type = st.selectbox(f"Type {i+1}", ["call", "put"], key=f"leg_type_{i}")
            with col2:
                strike = st.number_input(f"Strike (K{i+1})", min_value=0.01, value=100.0, key=f"leg_K_{i}")
            with col3:
                premium = st.number_input(f"Premium (Leg {i+1})", min_value=0.0, value=5.0, step=0.1, key=f"leg_prem_{i}")
            with col4:
                position = st.selectbox(f"Position {i+1}", ["Long", "Short"], key=f"leg_pos_{i}")
            legs.append({
                "type": opt_type,
                "K": strike,
                "premium": premium,
                "position": 1 if position == "Long" else -1
            })
    
        S = st.number_input("Spot Price (S)", min_value=0.01, value=100.0, key="strategy_spot")
        S_range = np.linspace(0.5 * S, 1.5 * S, 200)
        payoff = combined_strategy_payoff(legs, S_range)
    
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S_range, payoff, label="Strategy Payoff", color='blue')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(S, linestyle="--", color='gray', label=f"Spot: {S:.2f}")
        ax.set_title("Total Strategy Payoff at Expiration")
        ax.set_xlabel("Underlying Price at Expiry")
        ax.set_ylabel("Net P&L")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    
        net_premium = sum([leg["premium"] * -leg["position"] for leg in legs])
        max_profit = np.max(payoff)
        max_loss = np.min(payoff)
    
        breakeven_points = []
        for i in range(1, len(S_range)):
            if np.sign(payoff[i - 1]) != np.sign(payoff[i]):
                x1, x2 = S_range[i - 1], S_range[i]
                y1, y2 = payoff[i - 1], payoff[i]
                if y2 - y1 != 0:
                    x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
                    breakeven_points.append(x_zero)
    
        st.markdown("### 💡 Cost & Breakeven")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Net Premium Paid:** {net_premium:.2f}")
        with col_b:
            if breakeven_points:
                breakeven_str = ", ".join([f"{x:.2f}" for x in breakeven_points])
                st.write(f"**Breakeven Point(s):** {breakeven_str}")
            else:
                st.write("**Breakeven Point(s):** Not detected")
    
        st.markdown("### 📊 Strategy Summary")
        st.write(f"**Max Profit (Est.):** {max_profit:.2f}")
        st.write(f"**Max Loss (Est.):** {max_loss:.2f}")


       

            

