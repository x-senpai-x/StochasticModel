import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.integrate import quad
import plotly.express as px
from scipy.stats import norm

# Set page configuration
st.set_page_config(
    page_title="Heston Model Options Calculator",
    page_icon="📈",
    layout="wide"
)

# Import the Heston model class from heston.py
from heston import HestonModel

# Title and description
st.title("Heston Model Options Calculator")
st.markdown("""
This app calculates European option prices using the Heston stochastic volatility model.
The Heston model extends the Black-Scholes model by allowing for non-constant volatility.
""")

# Create sidebar for inputs
st.sidebar.header("Model Parameters")

# Parameter inputs
S0 = st.sidebar.number_input("Stock Price (S₀)", value=100.0, step=0.1)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=0.1)
tau = st.sidebar.number_input("Time to Maturity (τ) in years", value=1.0, min_value=0.01, step=0.1)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)

# Advanced parameters with expander
with st.sidebar.expander("Advanced Heston Parameters"):
    v0 = st.number_input("Initial Variance (v₀)", value=0.04, min_value=0.001, step=0.001)
    kappa = st.number_input("Mean Reversion Rate (κ)", value=2.0, step=0.1)
    theta = st.number_input("Long-term Variance (θ)", value=0.04, min_value=0.001, step=0.001)
    sigma = st.number_input("Volatility of Variance (σ)", value=0.3, min_value=0.01, step=0.01)
    rho = st.number_input("Correlation (ρ)", value=-0.7, min_value=-1.0, max_value=1.0, step=0.1)
    lambd = st.number_input("Risk Premium (λ)", value=0.0, step=0.1)

option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

# Black-Scholes implementation for comparison
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    else:
        return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def norm_cdf(x):
    return norm.cdf(x)

# Calculate price button
if st.sidebar.button("Calculate Option Price", type="primary"):
    # Create Heston model instance
    heston_model = HestonModel(
        S0=S0, K=K, v0=v0, kappa=kappa, theta=theta, 
        sigma=sigma, rho=rho, lambd=lambd, tau=tau, r=r
    )
    
    # Calculate prices
    heston_price = heston_model.heston_price(option_type)
    bs_price = black_scholes_price(S0, K, tau, r, np.sqrt(v0), option_type)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label=f"Heston Model {option_type.capitalize()} Price",
            value=f"${heston_price:.4f}"
        )
    
    with col2:
        st.metric(
            label=f"Black-Scholes {option_type.capitalize()} Price",
            value=f"${bs_price:.4f}",
            delta=f"{(heston_price - bs_price):.4f}",
            delta_color="normal"
        )
    
    # Create visualizations
    st.header("Model Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Price vs. Strike", "Price vs. Time", "Implied Volatility"])
    
    with tab1:
        # Generate prices for different strikes
        strikes = np.linspace(K * 0.7, K * 1.3, 100)
        heston_prices = []
        bs_prices = []
        
        for strike in strikes:
            heston_model_k = HestonModel(
                S0=S0, K=strike, v0=v0, kappa=kappa, theta=theta, 
                sigma=sigma, rho=rho, lambd=lambd, tau=tau, r=r
            )
            heston_prices.append(heston_model_k.heston_price(option_type))
            bs_prices.append(black_scholes_price(S0, strike, tau, r, np.sqrt(v0), option_type))
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes, y=heston_prices, mode='lines', name='Heston Model'))
        fig.add_trace(go.Scatter(x=strikes, y=bs_prices, mode='lines', name='Black-Scholes'))
        fig.update_layout(
            title=f"Option Price vs Strike Price ({option_type.capitalize()})",
            xaxis_title="Strike Price",
            yaxis_title="Option Price",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        This chart shows how option prices vary with different strike prices. 
        The Heston model better captures the volatility smile/skew observed in real markets.
        """)
    
    with tab2:
        # Generate prices for different times to maturity
        times = np.linspace(0.1, 2, 100)
        heston_prices_t = []
        bs_prices_t = []
        
        for t in times:
            heston_model_t = HestonModel(
                S0=S0, K=K, v0=v0, kappa=kappa, theta=theta, 
                sigma=sigma, rho=rho, lambd=lambd, tau=t, r=r
            )
            heston_prices_t.append(heston_model_t.heston_price(option_type))
            bs_prices_t.append(black_scholes_price(S0, K, t, r, np.sqrt(v0), option_type))
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=heston_prices_t, mode='lines', name='Heston Model'))
        fig.add_trace(go.Scatter(x=times, y=bs_prices_t, mode='lines', name='Black-Scholes'))
        fig.update_layout(
            title=f"Option Price vs Time to Maturity ({option_type.capitalize()})",
            xaxis_title="Time to Maturity (years)",
            yaxis_title="Option Price",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        This chart shows how option prices vary with different times to maturity.
        The Heston model can better capture the term structure of volatility.
        """)
    
    with tab3:
        # Generate implied volatility for different strikes
        def calculate_implied_vol(price, S, K, T, r, option_type):
            # Simple iterative solver for implied volatility
            MAX_ITERATIONS = 100
            PRECISION = 1.0e-8
            
            sigma = 0.3  # Initial guess
            
            for i in range(MAX_ITERATIONS):
                price_diff = black_scholes_price(S, K, T, r, sigma, option_type) - price
                
                if abs(price_diff) < PRECISION:
                    return sigma
                
                vega = S * np.sqrt(T) * norm_cdf((np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))) / np.sqrt(2 * np.pi)
                sigma = sigma - price_diff / (vega + 1e-10)  # Add small constant to avoid division by zero
                
                # Bounds check
                if sigma <= 0.001:
                    sigma = 0.001
                
            return sigma  # Return best estimate if max iterations reached
        
        strikes_iv = np.linspace(K * 0.7, K * 1.3, 20)
        heston_iv = []
        
        for strike in strikes_iv:
            heston_model_iv = HestonModel(
                S0=S0, K=strike, v0=v0, kappa=kappa, theta=theta, 
                sigma=sigma, rho=rho, lambd=lambd, tau=tau, r=r
            )
            price = heston_model_iv.heston_price(option_type)
            implied_vol = calculate_implied_vol(price, S0, strike, tau, r, option_type)
            heston_iv.append(implied_vol)
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes_iv, y=heston_iv, mode='lines+markers', name='Implied Volatility'))
        fig.add_trace(go.Scatter(x=strikes_iv, y=[np.sqrt(v0)] * len(strikes_iv), mode='lines', 
                                name='Constant BS Volatility', line=dict(dash='dash')))
        fig.update_layout(
            title="Implied Volatility Smile/Skew",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        This chart shows the implied volatility smile/skew generated by the Heston model.
        In the Black-Scholes world, this would be a flat line, but real market data shows smiles or skews.
        """)

# About section
st.header("About the Heston Model")
st.markdown("""
The Heston model, developed by Steven Heston in 1993, is a stochastic volatility model used in mathematical finance to model the evolution of volatility over time. Unlike the Black-Scholes model which assumes constant volatility, the Heston model accounts for:

- **Mean-reverting volatility:** Volatility tends to revert to a long-term average
- **Volatility clustering:** Periods of high volatility tend to be followed by high volatility
- **Volatility of volatility:** The variance itself has stochastic behavior
- **Correlation between asset price and volatility:** Often negative for equity markets

These features allow the Heston model to better capture market phenomena like:

- Volatility smiles and skews observed in option markets
- Fat tails in return distributions
- Realistic term structures of implied volatilities

The model defines the dynamics of the stock price and its variance using the following stochastic differential equations:

dS(t) = μS(t)dt + √v(t)S(t)dW₁(t)  
dv(t) = κ(θ - v(t))dt + σ√v(t)dW₂(t)

Where:
- S(t) is the stock price
- v(t) is the variance
- μ is the drift
- κ is the rate of mean reversion
- θ is the long-term variance
- σ is the volatility of variance
- dW₁ and dW₂ are Wiener processes with correlation ρ
""")

# Add footer
st.markdown("---")
st.markdown("Heston Model Options Calculator © 2025")