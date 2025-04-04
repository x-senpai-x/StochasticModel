# Heston Model Options Calculator

This web application allows you to calculate European option prices using the Heston stochastic volatility model. The Heston model extends the Black-Scholes model by allowing for non-constant volatility, which helps to better capture market phenomena like volatility smiles/skews and fat tails in return distributions.

## Features

- Calculate European option prices (call/put) using the Heston model
- Compare with Black-Scholes prices
- Interactive visualizations:
  - Price vs. Strike charts
  - Price vs. Time to maturity charts
  - Implied volatility smile/skew

## Setup Instructions

1. Clone this repository:
   ```
   git clone <repository-url>
   cd heston-model-calculator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

5. Open your web browser and go to the URL displayed in the terminal (typically http://localhost:8501)

## Model Parameters

- **S₀**: Initial stock price
- **K**: Strike price
- **τ**: Time to maturity in years
- **r**: Risk-free interest rate
- **v₀**: Initial variance
- **κ**: Mean reversion rate for variance
- **θ**: Long-term variance
- **σ**: Volatility of variance
- **ρ**: Correlation between stock price and variance
- **λ**: Risk premium for volatility risk

## Implementation Details

The application implements the Heston model as described in Heston's 1993 paper. The option pricing is computed using Fourier transform methods with numerical integration.

## References

- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies*, 6(2), 327-343.