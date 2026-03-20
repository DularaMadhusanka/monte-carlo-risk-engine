# Monte Carlo Risk Engine

A Python-based Monte Carlo simulation framework for modeling financial price movements and portfolio risk analysis. This project includes three different simulation models for analyzing single-stock behavior, multi-asset portfolios, and risk comparisons.

## Features

- **Single Stock Simulation (monte.py)**: Backtests a trained model on historical data and compares predictions against actual price movements
- **Multi-Asset Portfolio (monte2.py)**: Simulates correlated movements across multiple assets using Cholesky decomposition
- **Risk Analysis (monte3.py)**: Compares naive and realistic VaR calculations accounting for asset correlations

## Project Structure

```
monte-carlo-risk-engine/
├── monte.py          # Single stock backtest (AAPL 2020-2025)
├── monte2.py         # 3-asset portfolio simulation
├── monte3.py         # Naive vs correlated VaR analysis
└── README.md         # This file
```

## Installation

### Requirements
- Python 3.10+
- numpy
- pandas
- matplotlib
- yfinance
- seaborn

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DularaMadhusanka/monte-carlo-risk-engine.git
cd monte-carlo-risk-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Deployment Pattern (Fast Runtime)
For production deployment, avoid downloading years of market data on each request.
Use a two-step workflow:

1. **Daily precompute job** (Airflow / GitHub Actions / cron)
2. **Runtime simulation** that only loads cached parameters

#### Step 1: Precompute market parameters once per day
```bash
python precompute_params.py --tickers AAPL MSFT TSLA --start 2020-01-01 --output artifacts/market_params.npz
```

This produces a compressed artifact containing:
- Drift vector (`annual_mu`)
- Covariance matrix (`annual_cov`)
- Cholesky matrix (`chol_L`)
- Latest prices (`s0`)
- Metadata (`as_of`, date range, tickers)

#### Step 2: Use cached parameters for user simulations
```bash
python simulate_cached.py --params artifacts/market_params.npz --years 1 --sims 5000 --initial 10000
```

This path is fast because it does **not** call Yahoo Finance and does **not** recompute covariance/cholesky on each request.

#### Step 2 (Server): Inference API for deployment
Use this when deploying to AWS/Render/Streamlit backend infrastructure.

> **Important (Render):** this project uses pinned package versions that are compatible with Python 3.10.  
> Set `PYTHON_VERSION=3.10.14` in Render Environment Variables (or use the included `.python-version`) to avoid pandas source-build failures.

1) Install API dependencies:
```bash
pip install fastapi uvicorn
```

2) Start the service (loads cached params once at startup):
```bash
PARAMS_PATH=artifacts/market_params.npz uvicorn api:app --host 0.0.0.0 --port 8000
```

3) Call simulation endpoint:
```bash
curl -X POST http://localhost:8000/simulate \
	-H "Content-Type: application/json" \
	-d '{"initial_value": 10000, "years": 1, "sims": 5000, "seed": 42}'
```

The API returns portfolio metrics (`expected_final_value`, `var_95_threshold`, `max_potential_loss_95`) in milliseconds since all heavy market preprocessing is already cached.

#### Step 3: Streamlit frontend (dashboard)
1) Install frontend dependency:
```bash
pip install streamlit requests
```

2) Launch the UI:
```bash
streamlit run frontend.py
```

The dashboard calls the FastAPI backend (`/health` and `/simulate`) and lets users change `initial_value`, `years`, and `sims` interactively.

### 1. Single Stock Backtest
```bash
python monte.py
```
Trains a model on AAPL data (2020-2024) and backtests against 2025 actual prices.

**Output:**
- Annualized Volatility & Return metrics
- 1000 simulated price paths
- Visual comparison with actual 2025 prices

### 2. Multi-Asset Portfolio Simulation
```bash
python monte2.py
```
Simulates a 3-stock portfolio (AAPL, MSFT, TSLA) with equal weights, accounting for market correlations.

**Output:**
- Portfolio risk analysis
- Value at Risk (VaR) at 95% confidence
- Maximum potential loss estimate

### 3. Naive vs Realistic Risk Analysis
```bash
python monte3.py
```
Compares two risk models: one assuming independent stocks and one accounting for correlations.

**Output:**
- Naive VaR (underestimated)
- Real VaR (accounts for correlation)
- Risk difference analysis

## Technical Details

### Monte Carlo Method
All simulations use the Geometric Brownian Motion (GBM) model:

```
dS/S = μ dt + σ dW
```

Where:
- **S**: Stock price
- **μ**: Drift (annualized return)
- **σ**: Volatility (annualized)
- **dW**: Wiener process increment

### Correlation Handling
Multi-asset simulations use Cholesky decomposition to generate correlated random shocks, preserving the covariance structure between assets.

## Data Source

Historical price data is fetched from Yahoo Finance using the `yfinance` library.

## Results Interpretation

- **monte.py**: Shows if 2025 actual prices fell within predicted confidence intervals
- **monte2.py**: Portfolio VaR at 95% means 5% chance of loss exceeding the calculated amount
- **monte3.py**: Demonstrates why accounting for correlations is crucial for risk management

## License

MIT License - feel free to use for educational and commercial purposes.

## Author

Dulara Madhusanka

---

**Note**: Past performance does not guarantee future results. Use these simulations for educational purposes only.
