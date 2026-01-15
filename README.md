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
pip install numpy pandas matplotlib yfinance seaborn
```

## Usage

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
