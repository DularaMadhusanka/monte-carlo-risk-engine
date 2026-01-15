import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

ticker = ['AAPL', 'MSFT', 'TSLA']
weights = np.array([1/3, 1/3, 1/3])
initial_portfolio_value = 10000

train_start = '2020-01-01'
train_end = '2025-01-01'
T_future = 1.0
dt = 1/252  # Daily steps
N = 1000  

print(f"Fetching data for {ticker}...")
data = yf.download(ticker, start=train_start, end=train_end)['Close']

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

#formula: ln(Today/Yesterday)
log_returns = np.log(data/data.shift(1)).dropna()

avg_daily_rets = log_returns.mean()
cov_matrix = log_returns.cov()

L = np.linalg.cholesky(cov_matrix)

print("\n2. Market Structure (Correlation) Captured!")
print(" Covariance Matrix Shape:", cov_matrix.shape)

variances = np.diag(cov_matrix)
drifts = (avg_daily_rets - 0.5 * variances).values

T_days = int(T_future * 252)
sim_returns = np.zeros((T_days, N, len(ticker)))

Z = np.random.normal(0, 1, (T_days, N, len(ticker)))

for t in range(T_days):
    # Z[t] shape is (N, 3), we need (3, N) for matrix multiplication
    correlated_z = np.dot(L, Z[t].T)  # (3,3) @ (3,N) = (3,N)
    correlated_dW = correlated_z.T * np.sqrt(dt)  # Transpose back to (N,3)
    daily_ret = np.exp(drifts.reshape(-1, 1) * dt + correlated_dW.T)
    sim_returns[t] = daily_ret.T

last_prices = data.iloc[-1].values
price_paths = np.zeros((T_days + 1, N, len(ticker)))
price_paths[0] = last_prices

for t in range(1, T_days + 1):
    price_paths[t] = price_paths[t - 1] * sim_returns[t - 1]

shares_owned = (initial_portfolio_value * weights) / last_prices
portfolio_value_paths = np.dot(price_paths, shares_owned)

plt.figure(figsize=(12,6))
plt.plot(portfolio_value_paths[:, :50], alpha=0.4, color = 'green', linewidth=0.5)
plt.title(f"Monte Carlo: Portfolio Value (AAPL, MSFT, TSLA) over {T_future} Year")
plt.xlabel('Days')
plt.ylabel('Portfolio Value ($)')
plt.grid(True, alpha =0.3)
plt.show()

final_values = portfolio_value_paths[-1]
var_95 = np.percentile(final_values, 5)
print(f"\n--- Portfolio Risk Analysis ---")
print(f"Initial Investment: ${initial_portfolio_value:,.2f}")
print(f"VaR 95% (Worst Case): ${var_95:,.2f}")
print(f"Max Potential Loss:${initial_portfolio_value - var_95:,.2f}")
      