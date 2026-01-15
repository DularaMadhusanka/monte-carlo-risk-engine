import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

ticker = 'AAPL'
train_start = '2020-01-01'
train_end = '2025-01-01'
test_end = '2026-01-01'
print(f"Fetching data for {ticker}...")
data = yf.download(ticker, start=train_start, end=test_end)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

#formula: ln(Today/Yesterday)
data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))

train_data = data[:train_end].copy()
test_data = data[train_end:].copy()

if len(test_data) == 0:
    print("No test data available. Please check the date range.")
else:
    print(f"Train data from {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} days)")
    print(f"Test data from {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data)} days)")

    daily_std = train_data['Log Returns'].std()
    daily_mean = train_data['Log Returns'].mean()

    sigma = daily_std * np.sqrt(252)  # Annualized volatility
    mu = daily_mean * 252 + 0.5 * sigma**2 # Annualized return

    S0 = train_data['Close'].iloc[-1]
    T_days = len(test_data) 
    dt = 1/252  # Daily steps
    N = 1000  # Number of simulations

    random_shocks = np.random.normal(0, np.sqrt(dt), (T_days, N))
    drift = (mu - 0.5*sigma**2)*dt
    diffusion = sigma*random_shocks
    daily_returns = np.exp(drift + diffusion)

    price_paths = np.zeros((T_days + 1, N))
    price_paths[0] = S0
    for t in range(1, T_days + 1):
        price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]


plt.figure(figsize=(12,6))
# Create index with one extra point for initial price
plot_index = [train_data.index[-1]] + list(test_data.index)
plt.plot(plot_index, price_paths, color = 'blue', alpha = 0.1, linewidth=0.5)
plt.plot(test_data.index, test_data['Close'], color = 'red', linewidth=2.5, label='Actual 2025 Price')

plt.title(f'Backtest: 2020-2024 model vs 2025 Reality {ticker}')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha =0.3)
plt.show()

lower_bound_tunnel = np.percentile(price_paths, 5, axis=1)
upper_bound_tunnel = np.percentile(price_paths, 95, axis=1)

actual_price_array = test_data['Close'].values.flatten()
limit = min(len(actual_price_array), len(lower_bound_tunnel))

inside_count = 0
for i in range(limit):
    if lower_bound_tunnel[i] <= actual_price_array[i] <= upper_bound_tunnel[i]:
        inside_count += 1

success_rate = inside_count / limit * 100
print(f"\n3. Backtest Score:")
print(f"The actual price stayed within the 90% confidence interval for {success_rate:.1f}% of the days.")




