import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Setup ---
tickers = ['AAPL', 'MSFT', 'TSLA']
weights = np.array([1/3, 1/3, 1/3]) 
initial_value = 10000 
train_start = "2020-01-01"
train_end   = "2025-01-01"
T_future = 1.0
dt = 1/252
N = 2000 

print(f"Fetching data for {tickers}...")
data = yf.download(tickers, start=train_start, end=train_end)['Close']
log_returns = np.log(data / data.shift(1)).dropna()


avg_daily_rets = log_returns.mean().values
variances = np.diag(log_returns.cov())
drifts = avg_daily_rets - 0.5 * variances


cov_matrix = log_returns.cov()
L_corr = np.linalg.cholesky(cov_matrix.values) # The "Linker"

# Matrix B: Naive World (Zero Correlation)
# We create a matrix with ONLY variance on diagonal, zeros elsewhere
cov_matrix_naive = np.diag(variances) 
L_naive = np.sqrt(cov_matrix_naive) # Just the standard deviations

# --- 3. Double Simulation Engine ---
T_days = int(T_future * 252)
num_stocks = len(tickers)

# Outputs
port_corr_final_values = []
port_naive_final_values = []

print("Running Monte Carlo (Parallel Universes)...")

# We simulate N paths
# To make it fair, we use the SAME random base numbers (Z) for both!
# This isolates the effect of Correlation.
Z = np.random.normal(0, 1, (T_days, num_stocks, N))

# Engine
# correlated_Z = L_corr @ Z
# naive_Z      = L_naive @ Z

# Pre-calculate matrix multiplication for speed
# Reshape Z to (T_days, N, Stocks) for easier broadcasting if needed, 
# but dot product works best with (Stocks, N)
path_corr = np.zeros((T_days, num_stocks, N))
path_naive = np.zeros((T_days, num_stocks, N))

for t in range(T_days):
    # L (3x3) @ Z[t] (3x1000) -> (3x1000)
    
    # 1. Real World
    shocks_corr = np.dot(L_corr, Z[t]) * np.sqrt(dt)
    path_corr[t] = np.exp(drifts.reshape(-1,1) * dt + shocks_corr)
    
    # 2. Naive World
    shocks_naive = np.dot(L_naive, Z[t]) * np.sqrt(dt)
    path_naive[t] = np.exp(drifts.reshape(-1,1) * dt + shocks_naive)

# --- 4. Build Prices & Portfolio Values ---
# Start prices
S0 = data.iloc[-1].values.reshape(-1, 1) # (3,1)

# Accumulate returns (Cumprod)
# path_corr is (Days, Stocks, Sims). Transpose to (Days, Sims, Stocks) for cumprod maybe?
# Let's keep it simple: Cumulative Product along time axis
price_path_corr = S0 * np.cumprod(path_corr, axis=0) # Broadcasting S0 works
price_path_naive = S0 * np.cumprod(path_naive, axis=0)

# Calculate Portfolio Value for every Sim (Weighted Sum)
# shares = (Investment * Weight) / S0
shares = (initial_value * weights).reshape(-1, 1) / S0

# Final Prices (Last Day) -> Shape (Stocks, Sims)
final_prices_corr = price_path_corr[-1]
final_prices_naive = price_path_naive[-1]

# Portfolio Values = Sum(Price * Shares)
final_vals_corr = np.sum(final_prices_corr * shares, axis=0)
final_vals_naive = np.sum(final_prices_naive * shares, axis=0)

# --- 5. Analysis & Visualization ---
plt.figure(figsize=(12, 6))

# Plot Density (Smooth Histograms)
sns.kdeplot(final_vals_naive, fill=True, color='blue', label='Naive (Uncorrelated)', alpha=0.3)
sns.kdeplot(final_vals_corr, fill=True, color='red', label='Real (Correlated)', alpha=0.3)

plt.axvline(initial_value, color='k', linestyle='--')
plt.title("The Danger of Ignoring Correlation: Portfolio Outcome Distributions")
plt.xlabel("Portfolio Value after 1 Year ($)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print Stats
var_95_corr = np.percentile(final_vals_corr, 5)
var_95_naive = np.percentile(final_vals_naive, 5)

print("\n--- RESULTS ---")
print(f"Naive VaR (95%): ${var_95_naive:,.2f} (Underestimated Risk)")
print(f"Real VaR  (95%): ${var_95_corr:,.2f}  (True Risk)")
print(f"Difference:      ${var_95_naive - var_95_corr:,.2f}")
print("\nInterpretation: Because these stocks move TOGETHER, the real risk is HIGHER.")
print("The Naive model assumes that when Apple falls, Tesla might rise to save you.")
print("The Real model knows that when Tech crashes, they all crash.")