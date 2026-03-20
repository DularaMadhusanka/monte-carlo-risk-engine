import argparse
import json
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute market parameters (mu, Sigma, L, S0) for fast deployment simulation."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "TSLA"],
        help="List of tickers to include in the parameter set.",
    )
    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date for historical download (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date for historical download (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output",
        default="artifacts/market_params.npz",
        help="Output path for precomputed parameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = args.tickers

    print(f"Fetching close prices for: {tickers}")
    data = yf.download(tickers, start=args.start, end=args.end, progress=False)["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])

    data = data.dropna(how="any")
    if data.empty:
        raise ValueError("No price data returned after cleaning. Check tickers/date range.")

    log_returns = np.log(data / data.shift(1)).dropna(how="any")
    if log_returns.empty:
        raise ValueError("No log returns available. Need at least 2 valid price rows.")

    daily_mean = log_returns.mean().values
    daily_cov = log_returns.cov().values

    annual_cov = daily_cov * 252.0
    annual_mu = daily_mean * 252.0 + 0.5 * np.diag(annual_cov)
    chol_L = np.linalg.cholesky(annual_cov)
    s0 = data.iloc[-1].values

    metadata = {
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "rows_used": int(len(data)),
        "return_rows": int(len(log_returns)),
        "trading_days": 252,
        "notes": "annual_mu and annual_cov are annualized; use dt in years for simulation.",
    }

    np.savez_compressed(
        args.output,
        tickers=np.array(tickers),
        annual_mu=annual_mu,
        annual_cov=annual_cov,
        chol_L=chol_L,
        s0=s0,
        as_of=np.array(data.index[-1].strftime("%Y-%m-%d")),
        metadata_json=np.array(json.dumps(metadata)),
    )

    print(f"Saved parameters to: {args.output}")
    print(f"as_of: {data.index[-1].date()}")
    print(f"num_assets: {len(tickers)}")
    print("Saved keys: tickers, annual_mu, annual_cov, chol_L, s0, as_of, metadata_json")


if __name__ == "__main__":
    main()
