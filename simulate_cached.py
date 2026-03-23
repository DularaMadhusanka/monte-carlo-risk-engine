import argparse
import json

import numpy as np


STUDENT_T_DOF = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulation from cached market parameters."
    )
    parser.add_argument(
        "--params",
        default="artifacts/market_params.npz",
        help="Path to precomputed parameters produced by precompute_params.py",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=1.0,
        help="Simulation horizon in years.",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=5000,
        help="Number of Monte Carlo paths.",
    )
    parser.add_argument(
        "--initial",
        type=float,
        default=10000.0,
        help="Initial portfolio value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    params = np.load(args.params, allow_pickle=True)
    tickers = params["tickers"].tolist()
    annual_mu = params["annual_mu"]
    annual_cov = params["annual_cov"]
    chol_L = params["chol_L"]
    s0 = params["s0"]
    as_of = str(params["as_of"]) if "as_of" in params.files else "unknown"

    metadata = {}
    if "metadata_json" in params.files:
        metadata = json.loads(str(params["metadata_json"]))

    n_assets = len(tickers)
    if n_assets == 0:
        raise ValueError("No assets found in cached parameters.")

    if args.sims <= 0:
        raise ValueError("--sims must be > 0")

    if args.years <= 0:
        raise ValueError("--years must be > 0")

    dt = 1.0 / 252.0
    steps = int(args.years * 252)
    drift_term = (annual_mu - 0.5 * np.diag(annual_cov)) * dt

    rng = np.random.default_rng(args.seed)
    t_scaling = np.sqrt((STUDENT_T_DOF - 2.0) / STUDENT_T_DOF)
    z = rng.standard_t(df=STUDENT_T_DOF, size=(steps, args.sims, n_assets)) * t_scaling
    correlated_shocks = (z @ chol_L.T) * np.sqrt(dt)

    daily_growth = np.exp(drift_term.reshape(1, 1, -1) + correlated_shocks)

    price_paths = np.zeros((steps + 1, args.sims, n_assets))
    price_paths[0] = s0

    for t in range(1, steps + 1):
        price_paths[t] = price_paths[t - 1] * daily_growth[t - 1]

    weights = np.repeat(1.0 / n_assets, n_assets)
    shares_owned = (args.initial * weights) / s0
    portfolio_values = np.sum(price_paths * shares_owned.reshape(1, 1, -1), axis=2)

    final_values = portfolio_values[-1]
    var_95 = np.percentile(final_values, 5)
    expected_final = np.mean(final_values)

    print("--- Cached Monte Carlo Simulation ---")
    print(f"tickers: {tickers}")
    print(f"params_as_of: {as_of}")
    if metadata:
        print(f"trained_range: {metadata.get('start')} -> {metadata.get('end')}")
    print(f"horizon_years: {args.years}")
    print(f"simulations: {args.sims}")
    print(f"initial_value: ${args.initial:,.2f}")
    print(f"expected_final_value: ${expected_final:,.2f}")
    print(f"VaR 95% threshold: ${var_95:,.2f}")
    print(f"Max potential loss @95%: ${args.initial - var_95:,.2f}")


if __name__ == "__main__":
    main()
