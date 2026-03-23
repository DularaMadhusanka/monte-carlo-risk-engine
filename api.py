import os
from functools import lru_cache
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


TRADING_DAYS = 252
DEFAULT_PARAMS_PATH = os.getenv("PARAMS_PATH", "artifacts/market_params.npz")


class SimulationRequest(BaseModel):
    initial_value: float = Field(gt=0, description="Initial portfolio value")
    years: float = Field(default=1.0, gt=0, le=5, description="Simulation horizon in years")
    sims: int = Field(default=1000, gt=0, le=5000, description="Number of Monte Carlo paths")
    seed: Optional[int] = Field(default=None, description="Optional seed for reproducibility")
    weights: Optional[List[float]] = Field(
        default=None,
        description="Optional portfolio weights. Must match number of assets and sum to 1.",
    )


class SimulationResponse(BaseModel):
    tickers: List[str]
    params_as_of: str
    initial_value: float
    years: float
    sims: int
    expected_final_value: float
    median_final_value: float
    var_95_threshold: float
    cvar_95_expected_shortfall: float
    max_potential_loss_95: float


@lru_cache(maxsize=1)
def load_params(path: str = DEFAULT_PARAMS_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Parameter file not found at '{path}'. Run precompute_params.py first."
        )

    params = np.load(path, allow_pickle=True)
    return {
        "tickers": params["tickers"].tolist(),
        "annual_mu": params["annual_mu"],
        "annual_cov": params["annual_cov"],
        "chol_L": params["chol_L"],
        "s0": params["s0"],
        "as_of": str(params["as_of"]) if "as_of" in params.files else "unknown",
    }


def run_simulation(payload: SimulationRequest, cached: dict) -> SimulationResponse:
    tickers = cached["tickers"]
    annual_mu = cached["annual_mu"]
    annual_cov = cached["annual_cov"]
    chol_L = cached["chol_L"]
    s0 = cached["s0"]
    n_assets = len(tickers)

    if payload.weights is None:
        weights = np.repeat(1.0 / n_assets, n_assets)
    else:
        weights = np.array(payload.weights, dtype=float)
        if len(weights) != n_assets:
            raise HTTPException(
                status_code=400,
                detail=f"weights length must be {n_assets}, got {len(weights)}",
            )
        if np.any(weights < 0):
            raise HTTPException(status_code=400, detail="weights must be non-negative")
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise HTTPException(status_code=400, detail="weights must sum to 1.0")

    dt = 1.0 / TRADING_DAYS
    steps = int(payload.years * TRADING_DAYS)
    drift_term = (annual_mu - 0.5 * np.diag(annual_cov)) * dt

    rng = np.random.default_rng(payload.seed)
    z = rng.normal(0.0, 1.0, size=(steps, payload.sims, n_assets))
    correlated_shocks = (z @ chol_L.T) * np.sqrt(dt)
    daily_growth = np.exp(drift_term.reshape(1, 1, -1) + correlated_shocks)

    price_paths = np.zeros((steps + 1, payload.sims, n_assets))
    price_paths[0] = s0
    for t in range(1, steps + 1):
        price_paths[t] = price_paths[t - 1] * daily_growth[t - 1]

    shares_owned = (payload.initial_value * weights) / s0
    portfolio_values = np.sum(price_paths * shares_owned.reshape(1, 1, -1), axis=2)
    final_values = portfolio_values[-1]

    var_95 = float(np.percentile(final_values, 5))
    tail_losses = final_values[final_values <= var_95]
    cvar_95 = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_95

    return SimulationResponse(
        tickers=tickers,
        params_as_of=cached["as_of"],
        initial_value=float(payload.initial_value),
        years=float(payload.years),
        sims=int(payload.sims),
        expected_final_value=float(np.mean(final_values)),
        median_final_value=float(np.median(final_values)),
        var_95_threshold=var_95,
        cvar_95_expected_shortfall=cvar_95,
        max_potential_loss_95=float(payload.initial_value - var_95),
    )


app = FastAPI(title="Monte Carlo Risk Inference API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    try:
        cached = load_params()
        return {
            "status": "ok",
            "params_loaded": True,
            "tickers": cached["tickers"],
            "params_as_of": cached["as_of"],
        }
    except Exception as exc:
        return {"status": "error", "params_loaded": False, "error": str(exc)}


@app.post("/simulate", response_model=SimulationResponse)
def simulate(payload: SimulationRequest) -> SimulationResponse:
    try:
        cached = load_params()
        return run_simulation(payload, cached)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}") from exc
