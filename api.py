import os
from functools import lru_cache
from typing import List, Optional

import numpy as np
import yfinance as yf
from scipy.stats import chi2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


TRADING_DAYS = 252
DEFAULT_PARAMS_PATH = os.getenv("PARAMS_PATH", "artifacts/market_params.npz")
STUDENT_T_DOF = 4


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


class BacktestRequest(BaseModel):
    initial_value: float = Field(gt=0, description="Initial portfolio value")
    backtest_days: int = Field(default=252, gt=0, le=1000, description="Number of rolling days to test")
    window_days: int = Field(default=252, gt=30, le=756, description="Lookback window (trading days)")
    sims: int = Field(default=1000, gt=100, le=5000, description="Monte Carlo paths per day")
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0, description="VaR confidence level (e.g., 0.95)")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    weights: Optional[List[float]] = Field(
        default=None,
        description="Optional portfolio weights. Must match number of tickers and sum to 1.",
    )
    tickers: Optional[List[str]] = Field(
        default=None, description="Optional tickers. If omitted, uses cached tickers."
    )


class KupiecPOFResult(BaseModel):
    N: int
    x: int
    p: float
    p_hat: float
    expected_exceptions: float
    LR_pof: float
    p_value: float
    pass_: bool


class BacktestResponse(BaseModel):
    tickers: List[str]
    params_as_of: str
    initial_value: float
    window_days: int
    backtest_days: int
    confidence: float
    alpha: float
    dates: List[str]
    actual_returns: List[float]
    var_returns: List[float]
    exceptions: List[bool]
    kupiec_pof: KupiecPOFResult


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
    t_scaling = np.sqrt((STUDENT_T_DOF - 2.0) / STUDENT_T_DOF)
    z = rng.standard_t(df=STUDENT_T_DOF, size=(steps, payload.sims, n_assets)) * t_scaling
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


def _validate_and_normalize_weights(weights: Optional[List[float]], n_assets: int) -> np.ndarray:
    if weights is None:
        return np.repeat(1.0 / n_assets, n_assets)

    w = np.array(weights, dtype=float)
    if len(w) != n_assets:
        raise HTTPException(
            status_code=400,
            detail=f"weights length must be {n_assets}, got {len(w)}",
        )
    if np.any(w < 0):
        raise HTTPException(status_code=400, detail="weights must be non-negative")
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise HTTPException(status_code=400, detail="weights must sum to 1.0")
    return w


def _fetch_close_prices(tickers: List[str], period: str = "3y") -> tuple[list[str], np.ndarray]:
    close_data = yf.download(tickers, period=period, interval="1d", progress=False)["Close"]

    # yfinance returns a Series for a single ticker and a DataFrame for multiple tickers.
    if hasattr(close_data, "columns"):
        close_df = close_data
    else:
        close_df = close_data.to_frame(name=tickers[0])

    close_df = close_df.dropna(how="any")
    if close_df.empty:
        raise HTTPException(status_code=500, detail="No price history returned for backtest.")

    dates = [str(d.date()) for d in close_df.index]
    prices = close_df.to_numpy(dtype=float)  # (T, n_assets)
    return dates, prices


def _kupiec_pof_test(exceptions: np.ndarray, confidence: float) -> KupiecPOFResult:
    # Kupiec POF uses the exception probability p = alpha = 1 - confidence.
    p = float(1.0 - confidence)
    N = int(exceptions.shape[0])
    x = int(np.sum(exceptions))
    p_hat = x / N if N > 0 else 0.0

    expected_exceptions = float(N * p)

    # Clip to avoid log(0) when p_hat is 0 or 1.
    eps = 1e-12
    p_hat_safe = float(np.clip(p_hat, eps, 1.0 - eps))
    one_minus_p = float(max(1.0 - p, eps))

    logL0 = (N - x) * np.log(one_minus_p) + x * np.log(max(p, eps))
    logL1 = (N - x) * np.log(float(max(1.0 - p_hat_safe, eps))) + x * np.log(float(max(p_hat_safe, eps)))

    LR = float(-2.0 * (logL0 - logL1))
    p_value = float(chi2.sf(LR, df=1))
    critical = float(chi2.ppf(0.95, df=1))  # 3.84-ish

    return KupiecPOFResult(
        N=N,
        x=x,
        p=p,
        p_hat=float(p_hat),
        expected_exceptions=expected_exceptions,
        LR_pof=LR,
        p_value=p_value,
        pass_=LR <= critical,
    )


def run_backtest(payload: BacktestRequest, cached: dict) -> BacktestResponse:
    tickers = payload.tickers if payload.tickers is not None else cached["tickers"]
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided for backtest.")

    weights = _validate_and_normalize_weights(payload.weights, n_assets=len(tickers))

    window_days = int(payload.window_days)
    backtest_days = int(payload.backtest_days)
    required_len = window_days + backtest_days + 1  # need i+1

    dates_all, prices_all = _fetch_close_prices(tickers=tickers, period="3y")
    if len(dates_all) < required_len:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough price history for backtest. Need at least {required_len} trading days.",
        )

    prices = prices_all[-required_len:]
    dates = dates_all[-required_len:]

    log_returns = np.log(prices[1:] / prices[:-1])  # (T-1, n_assets)

    start_i = window_days
    sims = int(payload.sims)
    confidence = float(payload.confidence)
    dt = 1.0 / TRADING_DAYS
    t_scaling = np.sqrt((STUDENT_T_DOF - 2.0) / STUDENT_T_DOF)

    # Fixed-share portfolio throughout the backtest horizon.
    s0_test0 = prices[start_i]
    shares_owned = (float(payload.initial_value) * weights) / s0_test0  # (n_assets,)

    actual_returns = np.zeros(backtest_days, dtype=float)
    var_returns = np.zeros(backtest_days, dtype=float)
    exceptions = np.zeros(backtest_days, dtype=bool)

    rng = np.random.default_rng(payload.seed)

    for j in range(backtest_days):
        i = start_i + j
        window_returns = log_returns[i - window_days : i]  # (window_days, n_assets)

        daily_mean = window_returns.mean(axis=0)  # (n_assets,)
        daily_cov = np.cov(window_returns, rowvar=False, bias=False)
        daily_cov = np.atleast_2d(daily_cov)

        annual_cov = daily_cov * TRADING_DAYS
        annual_cov = annual_cov + 1e-12 * np.eye(len(tickers))
        annual_mu = daily_mean * TRADING_DAYS + 0.5 * np.diag(annual_cov)

        chol_L = np.linalg.cholesky(annual_cov)

        z = rng.standard_t(df=STUDENT_T_DOF, size=(sims, len(tickers))) * t_scaling
        correlated_shocks = (z @ chol_L.T) * np.sqrt(dt)  # (sims, n_assets)

        drift_term = (annual_mu - 0.5 * np.diag(annual_cov)) * dt  # (n_assets,)
        daily_growth = np.exp(drift_term.reshape(1, -1) + correlated_shocks)  # (sims, n_assets)

        s0 = prices[i]  # (n_assets,)
        portfolio_now = float(np.sum(prices[i] * shares_owned))
        portfolio_next_actual = float(np.sum(prices[i + 1] * shares_owned))

        portfolio_next_sim = np.sum((s0.reshape(1, -1) * daily_growth) * shares_owned.reshape(1, -1), axis=1)
        var_value = float(np.percentile(portfolio_next_sim, 100.0 * (1.0 - confidence)))

        actual_ret = portfolio_next_actual / portfolio_now - 1.0
        var_ret = var_value / portfolio_now - 1.0

        actual_returns[j] = actual_ret
        var_returns[j] = var_ret
        exceptions[j] = actual_ret <= var_ret

    kupiec = _kupiec_pof_test(exceptions=exceptions, confidence=confidence)

    # Output dates correspond to the realized day (i+1).
    dates_out = dates[start_i + 1 : start_i + 1 + backtest_days]

    return BacktestResponse(
        tickers=tickers,
        params_as_of=cached["as_of"],
        initial_value=float(payload.initial_value),
        window_days=window_days,
        backtest_days=backtest_days,
        confidence=confidence,
        alpha=float(1.0 - confidence),
        dates=dates_out,
        actual_returns=actual_returns.tolist(),
        var_returns=var_returns.tolist(),
        exceptions=exceptions.tolist(),
        kupiec_pof=kupiec,
    )


@app.post("/backtest", response_model=BacktestResponse)
def backtest(payload: BacktestRequest) -> BacktestResponse:
    try:
        cached = load_params()
        return run_backtest(payload, cached)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc
