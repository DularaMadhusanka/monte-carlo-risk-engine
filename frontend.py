import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

API_URL = os.getenv("API_URL", "https://monte-carlo-risk-engine.onrender.com")
REQUEST_TIMEOUT_SECONDS = 100
TIINGO_API_URL = "https://api.tiingo.com/tiingo/daily"

st.set_page_config(page_title="Monte Carlo Risk Engine", layout="wide")


def build_visual_paths(
    initial_value: float,
    expected_final_value: float,
    years: float,
    n_paths: int = 60,
    n_steps: int = 120,
) -> np.ndarray:
    visual_years = max(years, 1e-6)
    dt = visual_years / n_steps

    implied_mu = np.log(max(expected_final_value, 1.0) / max(initial_value, 1.0)) / visual_years
    visual_sigma = 0.25

    random_noise = np.random.normal(0.0, 1.0, size=(n_steps, n_paths))
    increment = (implied_mu - 0.5 * visual_sigma**2) * dt + visual_sigma * np.sqrt(dt) * random_noise
    growth = np.exp(increment)

    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = initial_value
    for step in range(1, n_steps + 1):
        paths[step] = paths[step - 1] * growth[step - 1]

    return paths


def estimate_best_case_value(expected_final_value: float, median_final_value: float) -> float:
    safe_expected = max(float(expected_final_value), 1e-9)
    safe_median = max(float(median_final_value), 1e-9)

    if safe_expected <= safe_median:
        return safe_expected

    sigma_sq = max(0.0, 2.0 * np.log(safe_expected / safe_median))
    sigma = np.sqrt(sigma_sq)
    z_95 = 1.6448536269514722
    return float(safe_median * np.exp(z_95 * sigma))


@st.cache_data(ttl=30)
def get_health(api_url: str) -> Dict[str, Any]:
    response = requests.get(f"{api_url}/health", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def call_simulation(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{api_url}/simulate",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code == 200:
        return response.json()

    detail: Optional[str] = None
    try:
        detail = response.json().get("detail")
    except Exception:
        detail = response.text

    raise RuntimeError(f"API Error ({response.status_code}): {detail}")


def get_tiingo_api_key() -> Optional[str]:
    key_from_env = os.getenv("TIINGO_API_KEY")
    if key_from_env:
        return key_from_env

    try:
        key_from_secrets = st.secrets.get("TIINGO_API_KEY")
        if key_from_secrets:
            return str(key_from_secrets)
    except Exception:
        pass

    return None


def period_to_date_range(period: str) -> tuple[str, str]:
    lookback_days = {
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 366,
        "2y": 731,
        "5y": 1827,
    }
    days = lookback_days.get(period, 186)
    end_dt = datetime.now(timezone.utc).date()
    start_dt = end_dt - timedelta(days=days)
    return start_dt.isoformat(), end_dt.isoformat()


def get_historical_ohlcv_tiingo(ticker: str, period: str = "6mo") -> pd.DataFrame:
    api_key = get_tiingo_api_key()
    if not api_key:
        return pd.DataFrame()

    clean_ticker = str(ticker).strip().upper()
    start_date, end_date = period_to_date_range(period)
    response = requests.get(
        f"{TIINGO_API_URL}/{clean_ticker}/prices",
        params={
            "startDate": start_date,
            "endDate": end_date,
            "resampleFreq": "daily",
            "columns": "open,high,low,close,volume,date",
            "token": api_key,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code != 200:
        return pd.DataFrame()

    payload = response.json()
    if not isinstance(payload, list) or len(payload) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(payload)
    if df.empty:
        return pd.DataFrame()

    required = ["open", "high", "low", "close", "volume", "date"]
    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    ohlcv = df[required].copy()
    ohlcv.columns = ["Open", "High", "Low", "Close", "Volume", "Date"]
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"], errors="coerce")
    ohlcv = ohlcv.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    if ohlcv.empty:
        return pd.DataFrame()

    ohlcv = ohlcv.set_index("Date").sort_index()
    if getattr(ohlcv.index, "tz", None) is not None:
        ohlcv.index = ohlcv.index.tz_localize(None)
    return ohlcv[["Open", "High", "Low", "Close", "Volume"]]


def get_historical_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    del interval
    return get_historical_ohlcv_tiingo(ticker=ticker, period=period)


def build_candlestick_volume_figure(ohlcv: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )

    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["Open"],
            high=ohlcv["High"],
            low=ohlcv["Low"],
            close=ohlcv["Close"],
            name=f"{ticker} OHLC",
            increasing_line_color="#00cc96",
            decreasing_line_color="#ef553b",
        ),
        row=1,
        col=1,
    )

    volume_colors = np.where(ohlcv["Close"] >= ohlcv["Open"], "#00cc96", "#ef553b")
    fig.add_trace(
        go.Bar(
            x=ohlcv.index,
            y=ohlcv["Volume"],
            marker_color=volume_colors,
            name="Volume",
            opacity=0.85,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=10, r=10, t=50, b=10),
        title=f"{ticker} Historical Price (OHLC) + Volume",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def get_synthetic_portfolio_ohlcv(
    tickers: list[str],
    weights: list[float],
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    if len(tickers) == 0 or len(tickers) != len(weights):
        return pd.DataFrame()

    ticker_frames: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        ticker_ohlcv = get_historical_ohlcv(ticker=ticker, period=period, interval=interval)
        if ticker_ohlcv.empty:
            return pd.DataFrame()
        ticker_frames[ticker] = ticker_ohlcv

    combined = pd.concat(ticker_frames, axis=1, join="inner").dropna()
    if combined.empty:
        return pd.DataFrame()

    weights_array = np.array(weights, dtype=float)
    if np.sum(weights_array) <= 0:
        return pd.DataFrame()
    weights_array = weights_array / np.sum(weights_array)

    base_notional = 100.0
    initial_closes = np.array(
        [float(combined[(ticker, "Close")].iloc[0]) for ticker in tickers],
        dtype=float,
    )
    shares = (base_notional * weights_array) / np.maximum(initial_closes, 1e-9)

    synthetic_data: Dict[str, np.ndarray] = {}
    for field in ["Open", "High", "Low", "Close"]:
        field_matrix = np.column_stack(
            [combined[(ticker, field)].to_numpy(dtype=float) for ticker in tickers]
        )
        synthetic_data[field] = field_matrix @ shares

    volume_matrix = np.column_stack(
        [combined[(ticker, "Volume")].to_numpy(dtype=float) for ticker in tickers]
    )
    synthetic_data["Volume"] = volume_matrix @ weights_array

    return pd.DataFrame(synthetic_data, index=combined.index)


def render_simulation_results(
    data: Dict[str, Any],
    asset_labels: list[str],
    normalized_weights: list[float],
    show_success: bool = False,
) -> None:
    if show_success:
        st.success("Simulation Complete!")

    st.subheader(f"Simulation Results (As of {data['params_as_of']})")

    st.divider()
    st.subheader("Historical Market Context")
    chart_col1, chart_col2, chart_col3 = st.columns([2, 1, 1])

    ticker_default = st.session_state["chart_selected_ticker"]
    if ticker_default not in asset_labels:
        ticker_default = asset_labels[0]

    period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
    period_default = st.session_state["chart_selected_period"]
    if period_default not in period_options:
        period_default = "6mo"

    mode_options = ["Both", "Asset Only", "Portfolio Only"]
    mode_default = st.session_state["chart_selected_mode"]
    if mode_default not in mode_options:
        mode_default = "Both"

    with chart_col1:
        selected_ticker = st.selectbox(
            "Asset",
            options=asset_labels,
            index=asset_labels.index(ticker_default),
        )
    with chart_col2:
        selected_period = st.selectbox(
            "History Range",
            options=period_options,
            index=period_options.index(period_default),
        )
    with chart_col3:
        chart_mode = st.selectbox(
            "Chart View",
            options=mode_options,
            index=mode_options.index(mode_default),
        )

    st.session_state["chart_selected_ticker"] = selected_ticker
    st.session_state["chart_selected_period"] = selected_period
    st.session_state["chart_selected_mode"] = chart_mode

    if chart_mode in ("Both", "Asset Only"):
        with st.spinner(f"Loading {selected_ticker} candles..."):
            ohlcv = get_historical_ohlcv(selected_ticker, period=selected_period)

        if ohlcv.empty:
            if not get_tiingo_api_key():
                st.error(
                    "Tiingo API key is missing. Add `TIINGO_API_KEY` to Streamlit secrets "
                    "or environment variables."
                )
            else:
                st.warning(
                    f"Could not fetch Tiingo OHLCV data for {selected_ticker} in {selected_period}. "
                    "Verify API key validity and ticker support."
                )
        else:
            candle_fig = build_candlestick_volume_figure(ohlcv, selected_ticker)
            st.plotly_chart(candle_fig, use_container_width=True)

    if chart_mode in ("Both", "Portfolio Only"):
        with st.spinner("Building synthetic portfolio candles..."):
            portfolio_ohlcv = get_synthetic_portfolio_ohlcv(
                tickers=asset_labels,
                weights=normalized_weights,
                period=selected_period,
            )

        if portfolio_ohlcv.empty:
            st.warning(
                "Unable to build synthetic portfolio OHLCV from Tiingo data. "
                "Check API key and ticker data availability."
            )
        else:
            portfolio_fig = build_candlestick_volume_figure(
                portfolio_ohlcv,
                "Synthetic Portfolio",
            )
            st.plotly_chart(portfolio_fig, use_container_width=True)
            st.caption(
                "Synthetic portfolio candles are built from fixed-share weighted prices "
                "(based on initial weights) across the selected history range."
            )

    expected_val = float(data["expected_final_value"])
    var_95_loss = float(data["max_potential_loss_95"])
    best_case = estimate_best_case_value(
        expected_final_value=expected_val,
        median_final_value=float(data["median_final_value"]),
    )

    st.divider()
    st.subheader("Portfolio Risk Summary")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.metric(label="Expected Portfolio Value", value=f"${expected_val:,.2f}")
    with kpi_col2:
        st.metric(
            label="95% Value at Risk (VaR)",
            value=f"-${var_95_loss:,.2f}",
            delta="-Risk",
            delta_color="inverse",
        )
    with kpi_col3:
        st.metric(
            label="Best Case Scenario (5%)",
            value=f"${best_case:,.2f}",
            delta="+Upside",
        )

    tab_summary, tab_animation, tab_dashboard, tab_raw = st.tabs(
        ["Summary", "Animation", "Dashboard", "Raw Data"]
    )

    with tab_summary:
        col1, col2, col3 = st.columns(3)
        col1.metric("Initial Portfolio Value", f"${data['initial_value']:,.2f}")
        col2.metric("Median Value", f"${data['median_final_value']:,.2f}")
        col3.metric(
            "Expected Shortfall (95% CVaR)",
            f"${data['cvar_95_expected_shortfall']:,.2f}",
        )

        st.markdown("### Tail Risk Metrics")
        col4, col5 = st.columns(2)
        col4.info(
            f"**Value at Risk (95% VaR):** ${data['var_95_threshold']:,.2f}\n\n"
            "*We are 95% confident the portfolio will not drop below this value.*"
        )
        col5.error(
            f"**Expected Shortfall (95% CVaR):** ${data['cvar_95_expected_shortfall']:,.2f}\n\n"
            "*If the worst 5% scenario happens, this is the average expected value.*"
        )

        st.markdown("### Outcome Snapshot")
        chart_df = pd.DataFrame(
            {
                "Metric": ["Initial", "Expected", "Median", "VaR 95%", "CVaR 95%"],
                "Portfolio Value": [
                    float(data["initial_value"]),
                    float(data["expected_final_value"]),
                    float(data["median_final_value"]),
                    float(data["var_95_threshold"]),
                    float(data["cvar_95_expected_shortfall"]),
                ],
            }
        ).set_index("Metric")
        st.bar_chart(chart_df)

    with tab_animation:
        st.subheader("Live Random Walk Preview")
        st.caption(
            "Sampled animation (60 paths) for visual intuition. "
            "Risk metrics still come from full backend simulation."
        )

        visual_paths = build_visual_paths(
            initial_value=float(data["initial_value"]),
            expected_final_value=float(data["expected_final_value"]),
            years=float(data["years"]),
            n_paths=60,
            n_steps=120,
        )

        path_names = [f"Path {index + 1}" for index in range(visual_paths.shape[1])]
        animate_placeholder = st.empty()

        for step in range(5, visual_paths.shape[0] + 1, 4):
            frame = pd.DataFrame(visual_paths[:step], columns=path_names)
            animate_placeholder.line_chart(frame)
            time.sleep(0.03)

    with tab_dashboard:
        st.subheader("Risk Dashboard")

        potential_loss_95 = float(data.get("max_potential_loss_95", 0.0))
        gauge_max = max(potential_loss_95 * 2.0, 1000.0)

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=potential_loss_95,
                title={"text": "Max Potential Loss (95%)", "font": {"size": 22}},
                number={"prefix": "$", "valueformat": ",.0f"},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar": {"color": "darkred"},
                    "steps": [
                        {"range": [0, gauge_max * 0.4], "color": "#d6f5d6"},
                        {"range": [gauge_max * 0.4, gauge_max * 0.7], "color": "#fff7cc"},
                        {"range": [gauge_max * 0.7, gauge_max], "color": "#ffd6d6"},
                    ],
                },
            )
        )
        gauge_fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))

        allocation_fig = go.Figure(
            data=[
                go.Pie(
                    labels=asset_labels,
                    values=normalized_weights,
                    hole=0.55,
                    textinfo="label+percent",
                )
            ]
        )
        allocation_fig.update_layout(
            title="Portfolio Allocation", height=360, margin=dict(l=10, r=10, t=50, b=10)
        )

        dashboard_col1, dashboard_col2 = st.columns(2)
        dashboard_col1.plotly_chart(gauge_fig, use_container_width=True)
        dashboard_col2.plotly_chart(allocation_fig, use_container_width=True)

        risk_bar_fig = go.Figure(
            data=[
                go.Bar(
                    x=["Expected", "Median", "VaR 95%", "CVaR 95%"],
                    y=[
                        float(data["expected_final_value"]),
                        float(data["median_final_value"]),
                        float(data["var_95_threshold"]),
                        float(data["cvar_95_expected_shortfall"]),
                    ],
                    marker_color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759"],
                )
            ]
        )
        risk_bar_fig.update_layout(
            title="Portfolio Value Comparison",
            yaxis_title="Value ($)",
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(risk_bar_fig, use_container_width=True)

    with tab_raw:
        st.json(data)


st.title("Portfolio Risk Engine (Monte Carlo)")
st.write("Powered by FastAPI & Geometric Brownian Motion")

st.sidebar.header("Simulation Parameters")
api_url = st.sidebar.text_input("FastAPI URL", value=API_URL)
if get_tiingo_api_key():
    st.sidebar.success("Market data source: Tiingo API key detected")
else:
    st.sidebar.warning("Market data source: Tiingo key missing (set TIINGO_API_KEY)")
initial_value = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=1000,
    value=10000,
    step=1000,
)
years = st.sidebar.slider(
    "Time Horizon (Years)",
    min_value=1,
    max_value=5,
    value=1,
    step=1,
)
sims = st.sidebar.slider(
    "Number of Simulations",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
)

try:
    health = get_health(api_url)
    if health.get("status") == "ok":
        tickers = health.get("tickers", [])
        assets_text = ", ".join(tickers) if tickers else "N/A"
        st.sidebar.success(f"Backend connected.\n\nAssets: {assets_text}")
    else:
        st.sidebar.error("Backend reported an error state.")
except requests.exceptions.RequestException as exc:
    st.sidebar.error(f"Cannot connect to backend: {exc}")

asset_labels = ["AAPL", "MSFT", "TSLA"]
if "health" in locals() and health.get("status") == "ok":
    tickers = health.get("tickers", [])
    if isinstance(tickers, list) and len(tickers) == 3:
        asset_labels = tickers

st.sidebar.header("Portfolio Allocation")
st.sidebar.write("Adjust asset weights. Values are auto-normalized to 100%.")

w_1 = st.sidebar.slider(f"{asset_labels[0]}", min_value=0, max_value=100, value=33, step=1)
w_2 = st.sidebar.slider(f"{asset_labels[1]}", min_value=0, max_value=100, value=33, step=1)
w_3 = st.sidebar.slider(f"{asset_labels[2]}", min_value=0, max_value=100, value=34, step=1)

total_weight = w_1 + w_2 + w_3
if total_weight == 0:
    st.sidebar.error("Total allocation cannot be 0%. Increase at least one asset weight.")
    st.stop()

normalized_weights = [w_1 / total_weight, w_2 / total_weight, w_3 / total_weight]
st.sidebar.caption(
    "Normalized: "
    f"{asset_labels[0]} {normalized_weights[0]:.1%}, "
    f"{asset_labels[1]} {normalized_weights[1]:.1%}, "
    f"{asset_labels[2]} {normalized_weights[2]:.1%}"
)

if "chart_selected_ticker" not in st.session_state:
    st.session_state["chart_selected_ticker"] = asset_labels[0]
if "chart_selected_period" not in st.session_state:
    st.session_state["chart_selected_period"] = "6mo"
if "chart_selected_mode" not in st.session_state:
    st.session_state["chart_selected_mode"] = "Both"

tab1, tab2 = st.tabs(["Main Dashboard", "Strategy Comparison (Ex-Ante)"])

with tab1:
    action_col1, action_col2, action_col3 = st.columns([2, 1, 1])
    run_clicked = action_col1.button("Run Risk Simulation", type="primary")
    clear_clicked = action_col2.button("Clear Saved Simulation")
    clear_all_clicked = action_col3.button("Clear All Dashboard State")

    if clear_clicked:
        st.session_state.pop("last_simulation_data", None)
        st.session_state.pop("last_simulation_asset_labels", None)
        st.session_state.pop("last_simulation_weights", None)
        st.info("Saved simulation has been cleared.")

    if clear_all_clicked:
        st.session_state.pop("last_simulation_data", None)
        st.session_state.pop("last_simulation_asset_labels", None)
        st.session_state.pop("last_simulation_weights", None)
        st.session_state.pop("strategy_duel_results", None)
        st.session_state["chart_selected_ticker"] = asset_labels[0]
        st.session_state["chart_selected_period"] = "6mo"
        st.session_state["chart_selected_mode"] = "Both"
        st.info("Dashboard state reset to defaults.")

    if run_clicked:
        payload = {
            "initial_value": float(initial_value),
            "years": float(years),
            "sims": int(sims),
            "weights": normalized_weights,
        }

        with st.spinner("Calculating parallel universes..."):
            try:
                data = call_simulation(api_url, payload)
                st.session_state["last_simulation_data"] = data
                st.session_state["last_simulation_asset_labels"] = asset_labels
                st.session_state["last_simulation_weights"] = normalized_weights
            except RuntimeError as exc:
                st.error(str(exc))
            except requests.exceptions.Timeout:
                st.error(
                    "The request timed out. The backend may be waking from cold start; "
                    "please retry in a few seconds."
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"Network error while calling API: {exc}")
            except Exception as exc:
                st.error(f"Failed to fetch data: {exc}")

    last_data = st.session_state.get("last_simulation_data")
    if isinstance(last_data, dict):
        last_asset_labels = st.session_state.get("last_simulation_asset_labels", asset_labels)
        if not isinstance(last_asset_labels, list) or len(last_asset_labels) != 3:
            last_asset_labels = asset_labels

        last_weights = st.session_state.get("last_simulation_weights", normalized_weights)
        if (
            not isinstance(last_weights, list)
            or len(last_weights) != 3
            or float(np.sum(last_weights)) <= 0
        ):
            last_weights = normalized_weights

        render_simulation_results(
            data=last_data,
            asset_labels=last_asset_labels,
            normalized_weights=last_weights,
            show_success=run_clicked,
        )

with tab2:
    st.header("Position Sizing Optimization")
    st.markdown(
        "Compare two different asset weight allocations to minimize your Value at Risk (VaR) "
        "before committing capital."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Strategy A")
        a_w1 = st.slider(f"{asset_labels[0]} %", 0, 100, 33, key="duel_a_w1")
        a_w2 = st.slider(f"{asset_labels[1]} %", 0, 100, 33, key="duel_a_w2")
        a_w3 = st.slider(f"{asset_labels[2]} %", 0, 100, 34, key="duel_a_w3")

    with col_b:
        st.subheader("Strategy B")
        b_w1 = st.slider(f"{asset_labels[0]} %", 0, 100, 50, key="duel_b_w1")
        b_w2 = st.slider(f"{asset_labels[1]} %", 0, 100, 40, key="duel_b_w2")
        b_w3 = st.slider(f"{asset_labels[2]} %", 0, 100, 10, key="duel_b_w3")

    a_total = a_w1 + a_w2 + a_w3
    b_total = b_w1 + b_w2 + b_w3

    a_weights = [a_w1 / a_total, a_w2 / a_total, a_w3 / a_total] if a_total > 0 else [0.0, 0.0, 0.0]
    b_weights = [b_w1 / b_total, b_w2 / b_total, b_w3 / b_total] if b_total > 0 else [0.0, 0.0, 0.0]

    norm_col_a, norm_col_b = st.columns(2)
    norm_col_a.caption(
        "Strategy A normalized: "
        f"{asset_labels[0]} {a_weights[0]:.1%}, "
        f"{asset_labels[1]} {a_weights[1]:.1%}, "
        f"{asset_labels[2]} {a_weights[2]:.1%}"
    )
    norm_col_b.caption(
        "Strategy B normalized: "
        f"{asset_labels[0]} {b_weights[0]:.1%}, "
        f"{asset_labels[1]} {b_weights[1]:.1%}, "
        f"{asset_labels[2]} {b_weights[2]:.1%}"
    )

    st.divider()

    duel_clicked = st.button("Run Strategy Duel", type="primary", key="run_strategy_duel")
    if duel_clicked:
        if a_total == 0 or b_total == 0:
            st.error("Both strategies must allocate above 0% total weight.")
        else:
            st.info(
                "Simulating both strategies... This sends 2 requests to the backend. "
                "For free-tier safety, keep simulations under 2,000 when possible."
            )

            payload_a = {
                "initial_value": float(initial_value),
                "years": float(years),
                "sims": int(sims),
                "weights": a_weights,
            }
            payload_b = {
                "initial_value": float(initial_value),
                "years": float(years),
                "sims": int(sims),
                "weights": b_weights,
            }

            with st.spinner("Running strategy comparison..."):
                try:
                    result_a = call_simulation(api_url, payload_a)
                    result_b = call_simulation(api_url, payload_b)
                    st.session_state["strategy_duel_results"] = {
                        "strategy_a": result_a,
                        "strategy_b": result_b,
                        "weights_a": a_weights,
                        "weights_b": b_weights,
                    }
                except RuntimeError as exc:
                    st.error(str(exc))
                except requests.exceptions.Timeout:
                    st.error(
                        "The strategy duel timed out. Backend may be waking from cold start; "
                        "retry in a few seconds."
                    )
                except requests.exceptions.RequestException as exc:
                    st.error(f"Network error while running strategy duel: {exc}")
                except Exception as exc:
                    st.error(f"Strategy duel failed: {exc}")

    duel_results = st.session_state.get("strategy_duel_results")
    if isinstance(duel_results, dict):
        strategy_a = duel_results.get("strategy_a", {})
        strategy_b = duel_results.get("strategy_b", {})

        if isinstance(strategy_a, dict) and isinstance(strategy_b, dict):
            var_a = float(strategy_a.get("max_potential_loss_95", 0.0))
            var_b = float(strategy_b.get("max_potential_loss_95", 0.0))
            expected_a = float(strategy_a.get("expected_final_value", 0.0))
            expected_b = float(strategy_b.get("expected_final_value", 0.0))
            var_savings = var_a - var_b

            st.subheader("Comparison Results")
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.metric(label="Strategy A (95% VaR)", value=f"-${var_a:,.2f}")

            with res_col2:
                delta_text = (
                    f"${abs(var_savings):,.2f} less risk!"
                    if var_savings > 0
                    else f"${abs(var_savings):,.2f} more risk"
                )
                st.metric(
                    label="Strategy B (95% VaR)",
                    value=f"-${var_b:,.2f}",
                    delta=delta_text,
                    delta_color="normal" if var_savings > 0 else "inverse",
                )

            comparison_fig = go.Figure(
                data=[
                    go.Bar(
                        name="Strategy A",
                        x=["Expected Value", "95% VaR Loss"],
                        y=[expected_a, var_a],
                        marker_color="#4e79a7",
                    ),
                    go.Bar(
                        name="Strategy B",
                        x=["Expected Value", "95% VaR Loss"],
                        y=[expected_b, var_b],
                        marker_color="#59a14f",
                    ),
                ]
            )
            comparison_fig.update_layout(
                barmode="group",
                title="Strategy A vs B: Expected Value and VaR",
                yaxis_title="USD ($)",
                height=360,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(comparison_fig, use_container_width=True)

            if var_savings > 0:
                st.success(
                    "Strategy B is mathematically safer. By reallocating these weights, "
                    "you reduce tail-risk exposure while maintaining market presence."
                )
            elif var_savings < 0:
                st.warning(
                    "Strategy A is currently safer on 95% VaR. Consider reducing risk "
                    "in Strategy B allocations."
                )
            else:
                st.info("Both strategies have identical 95% VaR under current simulation settings.")
