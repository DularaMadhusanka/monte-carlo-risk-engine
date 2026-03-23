import os
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://monte-carlo-risk-engine.onrender.com")
REQUEST_TIMEOUT_SECONDS = 100

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


st.title("📈 Portfolio Risk Engine (Monte Carlo)")
st.write("Powered by FastAPI & Geometric Brownian Motion")

st.sidebar.header("Simulation Parameters")
api_url = st.sidebar.text_input("FastAPI URL", value=API_URL)
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

if st.button("Run Risk Simulation", type="primary"):
    payload = {
        "initial_value": float(initial_value),
        "years": float(years),
        "sims": int(sims),
        "weights": normalized_weights,
    }

    with st.spinner("Calculating parallel universes..."):
        try:
            data = call_simulation(api_url, payload)

            st.success("Simulation Complete!")

            st.subheader(f"Simulation Results (As of {data['params_as_of']})")
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
