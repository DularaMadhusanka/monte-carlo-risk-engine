import os
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 20

st.set_page_config(page_title="Monte Carlo Risk Engine", layout="wide")


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
api_url = st.sidebar.text_input("FastAPI URL", value=DEFAULT_API_URL)
initial_value = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=1000,
    value=10000,
    step=1000,
)
years = st.sidebar.slider(
    "Time Horizon (Years)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5,
)
sims = st.sidebar.select_slider(
    "Number of Simulations",
    options=[1000, 2000, 5000, 10000, 50000],
    value=2000,
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

if st.button("Run Risk Simulation", type="primary"):
    payload = {
        "initial_value": float(initial_value),
        "years": float(years),
        "sims": int(sims),
    }

    with st.spinner("Calculating parallel universes..."):
        try:
            data = call_simulation(api_url, payload)

            st.subheader(f"Simulation Results (As of {data['params_as_of']})")

            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Portfolio Value", f"${data['expected_final_value']:,.2f}")
            col2.metric("Median Value", f"${data['median_final_value']:,.2f}")
            col3.metric(
                "Max Potential Loss (95%)",
                f"${data['max_potential_loss_95']:,.2f}",
                delta="- Risk",
                delta_color="inverse",
            )

            st.divider()

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

            st.divider()
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

        except RuntimeError as exc:
            st.error(str(exc))
        except requests.exceptions.RequestException as exc:
            st.error(f"Network error while calling API: {exc}")
        except Exception as exc:
            st.error(f"Failed to fetch data: {exc}")
