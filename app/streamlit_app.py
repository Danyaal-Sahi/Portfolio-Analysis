from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_lab.data import fetch_prices, read_holdings_csv, to_returns
from portfolio_lab.estimate import covariance_matrix, expected_returns
from portfolio_lab.metrics import (
    contribution_by_asset,
    portfolio_returns,
    run_historical_stress,
    sleeve_weights,
    summary_table,
)
from portfolio_lab.optimize import max_sharpe, min_variance, risk_parity


STRESS_WINDOWS = [
    ("GFC (2008-2009)", "2008-09-01", "2009-03-31"),
    ("COVID Crash (2020)", "2020-02-20", "2020-04-30"),
    ("Inflation/Rates (2022)", "2022-01-03", "2022-10-14"),
]


st.set_page_config(page_title="Portfolio Construction & Risk Lab", layout="wide")
st.title("Portfolio Construction & Risk Lab")
st.caption("Upload holdings → analyze risk/performance → stress test → propose a target portfolio + rebalance trades.")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Holdings CSV", type=["csv"])
    use_sample = st.toggle("Use sample holdings", value=uploaded is None)
    start = st.text_input("Start date", value="2018-01-01")
    end = st.text_input("End date (optional)", value="")
    rf = st.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.20, value=0.02, step=0.005)
    ret_method = st.selectbox("Return type", ["simple", "log"], index=0)
    mu_method = st.selectbox("Expected returns", ["historical", "ewma"], index=0)
    cov_method = st.selectbox("Covariance", ["sample", "ledoit_wolf"], index=0)

    st.header("Target Portfolio")
    opt_method = st.selectbox("Method", ["Max Sharpe", "Min Variance", "Risk Parity"], index=0)
    max_weight = st.slider("Max single weight", min_value=0.10, max_value=1.0, value=0.35, step=0.05)
    turnover_penalty = st.slider("Turnover penalty", min_value=0.0, max_value=2.0, value=0.25, step=0.05)


def load_holdings() -> pd.DataFrame:
    if use_sample:
        return read_holdings_csv("data/sample_holdings.csv")
    if uploaded is None:
        st.stop()
    df = pd.read_csv(uploaded)
    tmp_path = "_uploaded_holdings.csv"
    df.to_csv(tmp_path, index=False)
    return read_holdings_csv(tmp_path)


holdings = load_holdings()
tickers = holdings["ticker"].tolist()

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Holdings")
    st.dataframe(holdings, use_container_width=True)
with col2:
    st.subheader("Sleeve Weights")
    sw = sleeve_weights(holdings)
    st.dataframe(sw.rename("weight").to_frame(), use_container_width=True)

end_arg = end.strip() or None
with st.spinner("Fetching prices..."):
    prices = fetch_prices(tickers, start=start, end=end_arg).prices
rets = to_returns(prices, method=ret_method)

weights = holdings.set_index("ticker")["weight"]
port_daily = portfolio_returns(rets, weights)
equity = (1.0 + port_daily).cumprod()

st.subheader("Performance")
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Portfolio"))
fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Growth of $1")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Risk & Return Summary")
st.dataframe(summary_table(port_daily, rf_annual=rf), use_container_width=True)

st.subheader("Average Return Contribution (by holding)")
contrib = contribution_by_asset(rets, weights)
st.dataframe(contrib.rename("avg daily contrib").to_frame(), use_container_width=True)

st.subheader("Historical Stress Tests (portfolio path)")
stress_rows = []
for name, s, e in STRESS_WINDOWS:
    res = run_historical_stress(port_daily, name=name, start=s, end=e)
    stress_rows.append(res.__dict__)
st.dataframe(pd.DataFrame(stress_rows), use_container_width=True)

st.subheader("Target Portfolio + Rebalance")
mu = expected_returns(rets, method=mu_method)
try:
    cov = covariance_matrix(rets, method=cov_method)
except Exception as e:
    st.warning(f"Covariance '{cov_method}' failed ({e}); falling back to sample covariance.")
    cov = covariance_matrix(rets, method="sample")

if opt_method == "Max Sharpe":
    opt = max_sharpe(mu, cov, rf_annual=rf, prev_weights=weights, turnover_penalty=turnover_penalty, max_weight=max_weight)
    target = opt.weights
elif opt_method == "Min Variance":
    opt = min_variance(cov, max_weight=max_weight)
    target = opt.weights
else:
    opt = risk_parity(cov)
    target = opt.weights

rebalance = (
    pd.DataFrame(
        {
            "ticker": target.index,
            "current_weight": weights.reindex(target.index).fillna(0.0).astype(float).values,
            "target_weight": target.values,
        }
    )
    .assign(trade_weight=lambda d: d["target_weight"] - d["current_weight"])
    .sort_values("trade_weight", key=lambda s: s.abs(), ascending=False)
)

st.caption(f"Optimizer status: {opt.status}")
st.dataframe(rebalance, use_container_width=True)
