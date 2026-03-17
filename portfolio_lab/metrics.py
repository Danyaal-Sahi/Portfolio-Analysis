from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def annualize_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0)


def annualize_vol(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(daily_returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_daily
    denom = excess.std(ddof=1)
    if denom == 0:
        return float("nan")
    return float(excess.mean() / denom * np.sqrt(periods_per_year))


def max_drawdown(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def var_cvar(daily_returns: pd.Series, level: float = 0.95) -> tuple[float, float]:
    r = daily_returns.dropna()
    if r.empty:
        return (float("nan"), float("nan"))
    q = float(np.quantile(r, 1.0 - level))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else float("nan")
    return (q, cvar)


@dataclass(frozen=True)
class StressResult:
    name: str
    start: str
    end: str
    cumulative_return: float
    max_drawdown: float


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    common = asset_returns.columns.intersection(weights.index)
    if len(common) == 0:
        raise ValueError("No overlap between returns columns and weights index.")
    w = weights.loc[common].astype(float)
    w = w / w.sum()
    return (asset_returns[common].fillna(0.0) * w).sum(axis=1)


def sleeve_weights(holdings: pd.DataFrame) -> pd.Series:
    by = holdings.groupby("asset_class", dropna=False)["weight"].sum()
    return by / float(by.sum())


def contribution_by_asset(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    common = returns.columns.intersection(weights.index)
    w = weights.loc[common].astype(float)
    w = w / w.sum()
    return (returns[common].fillna(0.0).mean() * w).sort_values(ascending=False)


def risk_contributions(cov: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    port_var = float(w.T @ cov @ w)
    if port_var <= 0:
        return np.full_like(w, np.nan)
    mrc = cov @ w  # marginal contribution to variance
    rc = w * mrc / port_var
    return rc


def summary_table(port_daily: pd.Series, *, rf_annual: float = 0.0) -> pd.DataFrame:
    ann_ret = annualize_return(port_daily)
    ann_vol = annualize_vol(port_daily)
    sr = sharpe_ratio(port_daily, rf_annual=rf_annual)
    mdd = max_drawdown(port_daily)
    var95, cvar95 = var_cvar(port_daily, level=0.95)
    return pd.DataFrame(
        {
            "Metric": ["Ann. return", "Ann. vol", "Sharpe", "Max drawdown", "VaR 95% (daily)", "CVaR 95% (daily)"],
            "Value": [ann_ret, ann_vol, sr, mdd, var95, cvar95],
        }
    )


def run_historical_stress(
    portfolio_daily: pd.Series,
    name: str,
    start: str,
    end: str,
) -> StressResult:
    window = portfolio_daily.loc[start:end].dropna()
    if window.empty:
        return StressResult(name=name, start=start, end=end, cumulative_return=float("nan"), max_drawdown=float("nan"))
    cum = float((1.0 + window).prod() - 1.0)
    mdd = max_drawdown(window)
    return StressResult(name=name, start=start, end=end, cumulative_return=cum, max_drawdown=mdd)

