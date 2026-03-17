from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    status: str
    fun: float


def _portfolio_stats(mu: np.ndarray, cov: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    exp_ret = float(w @ mu)
    vol = float(np.sqrt(max(w.T @ cov @ w, 0.0)))
    return exp_ret, vol


def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    *,
    rf_annual: float = 0.0,
    bounds: tuple[float, float] = (0.0, 1.0),
    max_weight: float | None = None,
    prev_weights: Optional[pd.Series] = None,
    turnover_penalty: float = 0.0,
) -> OptimizationResult:
    assets = mu.index.intersection(cov.index).intersection(cov.columns)
    if len(assets) < 2:
        raise ValueError("Need at least 2 assets with mu and cov to optimize.")

    mu_v = mu.loc[assets].astype(float).to_numpy()
    cov_m = cov.loc[assets, assets].astype(float).to_numpy()

    n = len(assets)
    w0 = np.repeat(1.0 / n, n)
    if prev_weights is not None:
        prev = prev_weights.reindex(assets).fillna(0.0).to_numpy()
        if prev.sum() > 0:
            w0 = prev / prev.sum()

    lo, hi = bounds
    if max_weight is not None:
        hi = min(hi, float(max_weight))
    bnds = [(lo, hi) for _ in range(n)]

    def objective(w: np.ndarray) -> float:
        exp_ret, vol = _portfolio_stats(mu_v, cov_m, w)
        if vol <= 0:
            return 1e6
        sharpe = (exp_ret - rf_annual) / vol
        penalty = 0.0
        if prev_weights is not None and turnover_penalty > 0:
            penalty = turnover_penalty * float(np.sum(np.abs(w - prev)))
        return -sharpe + penalty

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(objective, w0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 200})
    w_opt = pd.Series(res.x, index=assets)
    w_opt = w_opt.clip(lower=0.0)
    if w_opt.sum() > 0:
        w_opt = w_opt / float(w_opt.sum())
    return OptimizationResult(weights=w_opt, status=str(res.message), fun=float(res.fun))


def min_variance(
    cov: pd.DataFrame,
    *,
    bounds: tuple[float, float] = (0.0, 1.0),
    max_weight: float | None = None,
) -> OptimizationResult:
    assets = cov.index.intersection(cov.columns)
    if len(assets) < 2:
        raise ValueError("Need at least 2 assets to optimize.")

    cov_m = cov.loc[assets, assets].astype(float).to_numpy()
    n = len(assets)
    w0 = np.repeat(1.0 / n, n)

    lo, hi = bounds
    if max_weight is not None:
        hi = min(hi, float(max_weight))
    bnds = [(lo, hi) for _ in range(n)]

    def objective(w: np.ndarray) -> float:
        return float(w.T @ cov_m @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(objective, w0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 200})
    w_opt = pd.Series(res.x, index=assets)
    w_opt = w_opt.clip(lower=0.0)
    if w_opt.sum() > 0:
        w_opt = w_opt / float(w_opt.sum())
    return OptimizationResult(weights=w_opt, status=str(res.message), fun=float(res.fun))


def risk_parity(cov: pd.DataFrame, *, max_iter: int = 5000, tol: float = 1e-9) -> OptimizationResult:
    assets = cov.index.intersection(cov.columns)
    if len(assets) < 2:
        raise ValueError("Need at least 2 assets to optimize.")

    cov_m = cov.loc[assets, assets].astype(float).to_numpy()
    n = len(assets)
    w = np.repeat(1.0 / n, n)

    def port_var(wv: np.ndarray) -> float:
        return float(wv.T @ cov_m @ wv)

    for _ in range(max_iter):
        mrc = cov_m @ w
        rc = w * mrc
        target = port_var(w) / n
        grad = rc - target
        step = 0.05
        w_new = w - step * grad
        w_new = np.clip(w_new, 1e-8, None)
        w_new = w_new / float(np.sum(w_new))
        if float(np.max(np.abs(w_new - w))) < tol:
            w = w_new
            break
        w = w_new

    return OptimizationResult(weights=pd.Series(w, index=assets), status="ok", fun=float(port_var(w)))

