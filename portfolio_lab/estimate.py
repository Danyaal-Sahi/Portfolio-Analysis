from __future__ import annotations

import numpy as np
import pandas as pd


def expected_returns(returns: pd.DataFrame, *, method: str = "historical", span: int = 60) -> pd.Series:
    r = returns.dropna(how="all")
    if r.empty:
        raise ValueError("No returns data.")

    if method == "historical":
        mu_daily = r.mean()
    elif method == "ewma":
        mu_daily = r.ewm(span=span, adjust=False).mean().iloc[-1]
    else:
        raise ValueError("method must be 'historical' or 'ewma'")

    mu_annual = (1.0 + mu_daily).pow(252) - 1.0
    return mu_annual.astype(float)


def covariance_matrix(returns: pd.DataFrame, *, method: str = "sample") -> pd.DataFrame:
    r = returns.dropna(how="all")
    if r.empty:
        raise ValueError("No returns data.")

    if method == "sample":
        cov_daily = r.cov()
    elif method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Ledoit-Wolf requires scikit-learn. Install scikit-learn or use method='sample'.") from e

        lw = LedoitWolf().fit(r.dropna().to_numpy())
        cov_daily = pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns)
    else:
        raise ValueError("method must be 'sample' or 'ledoit_wolf'")

    return cov_daily * 252.0

