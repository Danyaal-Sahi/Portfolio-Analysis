from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class PriceFetchResult:
    prices: pd.DataFrame
    tickers: list[str]


def read_holdings_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("Holdings CSV must include a 'ticker' column.")

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"] != ""].copy()
    if df.empty:
        raise ValueError("Holdings CSV has no valid tickers.")

    if "weight" not in df.columns:
        df["weight"] = 1.0 / len(df)
    else:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        if df["weight"].isna().any():
            raise ValueError("Found non-numeric weights in 'weight' column.")
        total = float(df["weight"].sum())
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        df["weight"] = df["weight"] / total

    if "asset_class" not in df.columns:
        df["asset_class"] = "Unclassified"
    else:
        df["asset_class"] = df["asset_class"].astype(str).fillna("Unclassified")

    return df[["ticker", "weight", "asset_class"]]


def _default_cache_dir() -> Path:
    return Path(".cache") / "prices"


def fetch_prices(
    tickers: Iterable[str],
    start: str | date,
    end: str | date | None = None,
    *,
    cache_dir: str | Path | None = None,
    auto_adjust: bool = True,
) -> PriceFetchResult:
    ticker_list = sorted({str(t).strip() for t in tickers if str(t).strip()})
    if not ticker_list:
        raise ValueError("No tickers provided.")

    cache_path = (Path(cache_dir) if cache_dir else _default_cache_dir())
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_key = f"{pd.to_datetime(start).date().isoformat()}_{pd.to_datetime(end).date().isoformat() if end else 'latest'}_{'adj' if auto_adjust else 'raw'}"
    file_path = cache_path / (",".join(ticker_list).replace("/", "_") + f"__{cache_key}.pkl")

    if file_path.exists():
        prices = pd.read_pickle(file_path)
        return PriceFetchResult(prices=prices, tickers=ticker_list)

    data = yf.download(
        ticker_list,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs("Close", axis=1, level=0, drop_level=True)
    else:
        if "Close" not in data.columns:
            raise ValueError("Unexpected price data format from yfinance (missing Close).")
        prices = data["Close"].to_frame(name=ticker_list[0])

    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("No price data returned. Check tickers and date range.")

    prices.to_pickle(file_path)
    return PriceFetchResult(prices=prices, tickers=ticker_list)


def to_returns(prices: pd.DataFrame, *, method: str = "log") -> pd.DataFrame:
    px = prices.sort_index().ffill().dropna(how="all")
    if method == "simple":
        rets = px.pct_change()
    elif method == "log":
        rets = np.log(px / px.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'")
    return rets.dropna(how="all")
