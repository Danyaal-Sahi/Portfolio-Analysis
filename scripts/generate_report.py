from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_lab.data import fetch_prices, read_holdings_csv, to_returns
from portfolio_lab.estimate import covariance_matrix, expected_returns
from portfolio_lab.metrics import run_historical_stress, summary_table, portfolio_returns
from portfolio_lab.optimize import max_sharpe
from portfolio_lab.report import generate_pdf_report


STRESS_WINDOWS = [
    ("GFC (2008-2009)", "2008-09-01", "2009-03-31"),
    ("COVID Crash (2020)", "2020-02-20", "2020-04-30"),
    ("Inflation/Rates (2022)", "2022-01-03", "2022-10-14"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", required=True, help="Path to holdings CSV")
    ap.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    ap.add_argument("--rf", type=float, default=0.02, help="Annual risk-free rate (e.g., 0.02)")
    ap.add_argument("--mu", default="historical", choices=["historical", "ewma"], help="Expected return method")
    ap.add_argument("--cov", default="sample", choices=["sample", "ledoit_wolf"], help="Covariance method")
    ap.add_argument("--out", default="reports", help="Output directory")
    args = ap.parse_args()

    holdings = read_holdings_csv(args.holdings)
    tickers = holdings["ticker"].tolist()

    prices = fetch_prices(tickers, start=args.start, end=args.end).prices
    rets = to_returns(prices, method="simple")

    weights = holdings.set_index("ticker")["weight"]
    port_daily = portfolio_returns(rets, weights)

    mu = expected_returns(rets, method=args.mu)
    try:
        cov = covariance_matrix(rets, method=args.cov)
    except Exception as e:
        if args.cov == "ledoit_wolf":
            print(f"Covariance 'ledoit_wolf' failed ({e}); falling back to sample covariance.")
            cov = covariance_matrix(rets, method="sample")
        else:
            raise

    opt = max_sharpe(mu, cov, rf_annual=args.rf, prev_weights=weights, turnover_penalty=0.25, max_weight=0.35)
    rebalance = (
        pd.DataFrame(
            {
                "ticker": opt.weights.index,
                "current_weight": weights.reindex(opt.weights.index).fillna(0.0).astype(float).values,
                "target_weight": opt.weights.values,
            }
        )
        .assign(trade_weight=lambda d: d["target_weight"] - d["current_weight"])
        .sort_values("trade_weight", key=lambda s: s.abs(), ascending=False)
    )

    summary = summary_table(port_daily, rf_annual=args.rf)
    stresses = []
    for name, s, e in STRESS_WINDOWS:
        res = run_historical_stress(port_daily, name=name, start=s, end=e)
        stresses.append(res.__dict__)
    stress_df = pd.DataFrame(stresses)

    out_dir = Path(args.out)
    report = generate_pdf_report(
        title="Portfolio Review",
        holdings=holdings,
        summary_table=summary,
        portfolio_daily=port_daily,
        stress_rows=stress_df,
        rebalance_table=rebalance,
        out_dir=out_dir,
    )
    print(f"Wrote {report.pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
