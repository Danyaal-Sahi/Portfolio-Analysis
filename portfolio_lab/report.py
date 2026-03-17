from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


@dataclass(frozen=True)
class ReportArtifacts:
    pdf_path: Path
    charts_dir: Path


def _save_equity_curve_chart(portfolio_daily: pd.Series, out_path: Path) -> None:
    r = portfolio_daily.dropna()
    equity = (1.0 + r).cumprod()
    plt.figure(figsize=(9, 3))
    plt.plot(equity.index, equity.values)
    plt.title("Equity Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_drawdown_chart(portfolio_daily: pd.Series, out_path: Path) -> None:
    r = portfolio_daily.dropna()
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    plt.figure(figsize=(9, 2.8))
    plt.fill_between(dd.index, dd.values, 0.0)
    plt.title("Drawdown")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def generate_pdf_report(
    *,
    title: str,
    holdings: pd.DataFrame,
    summary_table: pd.DataFrame,
    portfolio_daily: pd.Series,
    stress_rows: pd.DataFrame,
    rebalance_table: pd.DataFrame | None,
    out_dir: str | Path,
) -> ReportArtifacts:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    equity_path = charts_dir / "equity_curve.png"
    dd_path = charts_dir / "drawdown.png"
    _save_equity_curve_chart(portfolio_daily, equity_path)
    _save_drawdown_chart(portfolio_daily, dd_path)

    pdf_path = out_dir / "portfolio_review.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    y = height - 0.75 * inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.75 * inch, y, title)

    y -= 0.4 * inch
    c.setFont("Helvetica", 10)
    c.drawString(0.75 * inch, y, f"Holdings count: {len(holdings)}")

    y -= 0.3 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, y, "Key Risk & Return")

    y -= 0.2 * inch
    c.setFont("Helvetica", 9)
    for _, row in summary_table.iterrows():
        metric = str(row["Metric"])
        val = row["Value"]
        try:
            val_str = f"{float(val):.4f}"
        except Exception:
            val_str = str(val)
        c.drawString(0.85 * inch, y, f"{metric}: {val_str}")
        y -= 0.17 * inch

    y -= 0.1 * inch
    c.drawImage(str(equity_path), 0.75 * inch, y - 2.1 * inch, width=7.0 * inch, height=2.1 * inch)
    y -= 2.25 * inch
    c.drawImage(str(dd_path), 0.75 * inch, y - 2.0 * inch, width=7.0 * inch, height=2.0 * inch)
    y -= 2.2 * inch

    c.showPage()

    y = height - 0.75 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, y, "Historical Stress Tests")
    y -= 0.3 * inch
    c.setFont("Helvetica", 9)
    for _, row in stress_rows.iterrows():
        line = f"{row['name']} ({row['start']} to {row['end']}): cum={row['cumulative_return']:.2%}, mdd={row['max_drawdown']:.2%}"
        c.drawString(0.85 * inch, y, line)
        y -= 0.18 * inch

    if rebalance_table is not None and not rebalance_table.empty:
        y -= 0.2 * inch
        c.setFont("Helvetica-Bold", 12)
        c.drawString(0.75 * inch, y, "Rebalance Trades (Weights)")
        y -= 0.3 * inch
        c.setFont("Helvetica", 9)
        for _, row in rebalance_table.head(25).iterrows():
            c.drawString(
                0.85 * inch,
                y,
                f"{row['ticker']}: current={row['current_weight']:.2%} -> target={row['target_weight']:.2%} (trade {row['trade_weight']:+.2%})",
            )
            y -= 0.17 * inch

    c.save()
    return ReportArtifacts(pdf_path=pdf_path, charts_dir=charts_dir)

