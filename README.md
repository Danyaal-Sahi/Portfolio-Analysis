# Portfolio Construction & Risk Lab
# https://portfolioanalysis-danyaal.streamlit.app

End-to-end portfolio construction project aligned to portfolio construction / portfolio consulting roles:

- Ingest holdings (ETFs or stocks)
- Compute performance + risk + exposures
- Run stress tests and scenarios
- Propose a target portfolio (with constraints) and generate rebalance trades
- Produce a client-ready PDF + an interactive dashboard

## Quick start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

### 3) Generate a PDF report (CLI)

```bash
python scripts/generate_report.py --holdings data/sample_holdings.csv --out reports
```

## Holdings format

CSV columns:

- `ticker` (required)
- `weight` (optional; if missing, equal weights)
- `asset_class` (optional; used for sleeve attribution)

Example: `data/sample_holdings.csv`.

## Notes

- Data source: Yahoo Finance (via `yfinance`).
- Optional: covariance shrinkage (`ledoit_wolf`) requires `scikit-learn` (not included in `requirements.txt`).
- This repo is intentionally lightweight and educational; do not use it for real trading or investment advice.
