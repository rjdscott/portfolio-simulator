# Data Dictionary

All datasets live under `data/`. Raw (baseline) data lives in `data/raw/`.

---

## Metadata Files

### `data/raw/metadata/sp500_constituents.csv`

Full S&P 500 constituent list as of the fetch date.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Exchange ticker symbol (e.g. `AAPL`) |
| `name` | str | Company name |
| `sector` | str | GICS sector |
| `sub_industry` | str | GICS sub-industry |
| `cik` | str | SEC CIK identifier |

### `data/raw/metadata/universe.csv`

**The benchmark universe** — the top 100 S&P 500 stocks by market cap that have
complete (≥ 95% of trading days) price history over the study window.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Exchange ticker symbol |
| `name` | str | Company name |
| `sector` | str | GICS sector |
| `market_cap_usd` | float | Market capitalisation (USD) at fetch date |
| `data_completeness` | float | Fraction of trading days with non-null prices |
| `rank` | int | Rank by market cap (1 = largest) |

### `data/raw/metadata/data_manifest.json`

Machine-readable record of the data fetch.

```json
{
  "fetch_date": "YYYY-MM-DD",
  "study_start": "2020-01-01",
  "study_end": "2024-12-31",
  "expected_trading_days": 1258,
  "universe_size": 100,
  "portfolio_k": 15,
  "yfinance_version": "x.x.x",
  "tickers": ["AAPL", "MSFT", ...],
  "excluded_tickers": ["XXXX", ...],
  "exclusion_reasons": {"XXXX": "< 95% data completeness"}
}
```

### `data/raw/metadata/validation_report.json`

Output of `src/data/validate.py`.

```json
{
  "run_timestamp": "...",
  "universe_size": 100,
  "trading_days": 1258,
  "missing_pct_per_ticker": {"AAPL": 0.0, ...},
  "excluded_tickers": [],
  "warnings": []
}
```

---

## Price Data

### `data/raw/prices/<TICKER>.csv`   _(one file per stock — CSV baseline)_

Daily adjusted-close prices for a single stock.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Trading day (YYYY-MM-DD) |
| `adj_close` | float64 | Adjusted closing price (USD) |

Notes:
- Rows are sorted ascending by `date`
- No missing values (excluded or forward-filled at fetch time)
- This is the **worst-case storage format**: 100 files, each requiring a separate
  open/parse/close cycle when building the price matrix

### `data/raw/prices/prices_long.csv`   _(combined long format)_

All stocks in a single file, long (tidy) format.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Trading day (YYYY-MM-DD) |
| `ticker` | str | Stock ticker symbol |
| `adj_close` | float64 | Adjusted closing price (USD) |

Dimensions: `100 stocks × 1258 days = 125,800 rows`

### `data/raw/prices/prices_wide.csv`   _(combined wide format)_

All stocks in a single file, wide (date-indexed) format. This is the natural
input format for matrix operations.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Trading day (index) |
| `AAPL` | float64 | AAPL adjusted close |
| `MSFT` | float64 | MSFT adjusted close |
| `...` | float64 | ... |

Dimensions: `1258 rows × 101 columns (date + 100 tickers)`

---

## Portfolio Data

### `data/raw/portfolios/portfolios_<N>.csv`   _(materialised weight matrix)_

Generated for N ≤ 10M. Shape: N rows × (1 + K) columns.

| Column | Type | Description |
|--------|------|-------------|
| `portfolio_id` | int64 | Unique portfolio index (0-based) |
| `<TICKER_1>` | float64 | Portfolio weight for stock 1 (0 < w < 1) |
| `...` | float64 | ... |
| `<TICKER_K>` | float64 | Portfolio weight for stock K |

Constraints:
- All weights are positive (> 0)
- Weights sum to 1.0 within floating-point precision
- Each row uses exactly K=15 tickers (non-zero weight columns)
- The K tickers selected per portfolio vary; zero-weight stocks are omitted
  (sparse representation)

**Note on sparsity**: To avoid a 1M × 100 dense matrix (where 85 of 100 weights
are zero), the CSV uses only the K active columns per portfolio. A loader must
handle the variable column structure.

For a **dense** version (1M × 100), see the alternative:
`data/raw/portfolios/portfolios_<N>_dense.csv` — all 100 ticker columns present,
zeros for non-selected stocks. This file is larger but simpler to load.

---

## Results Data

### `results/<run_id>.json`

Schema defined in `common/schemas/benchmark_result.schema.json`.

```json
{
  "run_id": "uuid4",
  "timestamp": "ISO-8601",
  "config": {
    "implementation": "numpy_vectorised",
    "language": "python",
    "storage_format": "csv_wide",
    "portfolio_scale": 1000000,
    "portfolio_k": 15,
    "universe_size": 100,
    "seed": 42,
    "repetitions": 5
  },
  "hardware": {
    "cpu_model": "...",
    "cpu_cores": 20,
    "ram_gb": 64,
    "gpu_model": "..."
  },
  "software": {
    "os": "...",
    "python_version": "...",
    "numpy_version": "...",
    "implementation_version": "..."
  },
  "timings_sec": {
    "load_prices": [0.12, 0.11, ...],
    "compute_returns": [3.4, 3.3, ...],
    "total": [3.52, 3.41, ...]
  },
  "summary": {
    "median_total_sec": 3.45,
    "p10_total_sec": 3.41,
    "p90_total_sec": 3.52,
    "throughput_portfolios_per_sec": 289855.0,
    "peak_ram_mb": 4200.0
  }
}
```
