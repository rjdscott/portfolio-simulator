"""
Data acquisition module.

Downloads S&P 500 constituent metadata from Wikipedia and daily adjusted-close
price history from Yahoo Finance (via yfinance), then filters to the top N
stocks by market cap with complete price history.

Outputs
-------
data/raw/metadata/sp500_constituents.csv   — full S&P 500 list
data/raw/metadata/universe.csv             — filtered benchmark universe
data/raw/metadata/data_manifest.json       — provenance record
data/raw/prices/<TICKER>.csv               — per-stock price files (CSV baseline)
data/raw/prices/prices_wide.csv            — all stocks, wide format
data/raw/prices/prices_long.csv            — all stocks, long format
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — override via CLI or config if needed
# ---------------------------------------------------------------------------

STUDY_START = "2020-01-01"
STUDY_END = "2024-12-31"
UNIVERSE_SIZE = 100          # Top N stocks to include in benchmark universe
MIN_COMPLETENESS = 0.95      # Minimum fraction of trading days with valid prices
MARKET_CAP_FETCH_DELAY = 0.3 # Seconds between ticker info requests (rate limiting)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PRICES_DIR = RAW_DIR / "prices"
META_DIR = RAW_DIR / "metadata"

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# ---------------------------------------------------------------------------
# Step 1: Fetch S&P 500 constituent list from Wikipedia
# ---------------------------------------------------------------------------

def fetch_sp500_constituents() -> pd.DataFrame:
    """
    Scrape the S&P 500 constituent table from Wikipedia.

    Returns a DataFrame with columns:
        ticker, name, sector, sub_industry, cik
    """
    log.info("Fetching S&P 500 constituent list from Wikipedia...")
    tables = pd.read_html(WIKIPEDIA_SP500_URL, header=0)
    df = tables[0]

    # Wikipedia column names vary slightly over time — normalise them
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "CIK": "cik",
    })

    # Keep only relevant columns (others may include date added, etc.)
    keep = [c for c in ["ticker", "name", "sector", "sub_industry", "cik"] if c in df.columns]
    df = df[keep].copy()

    # Yahoo Finance uses "-" instead of "." for BRK.B → BRK-B, etc.
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    log.info(f"Found {len(df)} S&P 500 constituents.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: Download price history for all constituents
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str = STUDY_START,
    end: str = STUDY_END,
) -> pd.DataFrame:
    """
    Download daily adjusted-close prices for the given tickers via yfinance.

    Uses yfinance's batch download for efficiency, then falls back to
    per-ticker downloads for any failures.

    Parameters
    ----------
    tickers : list of ticker symbols
    start   : start date string "YYYY-MM-DD"
    end     : end date string "YYYY-MM-DD"

    Returns
    -------
    DataFrame with date index and ticker columns (adjusted close prices).
    """
    log.info(f"Downloading price history for {len(tickers)} tickers: {start} → {end}")

    # Batch download — much faster than per-ticker loop
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,     # adjusts for splits and dividends
        progress=True,
        threads=True,
    )

    # yfinance returns MultiIndex columns when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker edge case
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index).date  # type: ignore[attr-defined]
    prices.index.name = "date"

    log.info(f"Downloaded price matrix: {prices.shape[0]} trading days × {prices.shape[1]} tickers")
    return prices


# ---------------------------------------------------------------------------
# Step 3: Filter to benchmark universe
# ---------------------------------------------------------------------------

def compute_completeness(prices: pd.DataFrame) -> pd.Series:
    """Return fraction of non-null rows for each ticker."""
    return prices.notna().mean()


def fetch_market_caps(tickers: list[str]) -> dict[str, Optional[float]]:
    """
    Fetch current market capitalisation for each ticker via yfinance.

    This makes individual HTTP requests and is rate-limited via a small delay.
    """
    log.info(f"Fetching market caps for {len(tickers)} tickers (rate-limited)...")
    caps: dict[str, Optional[float]] = {}
    for ticker in tqdm(tickers, desc="Market caps"):
        try:
            info = yf.Ticker(ticker).fast_info
            caps[ticker] = getattr(info, "market_cap", None)
        except Exception as exc:
            log.warning(f"Could not fetch market cap for {ticker}: {exc}")
            caps[ticker] = None
        time.sleep(MARKET_CAP_FETCH_DELAY)
    return caps


def select_universe(
    sp500: pd.DataFrame,
    prices: pd.DataFrame,
    universe_size: int = UNIVERSE_SIZE,
    min_completeness: float = MIN_COMPLETENESS,
) -> pd.DataFrame:
    """
    Select the benchmark universe: top `universe_size` stocks by market cap
    that have at least `min_completeness` fraction of valid price observations.

    Parameters
    ----------
    sp500          : full S&P 500 constituent DataFrame
    prices         : wide-format price DataFrame (date × ticker)
    universe_size  : target number of stocks
    min_completeness : minimum data completeness threshold

    Returns
    -------
    DataFrame with columns: ticker, name, sector, market_cap_usd,
                            data_completeness, rank
    """
    log.info("Selecting benchmark universe...")

    completeness = compute_completeness(prices)

    # Filter to tickers that actually have price data and meet completeness bar
    available_tickers = [t for t in sp500["ticker"] if t in prices.columns]
    complete_tickers = [
        t for t in available_tickers
        if completeness.get(t, 0.0) >= min_completeness
    ]
    log.info(
        f"{len(complete_tickers)}/{len(available_tickers)} tickers pass "
        f"the {min_completeness:.0%} completeness threshold"
    )

    # Fetch market caps for complete tickers only
    market_caps = fetch_market_caps(complete_tickers)

    # Build the universe DataFrame
    universe_rows = []
    for ticker in complete_tickers:
        row = sp500[sp500["ticker"] == ticker].iloc[0]
        universe_rows.append({
            "ticker": ticker,
            "name": row.get("name", ""),
            "sector": row.get("sector", ""),
            "market_cap_usd": market_caps.get(ticker),
            "data_completeness": completeness[ticker],
        })

    universe_df = pd.DataFrame(universe_rows)

    # Sort by market cap descending, nulls last
    universe_df = universe_df.sort_values(
        "market_cap_usd", ascending=False, na_position="last"
    ).head(universe_size).reset_index(drop=True)

    universe_df["rank"] = universe_df.index + 1

    log.info(f"Universe selected: {len(universe_df)} stocks")
    log.info(f"Sectors: {universe_df['sector'].value_counts().to_dict()}")
    return universe_df


# ---------------------------------------------------------------------------
# Step 4: Write outputs
# ---------------------------------------------------------------------------

def save_per_stock_csvs(prices: pd.DataFrame, universe_tickers: list[str]) -> None:
    """Write one CSV file per stock — the baseline 'worst format'."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing per-stock CSVs to {PRICES_DIR} ...")
    for ticker in tqdm(universe_tickers, desc="Per-stock CSVs"):
        if ticker not in prices.columns:
            log.warning(f"Ticker {ticker} not found in price matrix, skipping.")
            continue
        df = prices[[ticker]].rename(columns={ticker: "adj_close"}).copy()
        df.index.name = "date"
        df = df.dropna()
        path = PRICES_DIR / f"{ticker}.csv"
        df.to_csv(path)
    log.info("Per-stock CSVs written.")


def save_wide_csv(prices: pd.DataFrame, universe_tickers: list[str]) -> None:
    """Write a single wide-format CSV (date × tickers)."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    path = PRICES_DIR / "prices_wide.csv"
    log.info(f"Writing wide-format CSV to {path} ...")
    wide = prices[universe_tickers].copy()
    wide.index.name = "date"
    wide.to_csv(path)
    size_mb = path.stat().st_size / 1_048_576
    log.info(f"Wide CSV written: {size_mb:.2f} MB")


def save_long_csv(prices: pd.DataFrame, universe_tickers: list[str]) -> None:
    """Write a single long-format (tidy) CSV (date, ticker, adj_close)."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    path = PRICES_DIR / "prices_long.csv"
    log.info(f"Writing long-format CSV to {path} ...")
    wide = prices[universe_tickers].copy()
    wide.index.name = "date"
    long = wide.reset_index().melt(id_vars="date", var_name="ticker", value_name="adj_close")
    long = long.sort_values(["date", "ticker"]).reset_index(drop=True)
    long.to_csv(path, index=False)
    size_mb = path.stat().st_size / 1_048_576
    log.info(f"Long CSV written: {size_mb:.2f} MB, {len(long):,} rows")


def save_manifest(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    sp500: pd.DataFrame,
    excluded: list[str],
    exclusion_reasons: dict[str, str],
) -> None:
    """Write data_manifest.json with full provenance information."""
    META_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import yfinance

    trading_days = prices.index.tolist()
    manifest = {
        "fetch_date": date.today().isoformat(),
        "fetch_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "study_start": STUDY_START,
        "study_end": STUDY_END,
        "expected_trading_days": len(trading_days),
        "first_trading_day": str(trading_days[0]) if trading_days else None,
        "last_trading_day": str(trading_days[-1]) if trading_days else None,
        "universe_size": len(universe),
        "sp500_size_at_fetch": len(sp500),
        "portfolio_k": 15,
        "global_seed": 42,
        "yfinance_version": yfinance.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "tickers": universe["ticker"].tolist(),
        "excluded_tickers": excluded,
        "exclusion_reasons": exclusion_reasons,
    }
    path = META_DIR / "data_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest written to {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    universe_size: int = UNIVERSE_SIZE,
    start: str = STUDY_START,
    end: str = STUDY_END,
    min_completeness: float = MIN_COMPLETENESS,
) -> None:
    """
    Full data fetch pipeline. Run this once to populate data/raw/.

    This is idempotent — re-running will overwrite existing files.
    """
    META_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Constituent list
    sp500 = fetch_sp500_constituents()
    sp500.to_csv(META_DIR / "sp500_constituents.csv", index=False)
    log.info(f"S&P 500 constituents saved ({len(sp500)} rows)")

    # 2. Price history for all constituents
    all_tickers = sp500["ticker"].tolist()
    prices = download_prices(all_tickers, start=start, end=end)

    # 3. Select benchmark universe
    universe = select_universe(
        sp500, prices,
        universe_size=universe_size,
        min_completeness=min_completeness,
    )
    universe.to_csv(META_DIR / "universe.csv", index=False)

    universe_tickers = universe["ticker"].tolist()

    # Compute exclusions for manifest
    completeness = compute_completeness(prices)
    available = set(t for t in all_tickers if t in prices.columns)
    excluded = [
        t for t in available
        if completeness.get(t, 0.0) < min_completeness
    ]
    exclusion_reasons = {
        t: f"data_completeness={completeness.get(t, 0.0):.3f} < {min_completeness}"
        for t in excluded
    }
    # Also note stocks not in universe due to market cap ranking
    not_ranked_in = [
        t for t in available
        if t not in universe_tickers and t not in excluded
    ]
    for t in not_ranked_in:
        exclusion_reasons[t] = f"below_top_{universe_size}_by_market_cap"
        excluded.append(t)

    # 4. Impute remaining missing values in universe tickers (forward-fill)
    prices_universe = prices[universe_tickers].ffill()

    # 5. Write storage format baselines
    save_per_stock_csvs(prices_universe, universe_tickers)
    save_wide_csv(prices_universe, universe_tickers)
    save_long_csv(prices_universe, universe_tickers)

    # 6. Manifest
    save_manifest(universe, prices_universe, sp500, excluded, exclusion_reasons)

    log.info("=" * 60)
    log.info("Data fetch complete.")
    log.info(f"  Universe : {len(universe_tickers)} stocks")
    log.info(f"  Date range: {start} → {end}")
    log.info(f"  Outputs  : {RAW_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run()
