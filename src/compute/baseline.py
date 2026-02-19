"""
Baseline compute implementation: Pandas row loop + NumPy vectorised matmul.

Two variants are provided:

1. pandas_row_loop  — single-threaded Pandas iteration (worst-case compute)
2. numpy_matmul     — vectorised matmul using NumPy (BLAS-backed)

Both take the same inputs and produce identical results (within floating-point
tolerance), making them directly comparable. Results are validated against each
other via checksum.

Metrics computed per portfolio
------------------------------
- Cumulative total return: exp(sum(log_returns @ weights)) - 1
- Annualised Sharpe ratio: mean(portfolio_log_returns) / std(...) * sqrt(252)
  (risk-free rate = 0)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PRICES_DIR = ROOT / "data" / "raw" / "prices"
PORTFOLIOS_DIR = ROOT / "data" / "raw" / "portfolios"

TRADING_DAYS_PER_YEAR = 252.0


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_returns_from_wide_csv(path: Path | None = None) -> np.ndarray:
    """
    Load the wide-format price CSV and return the log-returns matrix.

    Returns
    -------
    returns : float64 ndarray of shape (T, U)
              T = trading days, U = universe size
              Row 0 is the first day with valid returns (day 1 of the price series).
    tickers : list of ticker symbols (length U)
    """
    if path is None:
        path = PRICES_DIR / "prices_wide.csv"
    log.info(f"Loading prices from {path}...")
    prices_df = pd.read_csv(path, index_col="date")
    tickers = prices_df.columns.tolist()
    prices = prices_df.to_numpy(dtype=np.float64)

    # Log returns: ln(P_t / P_{t-1}), drop the first NaN row
    log_returns = np.log(prices[1:] / prices[:-1])

    log.info(f"Returns matrix: {log_returns.shape[0]} days × {log_returns.shape[1]} stocks")
    return log_returns, tickers


def load_returns_from_per_stock_csvs(prices_dir: Path | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Load price data by reading one CSV per stock — the worst-case I/O pattern.

    Forces 100 individual file opens, seeks, and parses. This is the baseline
    storage benchmark.

    Returns
    -------
    returns : float64 ndarray of shape (T, U)
    tickers : list of ticker symbols (length U)
    """
    if prices_dir is None:
        prices_dir = PRICES_DIR

    csv_files = sorted(prices_dir.glob("*.csv"))
    # Exclude the combined files
    csv_files = [f for f in csv_files if f.stem not in ("prices_wide", "prices_long")]

    if not csv_files:
        raise FileNotFoundError(f"No per-stock CSVs found in {prices_dir}")

    log.info(f"Loading {len(csv_files)} per-stock CSVs from {prices_dir}...")

    series_list = []
    tickers = []
    for f in csv_files:
        df = pd.read_csv(f, index_col="date", parse_dates=False)
        series_list.append(df["adj_close"].values)
        tickers.append(f.stem)

    prices = np.column_stack(series_list).astype(np.float64)
    log_returns = np.log(prices[1:] / prices[:-1])
    log.info(f"Returns matrix built: {log_returns.shape}")
    return log_returns, tickers


# ---------------------------------------------------------------------------
# Compute variant 1: Pandas row loop (slowest — for tiny N only)
# ---------------------------------------------------------------------------

def compute_pandas_row_loop(
    returns: np.ndarray,          # shape (T, U)
    weights: np.ndarray,          # shape (N, U) — dense, zeros for non-selected
    tickers: list[str],
) -> np.ndarray:
    """
    Compute cumulative return and Sharpe ratio using a Python-level row loop.

    This is intentionally the worst-case compute implementation. Use only for
    N ≤ 10,000 (will be very slow at 1M).

    Returns
    -------
    results : float64 ndarray of shape (N, 2)
              columns: [cumulative_return, annualised_sharpe]
    """
    T, U = returns.shape
    N = weights.shape[0]
    results = np.empty((N, 2), dtype=np.float64)

    returns_df = pd.DataFrame(returns, columns=tickers)

    for i in range(N):
        w = weights[i]  # shape (U,)
        # Portfolio log returns as weighted sum across stocks
        port_returns = returns_df.values @ w  # shape (T,)
        cum_return = np.expm1(port_returns.sum())
        mean_r = port_returns.mean()
        std_r = port_returns.std(ddof=1)
        sharpe = (mean_r / std_r) * np.sqrt(TRADING_DAYS_PER_YEAR) if std_r > 0 else 0.0
        results[i, 0] = cum_return
        results[i, 1] = sharpe

    return results


# ---------------------------------------------------------------------------
# Compute variant 2: NumPy vectorised matmul (fast)
# ---------------------------------------------------------------------------

def compute_numpy_matmul(
    returns: np.ndarray,   # shape (T, U)
    weights: np.ndarray,   # shape (N, U)
) -> np.ndarray:
    """
    Compute cumulative return and Sharpe ratio via vectorised matrix multiply.

    Core operation: W @ R^T → portfolio_returns matrix of shape (N, T)

    This uses BLAS DGEMM under the hood (via NumPy). On a system with MKL or
    OpenBLAS, this will automatically use multiple threads.

    Parameters
    ----------
    returns : (T, U) float64 array — log returns matrix
    weights : (N, U) float32 or float64 array — portfolio weight matrix

    Returns
    -------
    results : (N, 2) float64 array
              columns: [cumulative_return, annualised_sharpe]
    """
    # Ensure float64 for numerical precision
    W = weights.astype(np.float64)  # (N, U)
    R = returns.astype(np.float64)  # (T, U)

    # Portfolio log returns: (N, U) @ (U, T) → (N, T)
    port_returns = W @ R.T  # shape (N, T)

    # Cumulative return: exp(sum over T) - 1
    cum_returns = np.expm1(port_returns.sum(axis=1))  # (N,)

    # Annualised Sharpe (Rf = 0)
    mean_r = port_returns.mean(axis=1)     # (N,)
    std_r = port_returns.std(axis=1, ddof=1)  # (N,)
    sharpe = np.where(std_r > 0, mean_r / std_r * np.sqrt(TRADING_DAYS_PER_YEAR), 0.0)

    results = np.column_stack([cum_returns, sharpe])  # (N, 2)
    return results


# ---------------------------------------------------------------------------
# Result checksum (cross-implementation validation)
# ---------------------------------------------------------------------------

def result_checksum(results: np.ndarray) -> str:
    """
    SHA-256 of the sorted results array.

    Sorting before hashing allows comparison across implementations that may
    process portfolios in different orders.
    """
    sorted_results = results[np.lexsort(results.T[::-1])]
    return hashlib.sha256(sorted_results.astype(np.float64).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Convenience: load + compute in one call
# ---------------------------------------------------------------------------

def run(
    portfolio_csv: Path,
    storage: Literal["wide_csv", "per_stock_csv"] = "wide_csv",
    engine: Literal["pandas_loop", "numpy_matmul"] = "numpy_matmul",
) -> tuple[np.ndarray, dict]:
    """
    Load data and compute metrics for all portfolios in a weight CSV.

    Parameters
    ----------
    portfolio_csv : path to portfolio weight CSV (from generator.py)
    storage       : price data storage format to read from
    engine        : compute variant to use

    Returns
    -------
    results : (N, 2) float64 array [cumulative_return, annualised_sharpe]
    meta    : dict with shape information and checksum
    """
    # Load prices
    if storage == "wide_csv":
        returns, tickers = load_returns_from_wide_csv()
    elif storage == "per_stock_csv":
        returns, tickers = load_returns_from_per_stock_csvs()
    else:
        raise ValueError(f"Unknown storage format: {storage}")

    # Load portfolios
    log.info(f"Loading portfolio weights from {portfolio_csv}...")
    df = pd.read_csv(portfolio_csv, index_col="portfolio_id")
    # Reorder columns to match universe ticker order in returns matrix
    df = df.reindex(columns=tickers, fill_value=0.0)
    weights = df.to_numpy(dtype=np.float32)

    N = weights.shape[0]
    log.info(f"Portfolio matrix loaded: {N:,} portfolios × {weights.shape[1]} stocks")

    # Compute
    if engine == "numpy_matmul":
        results = compute_numpy_matmul(returns, weights)
    elif engine == "pandas_loop":
        results = compute_pandas_row_loop(returns, weights, tickers)
    else:
        raise ValueError(f"Unknown engine: {engine}")

    meta = {
        "n_portfolios": N,
        "n_trading_days": returns.shape[0],
        "n_stocks": returns.shape[1],
        "engine": engine,
        "storage": storage,
        "result_checksum": result_checksum(results),
    }

    log.info(f"Computed results for {N:,} portfolios. Checksum: {meta['result_checksum'][:16]}...")
    return results, meta
