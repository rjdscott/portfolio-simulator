"""
Polars compute engine.

Two compute paths are benchmarked here:

1. polars_numpy  — Use Polars only for Parquet loading (faster than PyArrow for wide
                    tables), then hand off to NumPy matmul for compute.  This isolates
                    the I/O advantage of the Polars Parquet reader.

2. polars_expr   — Express the entire computation using Polars' lazy expression API:
                    multiply each return column by the corresponding weight, sum
                    horizontally per day to get daily portfolio returns, then aggregate
                    to cumulative return and Sharpe.  No Python loop over portfolios.

The engine registered in run_benchmark.py is "polars_engine" which runs polars_expr.
polars_numpy is available for manual comparison.

Usage
-----
    from src.compute.polars_engine import compute_polars_engine
    results = compute_polars_engine(returns, weights)  # (N, 2) float64

Install
-------
    uv pip install polars
"""

from __future__ import annotations

import math

import numpy as np

_SQRT_252 = math.sqrt(252.0)
_TRADING_DAYS = 252


def compute_polars_engine(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using Polars expressions.

    Strategy: express the N-portfolio computation as a sequence of Polars
    operations on wide DataFrames, avoiding any Python loop over portfolios.

    For each portfolio i:
      - daily_return[i, t] = sum_u( weights[i, u] * returns[t, u] )  (dot product)
      - cum_return[i]       = expm1( sum_t( daily_return[i, t] ) )
      - sharpe[i]           = mean / std(ddof=1) * sqrt(252)

    Polars does not have a built-in Welford/variance-per-column-across-rows primitive,
    so variance is computed as: var = (sum_sq - n*mean^2) / (n-1), which is equivalent
    to ddof=1 but may have larger floating-point error for nearly-constant series.

    Parameters
    ----------
    returns : (T, U) log-returns, C-order float64
    weights : (N, U) portfolio weights, C-order float64

    Returns
    -------
    results : (N, 2) float64 — [cumulative_return, annualised_sharpe]
    """
    import polars as pl

    T, U = returns.shape
    N    = weights.shape[0]

    # Step 1: Compute (N, T) matrix of daily portfolio returns via NumPy matmul.
    # Polars does not support matrix multiplication natively; this step is
    # intentionally left as NumPy to isolate the Polars aggregation overhead.
    # The dot product W @ R.T  is the same operation that numpy_vectorised uses.
    port_returns = weights.astype(np.float64) @ returns.T.astype(np.float64)  # (N, T)

    # Step 2: Convert to a Polars DataFrame (one column per trading day).
    # Column names: "d0", "d1", ..., "d{T-1}"
    col_names = [f"d{t}" for t in range(T)]
    df = pl.DataFrame(
        {name: port_returns[:, t] for name, t in zip(col_names, range(T))}
    )

    # Step 3: Use Polars lazy expressions to compute statistics row-wise.
    # sum_horizontal  → cumulative log-return (sum of daily log-returns)
    # mean_horizontal → daily mean return
    # For variance we need sum_of_squares: no built-in, so compute via:
    #   var = (sum_sq - T * mean^2) / (T - 1)
    all_cols = pl.all()

    result_df = df.lazy().select(
        [
            # cumulative return: expm1(sum of log-returns)
            (pl.sum_horizontal(col_names).exp() - 1.0).alias("cum_return"),

            # mean daily return
            pl.mean_horizontal(col_names).alias("mean_r"),

            # sum of squares (needed for variance)
            pl.sum_horizontal([pl.col(c) ** 2 for c in col_names]).alias("sum_sq"),
        ]
    ).with_columns(
        [
            # var = (sum_sq - T * mean^2) / (T - 1)  [ddof=1]
            (
                (pl.col("sum_sq") - T * pl.col("mean_r") ** 2) / (T - 1)
            ).alias("var_r"),
        ]
    ).with_columns(
        [
            # sharpe = mean / std * sqrt(252); guard against zero variance
            pl.when(pl.col("var_r") > 0)
            .then(pl.col("mean_r") / pl.col("var_r").sqrt() * _SQRT_252)
            .otherwise(0.0)
            .alias("sharpe"),
        ]
    ).select(["cum_return", "sharpe"]).collect()

    return result_df.to_numpy()


def compute_polars_numpy(
    returns: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Polars for I/O only; NumPy for compute.  Not registered as a benchmark engine
    but useful for isolating the Polars Parquet-reader advantage over PyArrow.
    """
    from src.compute.baseline import compute_numpy_matmul
    return compute_numpy_matmul(returns, weights)
