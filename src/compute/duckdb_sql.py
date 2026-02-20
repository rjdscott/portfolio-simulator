"""
DuckDB SQL compute engine.

Expresses the portfolio-return computation entirely as SQL using DuckDB's
in-process analytical engine.  The numpy arrays are registered as virtual
relations; DuckDB's parallel hash-aggregation executes the GROUP BY.

Computation expressed in SQL:

    WITH portfolio_daily AS (
        SELECT
            w.portfolio_id,
            r.day_idx,
            SUM(w.weight * r.log_return) AS daily_log_return
        FROM weights_rel w
        JOIN returns_rel r ON w.ticker_idx = r.ticker_idx
        GROUP BY w.portfolio_id, r.day_idx
    ),
    aggregated AS (
        SELECT
            portfolio_id,
            EXP(SUM(daily_log_return))  - 1.0    AS cum_return,
            AVG(daily_log_return)                 AS mean_r,
            STDDEV_SAMP(daily_log_return)         AS std_r
        FROM portfolio_daily
        GROUP BY portfolio_id
    )
    SELECT
        portfolio_id,
        cum_return,
        CASE WHEN std_r > 0 THEN mean_r / std_r * 15.874507866 ELSE 0 END AS sharpe
    FROM aggregated
    ORDER BY portfolio_id

Where 15.874507866 = sqrt(252).

The (N×U) weight matrix and (T×U) returns matrix are "melted" into long-form
relations before querying.  This reshape step is included in the timed window
because it is part of the DuckDB workload — a pure-SQL pipeline would require it.

Usage
-----
    from src.compute.duckdb_sql import compute_duckdb_sql
    results = compute_duckdb_sql(returns, weights)  # (N, 2) float64

Install
-------
    uv pip install duckdb
"""

from __future__ import annotations

import math

import numpy as np

_SQRT_252 = math.sqrt(252.0)


def compute_duckdb_sql(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using DuckDB SQL.

    Parameters
    ----------
    returns : (T, U) log-returns, C-order float64
    weights : (N, U) portfolio weights, C-order float64

    Returns
    -------
    results : (N, 2) float64 — [cumulative_return, annualised_sharpe]
    """
    import duckdb

    import pandas as pd

    T, U = returns.shape
    N    = weights.shape[0]

    # Melt returns into long format with explicit column names.
    # np.repeat / np.tile avoids the column-type ambiguity of np.column_stack.
    returns_df = pd.DataFrame({
        "day_idx":    np.repeat(np.arange(T, dtype=np.int32), U),
        "ticker_idx": np.tile(np.arange(U, dtype=np.int32), T),
        "log_return": returns.ravel(),
    })

    # Melt weights into long format.
    weights_df = pd.DataFrame({
        "portfolio_id": np.repeat(np.arange(N, dtype=np.int32), U),
        "ticker_idx":   np.tile(np.arange(U, dtype=np.int32), N),
        "weight":       weights.ravel(),
    })

    con = duckdb.connect()
    # DuckDB treats registered DataFrames as queryable relations with named columns.
    con.register("returns_tbl", returns_df)
    con.register("weights_tbl", weights_df)

    result = con.execute(f"""
        WITH portfolio_daily AS (
            SELECT
                w.portfolio_id,
                r.day_idx,
                SUM(w.weight * r.log_return) AS daily_log_return
            FROM weights_tbl w
            JOIN returns_tbl r ON w.ticker_idx = r.ticker_idx
            GROUP BY w.portfolio_id, r.day_idx
        ),
        aggregated AS (
            SELECT
                portfolio_id,
                EXP(SUM(daily_log_return)) - 1.0  AS cum_return,
                AVG(daily_log_return)              AS mean_r,
                STDDEV_SAMP(daily_log_return)      AS std_r
            FROM portfolio_daily
            GROUP BY portfolio_id
        )
        SELECT
            portfolio_id,
            cum_return,
            CASE WHEN std_r > 0
                 THEN mean_r / std_r * {_SQRT_252}
                 ELSE 0.0
            END AS sharpe
        FROM aggregated
        ORDER BY portfolio_id
    """).fetchnumpy()

    con.close()

    cum_return = result["cum_return"].astype(np.float64)
    sharpe     = result["sharpe"].astype(np.float64)

    return np.column_stack([cum_return, sharpe])
