"""
Dask threaded compute engine — Phase 4 distributed baseline.

Uses Dask's threaded scheduler (backed by concurrent.futures.ThreadPoolExecutor).
NumPy releases the GIL during matmul, so threads are effective here.
No separate Dask client/cluster is needed — the threaded scheduler runs
in-process and avoids serialisation overhead.

Portfolio weights are generated inside each delayed task using the
deterministic seeded generator — no CSV loading required.
"""

from __future__ import annotations

import numpy as np


def _compute_batch(
    start: int,
    end: int,
    returns: np.ndarray,
    tickers: list,
    universe: list,
    k: int,
    seed: int,
) -> np.ndarray:
    """Generate portfolio weights and compute metrics for one batch."""
    from src.portfolio.generator import generate_batch

    _, weights = generate_batch(start, end, universe, k=k, global_seed=seed)
    # Align universe order to tickers (no-op if they match exactly)
    ticker_idx = [universe.index(t) for t in tickers]
    weights_aligned = weights[:, ticker_idx].astype(np.float64)

    port_returns = weights_aligned @ returns.T  # (batch_size, T)
    cum_ret = np.expm1(port_returns.sum(axis=1))
    mean_r = port_returns.mean(axis=1)
    std_r = port_returns.std(axis=1, ddof=1)
    sharpe = np.where(std_r > 0, mean_r / std_r * np.sqrt(252), 0.0)
    return np.column_stack([cum_ret, sharpe])


def compute_dask_local(
    returns: np.ndarray,
    tickers: list,
    universe: list,
    n: int,
    k: int,
    seed: int,
    batch_size: int = 100_000,
) -> np.ndarray | dict:
    """
    Compute cumulative return and Sharpe for N portfolios using Dask (threaded).

    For N <= 10M, collects all results and returns an (N, 2) float64 array.
    For N > 10M, streams results through a Welford accumulator and returns
    an aggregate stats dict (no full array materialised).

    Parameters
    ----------
    returns   : (T, U) log-returns matrix, already loaded from price data
    tickers   : ticker symbols in price-data column order
    universe  : full universe ticker list (generator uses this order)
    n         : total number of portfolios to process
    k         : stocks per portfolio
    seed      : global seed for portfolio generation
    batch_size: portfolios per Dask delayed task

    Returns
    -------
    (N, 2) float64 array  [if N <= 10M]
    dict with aggregate stats [if N > 10M]
    """
    import dask

    ranges = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    if n <= 10_000_000:
        delayed_batches = [
            dask.delayed(_compute_batch)(start, end, returns, tickers, universe, k, seed)
            for start, end in ranges
        ]
        batch_results = dask.compute(*delayed_batches, scheduler="threads")
        return np.vstack(batch_results)
    else:
        # Stream results through Welford accumulator to avoid OOM
        accum_n = 0
        mean_ret = 0.0
        m2_ret = 0.0
        mean_shr = 0.0
        m2_shr = 0.0

        # Process in super-batches to limit concurrency overhead
        super_batch = 10  # number of delayed tasks to materialise at once
        for super_start in range(0, len(ranges), super_batch):
            chunk = ranges[super_start:super_start + super_batch]
            delayed_batches = [
                dask.delayed(_compute_batch)(s, e, returns, tickers, universe, k, seed)
                for s, e in chunk
            ]
            batch_results = dask.compute(*delayed_batches, scheduler="threads")

            for batch in batch_results:
                for cum_r, shr in batch:
                    accum_n += 1
                    d1 = cum_r - mean_ret
                    mean_ret += d1 / accum_n
                    m2_ret += d1 * (cum_r - mean_ret)
                    d2 = shr - mean_shr
                    mean_shr += d2 / accum_n
                    m2_shr += d2 * (shr - mean_shr)

        return {
            "n_portfolios": accum_n,
            "mean_cumulative_return": mean_ret,
            "std_cumulative_return": np.sqrt(m2_ret / max(accum_n - 1, 1)),
            "mean_sharpe": mean_shr,
            "std_sharpe": np.sqrt(m2_shr / max(accum_n - 1, 1)),
        }
