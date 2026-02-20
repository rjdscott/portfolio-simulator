"""
PySpark local compute engine — Phase 4 distributed baseline.

Distributes N portfolio computations across Spark tasks running locally
(SPARK_MASTER = "local[*]"). Portfolio weights are generated inside each
task using the deterministic seeded generator — no CSV loading required.

The SparkSession is created once at module level and reused across reps
(singleton pattern via _get_or_create_spark). Price return data is
broadcast to all workers once per benchmark run.
"""

from __future__ import annotations

import numpy as np

_SPARK_SESSION = None
SPARK_MASTER = "local[*]"


def _get_or_create_spark():
    global _SPARK_SESSION
    if _SPARK_SESSION is None:
        from pyspark.sql import SparkSession
        _SPARK_SESSION = (
            SparkSession.builder
            .master(SPARK_MASTER)
            .appName("portfolio-simulator")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.memory", "4g")
            .getOrCreate()
        )
        _SPARK_SESSION.sparkContext.setLogLevel("WARN")
    return _SPARK_SESSION


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


def compute_spark_local(
    returns: np.ndarray,
    tickers: list,
    universe: list,
    n: int,
    k: int,
    seed: int,
    batch_size: int = 100_000,
) -> np.ndarray | dict:
    """
    Compute cumulative return and Sharpe for N portfolios using PySpark local mode.

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
    batch_size: portfolios per Spark task

    Returns
    -------
    (N, 2) float64 array  [if N <= 10M]
    dict with aggregate stats [if N > 10M]
    """
    sc = _get_or_create_spark().sparkContext
    bc_returns = sc.broadcast(returns)
    bc_universe = sc.broadcast(universe)
    bc_tickers = sc.broadcast(tickers)

    ranges = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    def map_fn(batch_range):
        import numpy as _np
        from src.portfolio.generator import generate_batch as _gen
        start, end = batch_range
        ret = bc_returns.value
        univ = bc_universe.value
        tick = bc_tickers.value
        _, weights = _gen(start, end, univ, k=k, global_seed=seed)
        ticker_idx = [univ.index(t) for t in tick]
        w = weights[:, ticker_idx].astype(_np.float64)
        pr = w @ ret.T
        cum = _np.expm1(pr.sum(axis=1))
        mean_r = pr.mean(axis=1)
        std_r = pr.std(axis=1, ddof=1)
        shr = _np.where(std_r > 0, mean_r / std_r * _np.sqrt(252), 0.0)
        return _np.column_stack([cum, shr]).tolist()

    rdd = sc.parallelize(ranges, numSlices=len(ranges))

    if n <= 10_000_000:
        results_nested = rdd.map(map_fn).collect()
        return np.vstack([np.array(b) for b in results_nested])
    else:
        # Welford streaming reduce — never materialises the full array
        def seq_op(accum, batch_list):
            batch = np.array(batch_list)
            for row in batch:
                accum["n"] += 1
                for j, (mk, m2k) in enumerate(
                    [("mean_ret", "m2_ret"), ("mean_shr", "m2_shr")]
                ):
                    delta = row[j] - accum[mk]
                    accum[mk] += delta / accum["n"]
                    accum[m2k] += delta * (row[j] - accum[mk])
            return accum

        def comb_op(a, b):
            # Merge two Welford accumulators (parallel merge formula)
            na, nb = a["n"], b["n"]
            if na + nb == 0:
                return a
            combined = {"n": na + nb, "mean_ret": 0.0, "m2_ret": 0.0,
                        "mean_shr": 0.0, "m2_shr": 0.0}
            for mk, m2k in [("mean_ret", "m2_ret"), ("mean_shr", "m2_shr")]:
                delta = b[mk] - a[mk]
                combined[mk] = (a[mk] * na + b[mk] * nb) / (na + nb)
                combined[m2k] = a[m2k] + b[m2k] + delta ** 2 * na * nb / (na + nb)
            return combined

        zero = {"n": 0, "mean_ret": 0.0, "m2_ret": 0.0, "mean_shr": 0.0, "m2_shr": 0.0}
        accum = rdd.map(map_fn).aggregate(zero, seq_op, comb_op)

        n_total = accum["n"]
        return {
            "n_portfolios": n_total,
            "mean_cumulative_return": accum["mean_ret"],
            "std_cumulative_return": np.sqrt(accum["m2_ret"] / max(n_total - 1, 1)),
            "mean_sharpe": accum["mean_shr"],
            "std_sharpe": np.sqrt(accum["m2_shr"] / max(n_total - 1, 1)),
        }
