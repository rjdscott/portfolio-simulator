"""
Ray parallel compute engine — Phase 4 distributed baseline.

Uses Ray's local multi-process parallelism. The price returns matrix is
placed in Ray's shared-memory object store (ray.put) so all workers can
access it without repeated serialisation. Portfolio weights are generated
inside each remote task using the deterministic seeded generator.

ray.init(ignore_reinit_error=True) is idempotent — safe to call across reps.
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


def compute_ray_local(
    returns: np.ndarray,
    tickers: list,
    universe: list,
    n: int,
    k: int,
    seed: int,
    batch_size: int = 100_000,
) -> np.ndarray | dict:
    """
    Compute cumulative return and Sharpe for N portfolios using Ray (local).

    For N <= 10M, collects all results and returns an (N, 2) float64 array.
    For N > 10M, streams results through a Welford accumulator using a Ray
    actor to avoid materialising the full array.

    Parameters
    ----------
    returns   : (T, U) log-returns matrix, already loaded from price data
    tickers   : ticker symbols in price-data column order
    universe  : full universe ticker list (generator uses this order)
    n         : total number of portfolios to process
    k         : stocks per portfolio
    seed      : global seed for portfolio generation
    batch_size: portfolios per Ray remote task

    Returns
    -------
    (N, 2) float64 array  [if N <= 10M]
    dict with aggregate stats [if N > 10M]
    """
    import ray

    ray.init(ignore_reinit_error=True)
    returns_ref = ray.put(returns)
    universe_ref = ray.put(universe)

    @ray.remote
    def ray_batch(start, end, ret_ref, tick, univ_ref, k, seed):
        import numpy as _np
        from src.portfolio.generator import generate_batch as _gen
        ret = ray.get(ret_ref)
        univ = ray.get(univ_ref)
        _, weights = _gen(start, end, univ, k=k, global_seed=seed)
        ticker_idx = [univ.index(t) for t in tick]
        w = weights[:, ticker_idx].astype(_np.float64)
        pr = w @ ret.T
        cum = _np.expm1(pr.sum(axis=1))
        mean_r = pr.mean(axis=1)
        std_r = pr.std(axis=1, ddof=1)
        shr = _np.where(std_r > 0, mean_r / std_r * _np.sqrt(252), 0.0)
        return _np.column_stack([cum, shr])

    ranges = [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

    if n <= 10_000_000:
        futures = [
            ray_batch.remote(s, e, returns_ref, tickers, universe_ref, k, seed)
            for s, e in ranges
        ]
        batch_results = ray.get(futures)
        return np.vstack(batch_results)
    else:
        # Stream results through Welford accumulator; fetch futures in windows
        # to limit peak object-store usage
        window = 20  # concurrent futures in flight
        accum_n = 0
        mean_ret = 0.0
        m2_ret = 0.0
        mean_shr = 0.0
        m2_shr = 0.0

        pending = []
        for s, e in ranges:
            pending.append(
                ray_batch.remote(s, e, returns_ref, tickers, universe_ref, k, seed)
            )
            if len(pending) >= window:
                ready, pending = ray.wait(pending, num_returns=1)
                batch = ray.get(ready[0])
                for cum_r, shr in batch:
                    accum_n += 1
                    d1 = cum_r - mean_ret
                    mean_ret += d1 / accum_n
                    m2_ret += d1 * (cum_r - mean_ret)
                    d2 = shr - mean_shr
                    mean_shr += d2 / accum_n
                    m2_shr += d2 * (shr - mean_shr)

        for fut in pending:
            batch = ray.get(fut)
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
