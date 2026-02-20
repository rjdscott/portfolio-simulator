"""
NumPy float32 compute engine.

Casts both the returns matrix AND portfolio weights to float32 before the
matmul.  This halves the L3 cache footprint (100×1,257 × 4 B = ~0.5 MB vs
~1 MB for float64), which is the key question: does the memory-bandwidth
saving offset the precision loss and the extra cast overhead?

The statistics (mean, std, Sharpe) are computed in float64 after upcasting
the daily-return matrix back from float32.

Note: result_checksum will differ from the float64 engines — expected and
documented.  The difference is small (< 1e-5 in cumulative return) but
consistent, so same-engine reruns still reproduce identically.

Usage
-----
    from src.compute.numpy_float32 import compute_numpy_float32
    results = compute_numpy_float32(returns, weights)  # (N, 2) float64

Install
-------
    No extra dependencies — NumPy is already required.
"""

from __future__ import annotations

import numpy as np


def compute_numpy_float32(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using float32 matmul.

    Parameters
    ----------
    returns : (T, U) log-returns matrix
    weights : (N, U) portfolio weight matrix

    Returns
    -------
    results : (N, 2) float64 array — [cumulative_return, annualised_sharpe]
    """
    R = np.ascontiguousarray(returns, dtype=np.float32)
    W = np.ascontiguousarray(weights, dtype=np.float32)

    # Matmul in float32; upcast immediately to float64 for statistics
    port_returns = (W @ R.T).astype(np.float64)  # (N, T)

    cum_ret = np.expm1(port_returns.sum(axis=1))
    mean_r  = port_returns.mean(axis=1)
    std_r   = port_returns.std(axis=1, ddof=1)
    sharpe  = np.where(std_r > 0, mean_r / std_r * np.sqrt(252.0), 0.0)

    return np.column_stack([cum_ret, sharpe])
