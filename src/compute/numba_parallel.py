"""
Numba JIT compute engine — parallel prange over N portfolios.

Metrics computed (identical to baseline.py):
  - Cumulative total return: expm1(sum of log_returns @ weights)
  - Annualised Sharpe ratio: mean(port_returns) / std(port_returns) * sqrt(252)
    using Welford online variance (ddof=1, numerically stable)

The @njit(parallel=True) decorator causes Numba to launch one thread per
physical core (governed by NUMBA_NUM_THREADS). fastmath=True enables SIMD
fused-multiply-add; cache=True writes compiled bytecode to __pycache__/ so
the second run skips JIT compilation.
"""

from __future__ import annotations

import numpy as np

_TRADING_DAYS_PER_YEAR = 252.0


def compute_numba_parallel(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio for N portfolios using Numba.

    The Numba kernel uses prange (parallel range) over N so each thread
    owns a disjoint slice of portfolios — no synchronisation required.

    Parameters
    ----------
    returns : (T, U) log-returns matrix (float64)
    weights : (N, U) portfolio weight matrix (will be cast to float64)

    Returns
    -------
    results : (N, 2) array — columns [cumulative_return, annualised_sharpe]
    """
    from numba import njit, prange  # deferred import — only pay cost if engine used

    # JIT-compile on first call (subsequent calls hit cache)
    _kernel = _get_kernel(njit, prange)

    W = np.ascontiguousarray(weights, dtype=np.float64)
    R = np.ascontiguousarray(returns, dtype=np.float64)
    results = np.empty((W.shape[0], 2), dtype=np.float64)
    _kernel(R, W, results, _TRADING_DAYS_PER_YEAR)
    return results


# ---------------------------------------------------------------------------
# Kernel factory — defined once and cached by Python (not re-traced each call)
# ---------------------------------------------------------------------------

_COMPILED_KERNEL = None


def _get_kernel(njit, prange):
    global _COMPILED_KERNEL
    if _COMPILED_KERNEL is not None:
        return _COMPILED_KERNEL

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_kernel(R, W, results, annual_factor):
        """
        R : (T, U) float64 — log returns
        W : (N, U) float64 — portfolio weights
        results : (N, 2) float64 — output [cum_return, sharpe]
        annual_factor : float — sqrt(252)
        """
        N = W.shape[0]
        T = R.shape[0]
        U = R.shape[1]
        sqrt_annual = annual_factor ** 0.5

        for i in prange(N):
            # Compute portfolio log-return time series (dot product per day)
            # Welford online mean and M2 for variance (single pass, ddof=1)
            count = 0
            mean = 0.0
            M2 = 0.0
            log_sum = 0.0

            for t in range(T):
                # Weighted sum across stocks for day t
                port_r = 0.0
                for u in range(U):
                    port_r += W[i, u] * R[t, u]

                log_sum += port_r

                # Welford update
                count += 1
                delta = port_r - mean
                mean += delta / count
                delta2 = port_r - mean
                M2 += delta * delta2

            # Cumulative return
            # expm1(x) = e^x - 1, numerically stable for small x
            # Use Taylor-series-safe computation
            cum_return = (2.718281828459045 ** log_sum) - 1.0

            # Annualised Sharpe (ddof=1)
            if count > 1 and M2 > 0.0:
                variance = M2 / (count - 1)
                std_r = variance ** 0.5
                sharpe = (mean / std_r) * sqrt_annual
            else:
                sharpe = 0.0

            results[i, 0] = cum_return
            results[i, 1] = sharpe

    _COMPILED_KERNEL = _numba_kernel
    return _COMPILED_KERNEL
