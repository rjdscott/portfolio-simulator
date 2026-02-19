"""
CuPy GPU compute engine — same W @ R.T matmul pattern as numpy_matmul but on GPU.

GPU→CPU transfer time is included in compute_metrics phase intentionally:
it represents the full round-trip cost, which is what matters in practice.

Usage
-----
from src.compute.cupy_gpu import compute_cupy_gpu, is_available

if is_available():
    results = compute_cupy_gpu(returns, weights)
else:
    print("CuPy not available — no CUDA GPU detected or cupy not installed")

Notes
-----
- Allocating GPU memory and running a small kernel has ~1–5 ms fixed overhead,
  so CuPy wins only at large N (≥100K) where compute dominates.
- PCIe bandwidth is ~16 GB/s (Gen4 x16); for the 1M portfolio problem the
  weight matrix is ~800 MB, which takes ~50 ms to transfer — warmup hides this.
"""

from __future__ import annotations

import numpy as np

_TRADING_DAYS_PER_YEAR = 252.0


def is_available() -> bool:
    """Return True if CuPy is installed and at least one CUDA GPU is accessible."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()  # raises if no GPU
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def compute_cupy_gpu(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using CuPy (CUDA GPU).

    Intentionally includes GPU→CPU transfer in this phase to measure full
    round-trip cost (upload weights + upload returns + matmul + download).

    Parameters
    ----------
    returns : (T, U) log-returns matrix (will be cast to float64)
    weights : (N, U) portfolio weight matrix (will be cast to float64)

    Returns
    -------
    results : (N, 2) NumPy array — columns [cumulative_return, annualised_sharpe]

    Raises
    ------
    ImportError if cupy is not installed.
    RuntimeError if no CUDA GPU is accessible.
    """
    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError(
            "CuPy is not installed. Install with:\n"
            "  uv pip install cupy-cuda12x  (for CUDA 12.x)\n"
            "  uv pip install cupy-cuda11x  (for CUDA 11.x)"
        ) from e

    if not is_available():
        raise RuntimeError(
            "No CUDA GPU found. CuPy requires a CUDA-capable GPU."
        )

    # Upload to GPU (included in timing — full round-trip cost)
    W_gpu = cp.asarray(weights, dtype=cp.float64)  # (N, U)
    R_gpu = cp.asarray(returns, dtype=cp.float64)   # (T, U)

    # Portfolio log returns: (N, U) @ (U, T) → (N, T)
    port_returns_gpu = W_gpu @ R_gpu.T  # shape (N, T)

    # Cumulative return: exp(sum over T) - 1
    cum_returns_gpu = cp.expm1(port_returns_gpu.sum(axis=1))  # (N,)

    # Annualised Sharpe (Rf = 0, ddof=1)
    mean_r_gpu = port_returns_gpu.mean(axis=1)           # (N,)
    std_r_gpu = port_returns_gpu.std(axis=1, ddof=1)     # (N,)
    sqrt_annual = float(np.sqrt(_TRADING_DAYS_PER_YEAR))
    sharpe_gpu = cp.where(
        std_r_gpu > 0,
        mean_r_gpu / std_r_gpu * sqrt_annual,
        0.0,
    )

    # Stack and download to CPU (included in timing)
    results_gpu = cp.column_stack([cum_returns_gpu, sharpe_gpu])  # (N, 2)
    cp.cuda.Stream.null.synchronize()  # ensure GPU work is complete before transfer
    results: np.ndarray = cp.asnumpy(results_gpu)

    return results
