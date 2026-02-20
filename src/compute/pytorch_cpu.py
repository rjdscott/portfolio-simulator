"""
PyTorch CPU compute engine.

Uses PyTorch's ATen/MKL thread pool for the (N × U) × (U × T) matrix
multiplication.  PyTorch is imported lazily on first call so that the module
can be imported without PyTorch installed (the engine is silently skipped
during phase sweeps when torch is unavailable).

The matmul and statistical aggregation stay in float64 throughout, so the
result_checksum should match numpy_vectorised within floating-point
rounding tolerance.

Usage
-----
    from src.compute.pytorch_cpu import compute_pytorch_cpu
    results = compute_pytorch_cpu(returns, weights)  # (N, 2) float64

Install
-------
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import numpy as np


def compute_pytorch_cpu(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using PyTorch (CPU).

    Parameters
    ----------
    returns : (T, U) log-returns matrix
    weights : (N, U) portfolio weight matrix

    Returns
    -------
    results : (N, 2) float64 array — [cumulative_return, annualised_sharpe]

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    import torch

    R = torch.from_numpy(np.ascontiguousarray(returns, dtype=np.float64))
    W = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float64))

    # W @ R.T → (N, T); .numpy() is a zero-copy view when tensor is contiguous
    port_returns = (W @ R.T).numpy()  # (N, T) float64

    cum_ret = np.expm1(port_returns.sum(axis=1))
    mean_r  = port_returns.mean(axis=1)
    std_r   = port_returns.std(axis=1, ddof=1)
    sharpe  = np.where(std_r > 0, mean_r / std_r * np.sqrt(252.0), 0.0)

    return np.column_stack([cum_ret, sharpe])
