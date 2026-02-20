"""
JAX CPU compute engine.

JAX JIT-compiles the matmul + statistics kernel on first call (warmup run).
Subsequent calls hit the compiled XLA kernel directly.

JAX is forced onto the CPU backend via the JAX_PLATFORM_NAME environment
variable set before any jax import.  The module is imported lazily so the
file can be imported without jax installed.

Bessel correction note: jnp.std defaults to ddof=0.  The correction factor
sqrt(T / (T-1)) is applied to match the ddof=1 definition used by all other
engines.

Usage
-----
    from src.compute.jax_cpu import compute_jax_cpu
    results = compute_jax_cpu(returns, weights)  # (N, 2) float64

Install
-------
    uv pip install "jax[cpu]"
"""

from __future__ import annotations

import os

import numpy as np

# Force CPU backend before any jax import in this module.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

_JAX_KERNEL = None   # cached JIT-compiled function (module-level singleton)


def compute_jax_cpu(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using JAX (CPU, JIT).

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
        If JAX is not installed.
    """
    import jax
    import jax.numpy as jnp

    global _JAX_KERNEL

    if _JAX_KERNEL is None:
        @jax.jit
        def _kernel(R, W):
            pr     = W @ R.T                        # (N, T)
            cum    = jnp.expm1(pr.sum(axis=1))
            mean_  = pr.mean(axis=1)
            T      = pr.shape[1]
            # ddof=1 correction: std_ddof1 = std_ddof0 * sqrt(T / (T-1))
            std_   = pr.std(axis=1) * jnp.sqrt(T / (T - 1))
            sharpe = jnp.where(std_ > 0, mean_ / std_ * jnp.sqrt(252.0), 0.0)
            return jnp.stack([cum, sharpe], axis=1)

        _JAX_KERNEL = _kernel

    R = jnp.array(np.ascontiguousarray(returns, dtype=np.float64))
    W = jnp.array(np.ascontiguousarray(weights, dtype=np.float64))

    return np.array(_JAX_KERNEL(R, W))
