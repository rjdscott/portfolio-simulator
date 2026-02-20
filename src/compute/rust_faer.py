"""
ctypes wrapper for libportfolio_faer.so (Rust + faer linear algebra library).

faer is a pure-Rust BLAS-level library that auto-parallelises matrix
multiplication via Rayon.  The kernel uses faer's zero-copy matrix views
over the raw C pointers, calls faer's matmul for W * R^T, then computes
per-portfolio Welford statistics in parallel.

Expected performance: closes or exceeds the rust_rayon (stable) throughput
because faer's matmul is optimised with explicit SIMD at the library level,
independently of LLVM's FP-reassociation restrictions.

Usage
-----
    from src.compute.rust_faer import compute_rust_faer
    results = compute_rust_faer(returns, weights)  # (N, 2) float64

Build the library first:
    source $HOME/.cargo/env
    cargo build --release \\
        --manifest-path implementations/rust/faer/Cargo.toml
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_SO_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "implementations"
    / "rust"
    / "faer"
    / "target"
    / "release"
    / "libportfolio_faer.so"
)

_lib: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    if not _SO_PATH.exists():
        raise FileNotFoundError(
            f"Shared library not found: {_SO_PATH}\n"
            "Build it with:\n"
            "  source $HOME/.cargo/env\n"
            "  cargo build --release "
            "--manifest-path implementations/rust/faer/Cargo.toml"
        )

    lib = ctypes.CDLL(str(_SO_PATH))

    # void compute_portfolio_metrics(
    #     const double* r, const double* w, double* results,
    #     int64_t n, int64_t t, int64_t u
    # )
    lib.compute_portfolio_metrics.restype = None
    lib.compute_portfolio_metrics.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # r (returns)
        ctypes.POINTER(ctypes.c_double),  # w (weights)
        ctypes.POINTER(ctypes.c_double),  # results
        ctypes.c_int64,                   # n
        ctypes.c_int64,                   # t
        ctypes.c_int64,                   # u
    ]

    _lib = lib
    return _lib


def compute_rust_faer(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using Rust + faer.

    Parameters
    ----------
    returns : (T, U) log-returns matrix
    weights : (N, U) portfolio weight matrix

    Returns
    -------
    results : (N, 2) float64 array — [cumulative_return, annualised_sharpe]

    Raises
    ------
    FileNotFoundError
        If the shared library has not been built yet.
    """
    lib = _load_lib()

    R = np.ascontiguousarray(returns, dtype=np.float64)
    W = np.ascontiguousarray(weights, dtype=np.float64)

    N, U = W.shape
    T    = R.shape[0]

    results = np.empty((N, 2), dtype=np.float64)

    lib.compute_portfolio_metrics(
        R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(N),
        ctypes.c_int64(T),
        ctypes.c_int64(U),
    )

    return results
