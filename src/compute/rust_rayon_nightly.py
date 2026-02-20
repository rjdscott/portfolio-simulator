"""
ctypes wrapper for libportfolio_rayon_nightly.so (Rust nightly + fadd_fast).

Identical interface to rust_rayon.py.  The underlying kernel uses
std::intrinsics::fadd_fast on the inner dot-product accumulation, which
annotates each FP addition with LLVM's 'fast' flag.  This allows LLVM to
reassociate the reduction and emit vectorised vfmadd231pd — the same code path
that GCC takes with -ffast-math.

Expected result: closes the ~1.9× gap between rust_rayon (stable) and
cpp_openmp / numba_parallel.

Usage
-----
    from src.compute.rust_rayon_nightly import compute_rust_rayon_nightly
    results = compute_rust_rayon_nightly(returns, weights)  # (N, 2) float64

Build the library first:
    cargo +nightly build --release \\
        --manifest-path implementations/rust/rayon_nightly/Cargo.toml
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_SO_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "implementations"
    / "rust"
    / "rayon_nightly"
    / "target"
    / "release"
    / "libportfolio_rayon_nightly.so"
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
            "  cargo +nightly build --release "
            "--manifest-path implementations/rust/rayon_nightly/Cargo.toml"
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


def compute_rust_rayon_nightly(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using Rust nightly (fadd_fast).

    Parameters
    ----------
    returns : (T, U) log-returns matrix
    weights : (N, U) portfolio weight matrix

    Returns
    -------
    results : (N, 2) float64 array — [cumulative_return, annualised_sharpe]
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
