"""
ctypes wrapper for libportfolio_go.so (Go goroutines).

The Go kernel uses a static goroutine worker pool (runtime.GOMAXPROCS workers,
one chunk per worker) over the N portfolios.  The inner dot-product loop is
scalar — Go's gc compiler does not auto-vectorise FP reductions.

Expected performance: ~1.5–2.5× NumPy at N=1M.  Goroutine scheduling overhead
is minimal at this chunk granularity, but the lack of SIMD vectorisation in the
inner loop is the binding constraint.

Usage
-----
    from src.compute.go_goroutines import compute_go_goroutines
    results = compute_go_goroutines(returns, weights)  # (N, 2) float64

Build the library first (requires Go 1.22+):
    make -C implementations/go/goroutines build

Install Go if missing:
    # User-local (no sudo):
    curl -sL https://go.dev/dl/go1.22.12.linux-amd64.tar.gz | \\
        tar -xz -C ~/.local/go-sdk
    # Then add ~/.local/go-sdk/go/bin to PATH
    # Or: sudo apt-get install golang-go
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_ROOT   = Path(__file__).resolve().parent.parent.parent
_SO_PATH = (
    _ROOT / "implementations" / "go" / "goroutines" / "build" / "libportfolio_go.so"
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
            "  make -C implementations/go/goroutines build\n"
            "Requires Go 1.22+ in PATH or at ~/.local/go-sdk/go/bin/go"
        )

    lib = ctypes.CDLL(str(_SO_PATH))

    # void ComputePortfolioMetrics(
    #     double* r, double* w, double* out,
    #     int64_t N, int64_t T, int64_t U
    # )
    lib.ComputePortfolioMetrics.restype = None
    lib.ComputePortfolioMetrics.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # r
        ctypes.POINTER(ctypes.c_double),  # w
        ctypes.POINTER(ctypes.c_double),  # out
        ctypes.c_int64,                   # N
        ctypes.c_int64,                   # T
        ctypes.c_int64,                   # U
    ]

    _lib = lib
    return _lib


def compute_go_goroutines(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using Go goroutines.

    Parameters
    ----------
    returns : (T, U) log-returns matrix, C-order
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

    lib.ComputePortfolioMetrics(
        R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(N),
        ctypes.c_int64(T),
        ctypes.c_int64(U),
    )

    return results
