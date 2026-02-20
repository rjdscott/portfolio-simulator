"""
ctypes wrapper for libportfolio_openmp_unroll.so.

Engine: cpp_openmp_unroll

Optimization over cpp_openmp:
  - 8 independent scalar accumulators per dot-product (→ 2 AVX2 ymm FMA chains
    with -ffast-math, filling both execution ports)
  - Day-loop tiling (B_t=40) to keep 32 KB of R rows in L1d cache

Build:
    cmake -S implementations/cpp/openmp -B implementations/cpp/openmp/build
          -DCMAKE_BUILD_TYPE=Release
    cmake --build implementations/cpp/openmp/build --parallel
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_SO_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "implementations"
    / "cpp"
    / "openmp"
    / "build"
    / "libportfolio_openmp_unroll.so"
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
            "  cmake -S implementations/cpp/openmp -B implementations/cpp/openmp/build "
            "-DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build implementations/cpp/openmp/build --parallel"
        )

    lib = ctypes.CDLL(str(_SO_PATH))

    # void compute_portfolio_metrics_unroll(
    #     const double* R, const double* W, double* results,
    #     int64_t N, int64_t T, int64_t U
    # )
    lib.compute_portfolio_metrics_unroll.restype = None
    lib.compute_portfolio_metrics_unroll.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # R
        ctypes.POINTER(ctypes.c_double),  # W
        ctypes.POINTER(ctypes.c_double),  # results
        ctypes.c_int64,                   # N
        ctypes.c_int64,                   # T
        ctypes.c_int64,                   # U
    ]

    _lib = lib
    return _lib


def compute_cpp_openmp_unroll(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using C++ + OpenMP,
    8-wide accumulator unroll + L1-tiled day loop.

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
    T = R.shape[0]

    results = np.empty((N, 2), dtype=np.float64)

    lib.compute_portfolio_metrics_unroll(
        R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(N),
        ctypes.c_int64(T),
        ctypes.c_int64(U),
    )

    return results
