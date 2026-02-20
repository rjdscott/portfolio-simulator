"""
ctypes wrapper for libportfolio_eigen.so (C++ Eigen header-only library).

The Eigen kernel uses Eigen::Map to wrap raw C pointers (zero copy), then
calls Eigen's BLAS-backed matmul (W * R^T) and computes per-row statistics
using Eigen array operations.  OpenMP is still used for parallelism over N.

Usage
-----
    from src.compute.cpp_eigen import compute_cpp_eigen
    results = compute_cpp_eigen(returns, weights)  # (N, 2) float64

Build the library first:
    cmake -S implementations/cpp/eigen -B implementations/cpp/eigen/build \\
          -DCMAKE_BUILD_TYPE=Release
    cmake --build implementations/cpp/eigen/build --parallel

Prerequisite:
    sudo apt install libeigen3-dev
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_SO_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "implementations"
    / "cpp"
    / "eigen"
    / "build"
    / "libportfolio_eigen.so"
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
            "  cmake -S implementations/cpp/eigen "
            "-B implementations/cpp/eigen/build -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build implementations/cpp/eigen/build --parallel"
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


def compute_cpp_eigen(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using C++ Eigen.

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
