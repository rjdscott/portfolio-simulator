"""
ctypes wrapper for libportfolio_openmp.so.

Loads the shared library on first call and exposes a Python-friendly interface.

Usage
-----
    from portfolio_openmp import compute_cpp_openmp
    results = compute_cpp_openmp(returns, weights)  # (N, 2) float64

Build the library first:
    make -C implementations/cpp/openmp/
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

# Shared library path relative to this file: ../build/libportfolio_openmp.so
_SO_PATH = Path(__file__).resolve().parent.parent / "build" / "libportfolio_openmp.so"

_lib: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    if not _SO_PATH.exists():
        raise FileNotFoundError(
            f"Shared library not found: {_SO_PATH}\n"
            "Build it with:\n"
            "  make -C implementations/cpp/openmp/"
        )

    lib = ctypes.CDLL(str(_SO_PATH))

    # void compute_portfolio_metrics(
    #     const double* R, const double* W, double* results,
    #     int64_t N, int64_t T, int64_t U
    # )
    lib.compute_portfolio_metrics.restype = None
    lib.compute_portfolio_metrics.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # R
        ctypes.POINTER(ctypes.c_double),  # W
        ctypes.POINTER(ctypes.c_double),  # results
        ctypes.c_int64,                   # N
        ctypes.c_int64,                   # T
        ctypes.c_int64,                   # U
    ]

    _lib = lib
    return _lib


def compute_cpp_openmp(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using C++ + OpenMP.

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

    lib.compute_portfolio_metrics(
        R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        W.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(N),
        ctypes.c_int64(T),
        ctypes.c_int64(U),
    )

    return results
