"""
ctypes wrapper for libportfolio_openmp_tile4.so.

Engine: cpp_openmp_tile4

Key optimization: vectorize across 4 portfolios simultaneously using AVX2
registers (4 × double), instead of vectorizing over U stocks for one portfolio.

W is transposed to (U, N) in Python before the C call. This allows:
  - Broadcasting each R[t, u] scalar across 4 portfolio dot-products (1 _mm256_set1_pd)
  - Loading 4 weights from contiguous memory in one _mm256_loadu_pd
  - 4× reduction in unique R cache-line fetches vs. per-portfolio processing

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
    / "libportfolio_openmp_tile4.so"
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

    # void compute_portfolio_metrics_tile4(
    #     const double* R, const double* W_T, double* results,
    #     int64_t N, int64_t T, int64_t U
    # )
    lib.compute_portfolio_metrics_tile4.restype = None
    lib.compute_portfolio_metrics_tile4.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # R      (T, U)
        ctypes.POINTER(ctypes.c_double),  # W_T    (U, N) transposed
        ctypes.POINTER(ctypes.c_double),  # results (N, 2)
        ctypes.c_int64,                   # N
        ctypes.c_int64,                   # T
        ctypes.c_int64,                   # U
    ]

    _lib = lib
    return _lib


def compute_cpp_openmp_tile4(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using C++ + OpenMP,
    4-portfolio AVX2 tile with transposed W layout.

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

    # Transpose W: (N, U) → (U, N) contiguous — the C kernel expects W_T[u*N + i]
    # np.ascontiguousarray ensures the transpose is in C-contiguous (row-major) layout.
    W_T = np.ascontiguousarray(W.T, dtype=np.float64)  # shape (U, N)

    results = np.empty((N, 2), dtype=np.float64)

    lib.compute_portfolio_metrics_tile4(
        R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        W_T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(N),
        ctypes.c_int64(T),
        ctypes.c_int64(U),
    )

    return results
