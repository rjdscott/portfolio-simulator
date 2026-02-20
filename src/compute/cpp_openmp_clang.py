"""
ctypes wrapper for libportfolio_openmp_clang.so.

Engine: cpp_openmp_clang

Identical source to cpp_openmp (portfolio_compute.cpp), compiled with
Clang/LLVM instead of GCC. Clang's auto-vectorizer handles FP reductions
more aggressively under -ffast-math, matching Numba's LLVM backend.

This isolates the compiler backend as a controlled variable, answering:
  "How much of the C++/Numba gap is the compiler, not the code?"

Build (separate build directory to avoid GCC/Clang target collision):
    cmake -S implementations/cpp/openmp \\
          -B implementations/cpp/openmp/build_clang \\
          -DCMAKE_CXX_COMPILER=clang++ \\
          -DCMAKE_BUILD_TYPE=Release \\
          -DBUILD_CLANG_TARGET=ON
    cmake --build implementations/cpp/openmp/build_clang --parallel

Output: implementations/cpp/openmp/build_clang/libportfolio_openmp_clang.so
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
    / "build_clang"
    / "libportfolio_openmp_clang.so"
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
            "  cmake -S implementations/cpp/openmp \\\n"
            "        -B implementations/cpp/openmp/build_clang \\\n"
            "        -DCMAKE_CXX_COMPILER=clang++ \\\n"
            "        -DCMAKE_BUILD_TYPE=Release \\\n"
            "        -DBUILD_CLANG_TARGET=ON\n"
            "  cmake --build implementations/cpp/openmp/build_clang --parallel"
        )

    lib = ctypes.CDLL(str(_SO_PATH))

    # Reuses the same C function signature as the baseline cpp_openmp engine
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


def compute_cpp_openmp_clang(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using the identical C++ source
    as cpp_openmp, compiled with Clang/LLVM.

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
