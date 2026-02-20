"""
ctypes wrapper for libportfolio_fortran.so (FORTRAN 2008 + OpenMP).

The FORTRAN kernel uses ISO_C_BINDING for C-compatible ABI, OpenMP parallel
do over N portfolios, and -ffast-math for SIMD FP reductions.  The build
produces a shared library with the same function signature as the C++ engine.

Expected performance: within ±5% of cpp_openmp — both compile via the GCC
backend with the same -ffast-math / -march=native flags.

Usage
-----
    from src.compute.fortran_openmp import compute_fortran_openmp
    results = compute_fortran_openmp(returns, weights)  # (N, 2) float64

Build the library first (requires gfortran):
    # Via CMake (recommended):
    cmake -S implementations/fortran/openmp \\
          -B implementations/fortran/openmp/build \\
          -DCMAKE_BUILD_TYPE=Release
    cmake --build implementations/fortran/openmp/build --parallel

    # Or direct gfortran:
    make -C implementations/fortran/openmp build

Install gfortran if missing:
    sudo apt-get install gfortran
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent

# CMake build output
_SO_PATH_CMAKE = (
    _ROOT / "implementations" / "fortran" / "openmp" / "build" / "libportfolio_fortran.so"
)
# Direct make output
_SO_PATH_MAKE = (
    _ROOT / "implementations" / "fortran" / "openmp" / "build" / "libportfolio_fortran.so"
)

_lib: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    so_path = _SO_PATH_CMAKE if _SO_PATH_CMAKE.exists() else _SO_PATH_MAKE

    if not so_path.exists():
        raise FileNotFoundError(
            f"Shared library not found: {so_path}\n"
            "Build it with (requires gfortran + OpenMP):\n"
            "  cmake -S implementations/fortran/openmp "
            "-B implementations/fortran/openmp/build -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build implementations/fortran/openmp/build --parallel\n"
            "Install gfortran: sudo apt-get install gfortran"
        )

    lib = ctypes.CDLL(str(so_path))

    # subroutine compute_portfolio_metrics(r_ptr, w_ptr, out_ptr, N, T, U) BIND(C)
    #   type(C_PTR),        value :: r_ptr, w_ptr, out_ptr
    #   integer(C_INT64_T), value :: N, T, U
    lib.compute_portfolio_metrics.restype = None
    lib.compute_portfolio_metrics.argtypes = [
        ctypes.c_void_p,  # r_ptr
        ctypes.c_void_p,  # w_ptr
        ctypes.c_void_p,  # out_ptr
        ctypes.c_int64,   # N
        ctypes.c_int64,   # T
        ctypes.c_int64,   # U
    ]

    _lib = lib
    return _lib


def compute_fortran_openmp(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using FORTRAN + OpenMP.

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

    lib.compute_portfolio_metrics(
        R.ctypes.data_as(ctypes.c_void_p),
        W.ctypes.data_as(ctypes.c_void_p),
        results.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(N),
        ctypes.c_int64(T),
        ctypes.c_int64(U),
    )

    return results
