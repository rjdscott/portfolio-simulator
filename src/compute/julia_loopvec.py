"""
juliacall wrapper for the Julia LoopVectorization engine.

Starts a Julia session once per Python process via juliacall (in-process,
no subprocess overhead).  Passes numpy arrays directly via juliacall's memory
bridge — the arrays are wrapped as Julia matrices without copying.

The Julia kernel uses:
  - Threads.@threads for outer parallelism (one thread per core)
  - @turbo (LoopVectorization.jl) for inner SIMD vectorisation of the
    dot product — polyhedral analysis, no -ffast-math language restriction
  - Welford online variance (ddof=1)

Memory layout: numpy C-order (T, U) arrays are transposed and converted to
Fortran-order (column-major) before passing to Julia.  Julia sees (U, T) and
(U, N) column-major matrices with unit-stride on the stock (U) axis, enabling
@turbo SIMD vectorisation of the inner dot-product loop.

Expected performance: near or exceeding Numba at N=1M due to @turbo's
more aggressive polyhedral SIMD analysis.

Usage
-----
    from src.compute.julia_loopvec import compute_julia_loopvec
    results = compute_julia_loopvec(returns, weights)  # (N, 2) float64

Prerequisites
-------------
    # Install Julia (user-local, no sudo):
    curl -fsSL https://install.julialang.org | sh -s -- --yes
    source ~/.bashrc  # or restart shell so 'julia' is in PATH

    # Install Python bridge:
    uv pip install juliacall

    # Install Julia dependencies (first run will also trigger this):
    julia --project=implementations/julia -e 'using Pkg; Pkg.instantiate()'
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Both vars must be set before juliacall is first imported.
# JULIA_NUM_THREADS: "auto" = one Julia thread per logical CPU.
# PYTHON_JULIACALL_HANDLE_SIGNALS: required for safe multithreaded Julia + Python
# coexistence (juliacall docs); disables Python's default Ctrl-C handling.
os.environ.setdefault("JULIA_NUM_THREADS", "auto")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

_ROOT        = Path(__file__).resolve().parent.parent.parent
_JL_SRC      = _ROOT / "implementations" / "julia" / "src" / "portfolio_compute.jl"
_JL_PROJECT  = _ROOT / "implementations" / "julia"

_jl_session  = None
_compute_fn  = None


def _start_julia() -> None:
    global _jl_session, _compute_fn
    if _jl_session is not None:
        return

    try:
        import juliacall
    except ImportError as exc:
        raise ImportError(
            "juliacall is required for the Julia engine.\n"
            "Install with: uv pip install juliacall\n"
            "Then install Julia: curl -fsSL https://install.julialang.org | sh -s -- --yes"
        ) from exc

    if not _JL_SRC.exists():
        raise FileNotFoundError(
            f"Julia source not found: {_JL_SRC}\n"
            "The implementations/julia/ directory should be part of the repository."
        )

    # Start Julia session and load the kernel
    # juliacall uses the Julia found via juliaup or PATH
    jl = juliacall.newmodule("PortfolioComputeWrapper")

    # Activate the Julia project to resolve LoopVectorization.jl
    jl.seval(f'import Pkg; Pkg.activate("{_JL_PROJECT}")')
    jl.seval("import Pkg; Pkg.instantiate()")
    jl.seval("using LoopVectorization")
    jl.seval("using Base.Threads")
    jl.include(str(_JL_SRC))

    _jl_session = jl
    _compute_fn = jl.compute_portfolio_metrics


def compute_julia_loopvec(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using Julia + LoopVectorization.

    Parameters
    ----------
    returns : (T, U) log-returns matrix, C-order
    weights : (N, U) portfolio weight matrix

    Returns
    -------
    results : (N, 2) float64 array — [cumulative_return, annualised_sharpe]
    """
    _start_julia()

    # Transpose to (U, T) and (U, N) in Fortran-order (column-major) so that
    # the stock axis (u) has unit stride — optimal for the inner @turbo loop.
    # Pass raw data pointers + shapes to Julia; the kernel uses unsafe_wrap
    # to create native Julia arrays (satisfying @turbo's check_args) without
    # any element-by-element copy.  Keep R/W alive for the duration of the call.
    R = np.asfortranarray(returns.T.astype(np.float64))  # (U, T) F-order
    W = np.asfortranarray(weights.T.astype(np.float64))  # (U, N) F-order

    U, T = R.shape
    _U, N = W.shape

    result_jl = _compute_fn(
        int(R.ctypes.data),  # Ptr{Float64} to returns data
        int(W.ctypes.data),  # Ptr{Float64} to weights data
        int(U), int(T), int(N),
    )

    # Convert Julia matrix back to numpy
    return np.asarray(result_jl, dtype=np.float64)
