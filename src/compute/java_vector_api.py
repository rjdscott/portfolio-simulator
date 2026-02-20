"""
JPype wrapper for PortfolioCompute.jar (Java 21+ Vector API).

Starts a JVM once per Python process via JPype (in-process, no subprocess
overhead).  Passes numpy arrays to the Java method as Java double[] via
JPype's direct buffer protocol — zero serialisation in the timed window.

The Java kernel uses:
  - jdk.incubator.vector.DoubleVector (SPECIES_256 = 4 doubles per AVX2 lane)
  - IntStream.range(0, N).parallel() dispatched to ForkJoinPool.commonPool()
  - Welford online variance (ddof=1)

Expected performance: 2–4× NumPy at N=1M.  HotSpot C2 JIT compiles the
hot loop to native AVX2 after ~100 iterations (warmup run handles this).

Usage
-----
    from src.compute.java_vector_api import compute_java_vector_api
    results = compute_java_vector_api(returns, weights)  # (N, 2) float64

Prerequisites
-------------
    uv pip install jpype1
    # Java 21+ must be in PATH (system java is used by default)
    # Build the JAR first:
    make -C implementations/java/vector_api build
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_ROOT    = Path(__file__).resolve().parent.parent.parent
_JAR     = _ROOT / "implementations" / "java" / "vector_api" / "dist" / "portfolio_vector_api.jar"

_jvm_started: bool = False
_PortfolioCompute = None


def _start_jvm() -> None:
    global _jvm_started, _PortfolioCompute
    if _jvm_started:
        return

    try:
        import jpype
        import jpype.imports
    except ImportError as exc:
        raise ImportError(
            "jpype1 is required for the Java Vector API engine.\n"
            "Install with: uv pip install jpype1"
        ) from exc

    if not _JAR.exists():
        raise FileNotFoundError(
            f"Java JAR not found: {_JAR}\n"
            "Build it with:\n"
            "  make -C implementations/java/vector_api build"
        )

    if not jpype.isJVMStarted():
        # JVM args are positional *jvmargs, not a keyword argument.
        # --add-modules is required for jdk.incubator.vector on Java 21-25.
        jpype.startJVM(
            "--add-modules=jdk.incubator.vector",
            classpath=[str(_JAR)],
            convertStrings=False,
        )

    from com.portfoliosimulator import PortfolioCompute  # type: ignore[import]
    _PortfolioCompute = PortfolioCompute
    _jvm_started = True


def compute_java_vector_api(
    returns: np.ndarray,   # shape (T, U) float64
    weights: np.ndarray,   # shape (N, U) float32 or float64
) -> np.ndarray:           # shape (N, 2) float64
    """
    Compute cumulative return and Sharpe ratio using Java Vector API.

    Parameters
    ----------
    returns : (T, U) log-returns matrix, C-order
    weights : (N, U) portfolio weight matrix

    Returns
    -------
    results : (N, 2) float64 array — [cumulative_return, annualised_sharpe]
    """
    import jpype

    _start_jvm()

    R = np.ascontiguousarray(returns, dtype=np.float64)
    W = np.ascontiguousarray(weights, dtype=np.float64)

    N, U = W.shape
    T    = R.shape[0]

    # Flatten to 1-D for the Java API (which uses flat arrays with manual indexing)
    r_flat   = R.ravel()
    w_flat   = W.ravel()
    out_flat = np.zeros(N * 2, dtype=np.float64)

    # Convert to Java double[] via JPype's JArray bridge
    # JPype maps numpy float64 arrays to Java double[] without copying
    j_r   = jpype.JArray(jpype.JDouble)(r_flat)
    j_w   = jpype.JArray(jpype.JDouble)(w_flat)
    j_out = jpype.JArray(jpype.JDouble)(out_flat)

    _PortfolioCompute.computeMetrics(j_r, j_w, j_out, N, T, U)

    # Copy results back from Java array to numpy
    out_flat[:] = np.frombuffer(bytes(j_out), dtype=np.float64)

    return out_flat.reshape(N, 2)
