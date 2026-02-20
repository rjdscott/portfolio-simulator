# Phase 3b / 3c — Extended Compute Engine Analysis

**Date recorded:** 2026-02-20
**Status:** Phase 3b complete (7 engines, full sweep); Phase 3c partial (preliminary N=1K runs)
**Context:** Following Phase 3's finding that portfolio-parallel Welford outperforms BLAS
at every scale, these phases extend the compute engine comparison to eleven additional
engines across 7 languages and runtimes: Java (Vector API), Polars, Rust nightly
(`fadd_fast`), Rust faer, DuckDB, FORTRAN, Julia, Go, NumPy float32, PyTorch CPU, and JAX.

## Phase 3b Results Summary

| Engine              | N=100  | N=1K    | N=100K  | N=1M    | vs NumPy (1M) |
|---------------------|--------|---------|---------|---------|---------------|
| fortran_openmp      | 20,517 | 160,694 | 779,029 | 828,035 | **4.7×**      |
| julia_loopvec       | 23,250 | 199,681 | 707,409 | 745,475 | **4.2×**      |
| rust_rayon_nightly  | 10,045 | 79,994  | 655,304 | 654,643 | **3.7×**      |
| java_vector_api     | 19,585 | 117,481 | 460,764 | 476,253 | **2.7×**      |
| go_goroutines       | 20,773 | 118,287 | 323,932 | 344,630 | **1.9×**      |
| polars_engine       | 1,943  | 22,219  | 105,927 | 83,811  | 0.47×         |
| duckdb_sql          | 884    | 1,119   | 11,360  | —       | —             |

## Phase 3c Preliminary Results (N=1K only)

| Engine         | N=1K    | Notes |
|----------------|---------|-------|
| rust_faer      | 162,549 | faer BLAS GEMM + Rayon Welford |
| numpy_float32  | 49,408  | float32 dtype; cast overhead dominates at small N |

---

## Motivation

Phase 3 established that explicit parallelism + SIMD beats BLAS for this workload.
The remaining open questions are:

1. **Can the JVM close the native-code gap?** Java 21's Vector API gives explicit AVX2 SIMD
   without relying on JIT auto-vectorisation heuristics. If HotSpot can manage the overhead,
   this directly challenges the "JVM can't do SIMD" assumption.
2. **Does a high-level Rust-backed DataFrame (Polars) compete?** Polars uses Rayon internally
   — the same library as our `rust_rayon` engine. Can its expression API achieve similar
   throughput without writing any Rust?
3. **Can Rust reach parity with C++ using nightly fast-math?** Rust stable trails by ~1.9×
   because `stable` has no `-ffast-math` equivalent. Nightly's `std::intrinsics::fadd_fast`
   removes that restriction.
4. **Can SQL engines handle numerical compute?** DuckDB is increasingly adopted in quant
   fintech. This directly measures the cost of expressing the workload in SQL vs SIMD loops.
5. **Is FORTRAN still peak?** The traditional numerical computing language with gfortran
   `-ffast-math -O3` and OpenMP serves as a historical baseline.
6. **How does Julia's `@turbo` macro compare to Numba `fastmath=True`?** Both compile to
   LLVM IR with SIMD, but Julia is a whole-language JIT vs a Python-embedded kernel.
7. **How does Go's goroutine scheduler compare to Rayon/OpenMP work-stealing?** Go is
   widely used in finance infrastructure; its parallelism model is different from thread pools.

---

## Engine-by-Engine Analysis

### 1. Polars (`polars_engine`)

**Language:** Python (Rust backend via PyO3)
**Parallelism:** Rayon thread pool (same as `rust_rayon` engine)
**Install:** `uv pip install polars`

**Approach:**
Polars' native strength is columnar predicate-pushdown and Parquet I/O rather than custom
numerical kernels. For a head-to-head compute comparison (all engines receive pre-loaded
numpy arrays), Polars must express the portfolio-return computation via its expression API:

1. Convert numpy arrays to Polars DataFrames
2. Use `map_batches` with a numpy kernel (tests Python↔Polars overhead)
3. Or: use native Polars horizontal aggregation (e.g., `pl.sum_horizontal`) where possible

**Key hypothesis (H-PL1):** Polars cannot express Welford variance natively in its expression
API. `map_elements` incurs Python overhead per row and will be slower than all Phase 3
engines. However, Polars' Parquet reader may outperform PyArrow for the **loading step** —
worth measuring as a secondary finding.

**Expected range:** 50K–150K portfolios/sec (between `pandas_baseline` and `numpy_vectorised`
for the compute step; potentially faster than PyArrow for loading).

**Integration:**
Pure Python, no native build. `src/compute/polars_engine.py`

---

### 2. DuckDB (`duckdb_sql`)

**Language:** Python (C++ backend)
**Parallelism:** DuckDB internal parallel execution
**Install:** `uv pip install duckdb`

**Approach:**
Express the entire computation as SQL:

```sql
-- Phase 1: compute daily portfolio returns
WITH portfolio_daily AS (
    SELECT
        w.portfolio_id,
        SUM(w.weight * r.log_return) AS daily_log_return
    FROM weights_tbl w
    JOIN returns_tbl r USING (ticker_idx)
    GROUP BY w.portfolio_id, r.day_idx
),
-- Phase 2: aggregate to cumulative return and Sharpe ratio
aggregated AS (
    SELECT
        portfolio_id,
        EXP(SUM(daily_log_return)) - 1                           AS cum_return,
        AVG(daily_log_return) / STDDEV_SAMP(daily_log_return)
            * SQRT(252.0)                                        AS sharpe
    FROM portfolio_daily
    GROUP BY portfolio_id
)
SELECT portfolio_id, cum_return, sharpe
FROM aggregated
ORDER BY portfolio_id
```

The data must be presented in long format (rows: portfolio × ticker, values: weight). This
requires reshaping the (N×U) weight matrix and (T×U) returns matrix into long-form relations
before querying — the reshape overhead is included in the timed window.

**Key hypothesis (H-DK1):** The join + aggregation path will be significantly slower than
SIMD loops for this workload. DuckDB excels at aggregating large datasets row-by-row, but the
inner loop here is a dense dot product over U=100 dimensions — not a SQL strength. Estimated
throughput: 10K–50K portfolios/sec. Could be faster at large N if DuckDB's parallel hash
aggregation saturates all cores efficiently.

**Secondary finding:** DuckDB reading Parquet directly (bypassing PyArrow) may load faster
for the price data; this is worth measuring separately.

**Integration:**
Pure Python, no native build. `src/compute/duckdb_sql.py`

---

### 3. Rust nightly — `fadd_fast` (`rust_rayon_nightly`)

**Language:** Rust (nightly toolchain)
**Parallelism:** Rayon `par_chunks_mut` (same as `rust_rayon` stable engine)
**Build:** `rustup toolchain install nightly`, `cargo +nightly build --release`

**The problem with Rust stable:**
Rust stable disallows reassociating floating-point operations for correctness (IEEE 754
compliance). The inner dot-product loop:

```rust
for u in 0..U {
    sum += w[u] * r[u];
}
```

cannot be vectorised because each iteration depends on the previous value of `sum`. The
8-wide accumulator workaround (Phase 3) helps LLVM pack two ymm registers, but GCC with
`-ffast-math` applies more aggressive global reassociation and loop transforms, explaining
the ~1.9× gap.

**Nightly fix:**
`std::intrinsics::fadd_fast` applies the `llvm.fadd.fast` attribute at each operation:

```rust
#![feature(core_intrinsics)]
use std::intrinsics::fadd_fast;

unsafe {
    for u in 0..U {
        sum = fadd_fast(sum, w[u] * r[u]);
    }
}
```

This emits the same LLVM IR hint as `-ffast-math` for that specific operation, allowing
LLVM to vectorise the reduction with `vfmadd231pd`. The 8-wide accumulator trick is
retained as a fallback if the intrinsic underperforms.

**Key hypothesis (H-RN1):** Rust nightly with `fadd_fast` will close or eliminate the 1.9×
gap vs C++/Numba, reaching ~800K–950K portfolios/sec at N=1M.

**Toolchain config:**
```
# implementations/rust/rayon_nightly/rust-toolchain.toml
[toolchain]
channel = "nightly"
```

**Integration:**
ctypes wrapper, same pattern as `rust_rayon` stable. `src/compute/rust_rayon_nightly.py`

---

### 4. FORTRAN OpenMP (`fortran_openmp`)

**Language:** FORTRAN 2008 (ISO_C_BINDING + OpenMP)
**Parallelism:** `!$OMP PARALLEL DO` over portfolios
**Build:** `gfortran -O3 -march=native -ffast-math -fopenmp -shared -fPIC`

**Approach:**
FORTRAN is historically the language of numerical computing and is still used in
quant finance for legacy systems. gfortran with `-ffast-math` applies the same FP
reassociation as GCC's C++ frontend — theoretically producing identical machine code.

Memory layout note: FORTRAN is column-major, but this implementation uses `ISO_C_BINDING`
with `BIND(C)` and linear index arithmetic to accept C-order (row-major) numpy arrays
without copying. The returns matrix passed as a flat `C_PTR` is indexed as `r(t*U + u)`.

```fortran
subroutine compute_portfolio_metrics(r_ptr, w_ptr, out_ptr, N, T, U) BIND(C)
    use ISO_C_BINDING
    implicit none
    integer(C_INT64_T), value :: N, T, U
    type(C_PTR), value        :: r_ptr, w_ptr, out_ptr
    real(C_DOUBLE), pointer   :: r(:), w(:), out(:)
    integer(C_INT64_T)        :: i, t_idx, u_idx
    real(C_DOUBLE)            :: port_r, mean_r, m2, delta, delta2, log_sum

    call C_F_POINTER(r_ptr,   r,   [T*U])
    call C_F_POINTER(w_ptr,   w,   [N*U])
    call C_F_POINTER(out_ptr, out, [N*2])

    !$OMP PARALLEL DO PRIVATE(i,t_idx,u_idx,port_r,mean_r,m2,delta,delta2,log_sum)
    do i = 0, N-1
        log_sum = 0.0_C_DOUBLE
        mean_r  = 0.0_C_DOUBLE
        m2      = 0.0_C_DOUBLE
        do t_idx = 0, T-1
            port_r = 0.0_C_DOUBLE
            !$OMP SIMD REDUCTION(+:port_r)
            do u_idx = 0, U-1
                port_r = port_r + w(i*U + u_idx) * r(t_idx*U + u_idx)
            end do
            log_sum = log_sum + port_r
            delta   = port_r - mean_r
            mean_r  = mean_r + delta / (t_idx + 1)
            delta2  = port_r - mean_r
            m2      = m2 + delta * delta2
        end do
        out(i*2)     = exp(log_sum) - 1.0_C_DOUBLE
        out(i*2 + 1) = mean_r / sqrt(m2 / (T - 1)) * sqrt(252.0_C_DOUBLE)
    end do
    !$OMP END PARALLEL DO
end subroutine
```

**Key hypothesis (H-F1):** FORTRAN with `-ffast-math` will perform within ±5% of C++/OpenMP —
they compile through the same GCC backend and share the same optimisation passes. Any
difference reflects overhead in the ctypes call path, not the compute itself.

**Integration:**
ctypes wrapper, same pattern as `cpp_openmp`. `src/compute/fortran_openmp.py`

---

### 5. Julia — LoopVectorization.jl (`julia_loopvec`)

**Language:** Julia 1.x
**Parallelism:** `Threads.@threads` (Julia thread pool) + `@turbo` (LoopVectorization.jl SIMD)
**Install:** Julia binary + `uv pip install juliacall`

**Why Julia is interesting here:**
Julia is a whole-language JIT compiled via LLVM. Unlike Numba (which JITs a Python-embedded
kernel), Julia compiles the entire function — including bounds checks, type specialisations,
and SIMD hints — as a first-class compiled unit. `LoopVectorization.jl`'s `@turbo` macro
performs polyhedral loop analysis and emits SIMD intrinsics directly, with no `-ffast-math`
caveat: the programmer annotates which loops to vectorise.

```julia
using LoopVectorization
using Base.Threads

function compute_portfolio_metrics!(
    out::Matrix{Float64},
    returns::Matrix{Float64},   # T × U, C-order via unsafe_wrap
    weights::Matrix{Float64},   # N × U
)
    N, U = size(weights)
    T    = size(returns, 1)

    Threads.@threads for i in axes(weights, 1)
        log_sum = 0.0
        mean_r  = 0.0
        m2      = 0.0
        for t in axes(returns, 1)
            port_r = 0.0
            @turbo for u in axes(weights, 2)
                port_r += weights[i, u] * returns[t, u]
            end
            log_sum += port_r
            delta   = port_r - mean_r
            mean_r += delta / t
            m2     += delta * (port_r - mean_r)
        end
        out[i, 1] = expm1(log_sum)
        out[i, 2] = mean_r / sqrt(m2 / (T - 1)) * sqrt(252.0)
    end
end
```

**Integration via juliacall:**
```python
import juliacall
jl = juliacall.newmodule("PortfolioCompute")
jl.seval('using LoopVectorization')
jl.include("implementations/julia/src/portfolio_compute.jl")
# Pass numpy arrays via unsafe_wrap; no copy needed
```

Julia startup is ~5–10 s per session; `juliacall` starts Julia once per Python process
(same amortisation pattern as Numba `cache=True`).

**Key hypothesis (H-JL1):** Julia `@turbo` + `Threads.@threads` will perform within 10–20%
of C++/Numba at N=1M — potentially exceeding Numba given that `@turbo` applies polyhedral
SIMD analysis more aggressively than LLVM's auto-vectoriser with `fastmath=True`.

**Note on matrix layout:**
Julia is column-major; numpy is row-major. Wrapping numpy arrays via `unsafe_wrap` requires
transposing the indexing. The implementation must declare `returns` as `(U, T)` in Julia
and swap index order, or accept a transposed copy. Performance impact should be measured.

---

### 6. Go — goroutines (`go_goroutines`)

**Language:** Go 1.22+
**Parallelism:** Goroutine worker pool (runtime.GOMAXPROCS workers)
**Build:** `go build -buildmode=c-shared -o libportfolio_go.so`

**Approach:**
Go is widely adopted in fintech infrastructure (order management systems, data pipelines)
but rarely used for numerical compute. This benchmark answers: "if a fintech team wrote
this in idiomatic Go, what throughput would they get?"

Go's compiler (gc) is more conservative than GCC/LLVM with `-O3`: it focuses on fast
compilation and correctness over peak throughput. Specifically:
- No `-ffast-math` equivalent (FP is strictly IEEE 754)
- Auto-vectorisation is limited (no SIMD intrinsics in standard library Go)
- Goroutines are cheap (~2KB stack) but have higher scheduling overhead than POSIX threads
  for CPU-bound work (cooperative preemption at 10ms intervals)

```go
package main

/*
#include <stdint.h>
*/
import "C"
import (
    "math"
    "runtime"
    "sync"
    "unsafe"
)

//export ComputePortfolioMetrics
func ComputePortfolioMetrics(
    rPtr *C.double, wPtr *C.double, outPtr *C.double,
    N, T, U C.int64_t,
) {
    n, t, u := int(N), int(T), int(U)
    r   := unsafe.Slice((*float64)(unsafe.Pointer(rPtr)),   t*u)
    w   := unsafe.Slice((*float64)(unsafe.Pointer(wPtr)),   n*u)
    out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), n*2)

    nWorkers := runtime.GOMAXPROCS(0)
    chunkSize := (n + nWorkers - 1) / nWorkers
    var wg sync.WaitGroup
    for start := 0; start < n; start += chunkSize {
        end := start + chunkSize
        if end > n { end = n }
        wg.Add(1)
        go func(lo, hi int) {
            defer wg.Done()
            for i := lo; i < hi; i++ {
                // ... Welford + cumulative return
            }
        }(start, end)
    }
    wg.Wait()
}
```

**Key hypothesis (H-GO1):** Go goroutines will achieve 1.5–2.5× NumPy at N=1M —
comparable to `rust_rayon` stable but trailing C++/Numba due to the lack of fast-math
SIMD vectorisation. The goroutine scheduler overhead is expected to be small at chunk
sizes of N/28 portfolios each (thousands of portfolios per goroutine).

**Integration:**
ctypes wrapper. `src/compute/go_goroutines.py`. Go's c-shared build produces a `.so` and
a `.h` header; ctypes loads the `.so` and calls `ComputePortfolioMetrics`.

---

### 7. Java 21 — Vector API (`java_vector_api`)

**Language:** Java 21
**Parallelism:** `ForkJoinPool.commonPool()` via `IntStream.range(0,N).parallel()`
**SIMD:** `jdk.incubator.vector` — `DoubleVector`, `VectorSpecies`, `VectorOperators`
**Build:** `javac --add-modules jdk.incubator.vector; jar cf`
**Install:** `uv pip install jpype1`

**Why the Vector API matters:**
Java's traditional answer to "JVM vs native SIMD" was "wait for JIT auto-vectorisation."
The Vector API (JEP 417, finalized in Java 21) changes this by giving programmers explicit
control over SIMD lane width and operation type — the same leverage that Java's Vector API
gives as C's AVX intrinsics, but with a safe, portable API.

```java
import jdk.incubator.vector.*;

public class PortfolioCompute {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256; // 4 lanes

    public static void computeMetrics(
        double[] r, double[] w, double[] out, int N, int T, int U
    ) {
        IntStream.range(0, N).parallel().forEach(i -> {
            double logSum = 0.0, mean = 0.0, m2 = 0.0;
            for (int t = 0; t < T; t++) {
                // Vectorised dot product: w[i*U .. i*U+U] · r[t*U .. t*U+U]
                double portR = 0.0;
                int u = 0;
                for (; u <= U - SPECIES.length(); u += SPECIES.length()) {
                    DoubleVector vw = DoubleVector.fromArray(SPECIES, w, i*U + u);
                    DoubleVector vr = DoubleVector.fromArray(SPECIES, r, t*U + u);
                    portR += vw.mul(vr).reduceLanes(VectorOperators.ADD);
                }
                for (; u < U; u++) portR += w[i*U + u] * r[t*U + u];
                // Welford
                logSum += portR;
                double delta = portR - mean;
                mean += delta / (t + 1);
                m2   += delta * (portR - mean);
            }
            out[i*2]     = Math.expm1(logSum);
            out[i*2 + 1] = mean / Math.sqrt(m2 / (T - 1)) * Math.sqrt(252.0);
        });
    }
}
```

**Integration via JPype:**
```python
import jpype
import jpype.imports
jpype.startJVM(
    classpath=["implementations/java/vector_api/dist/portfolio_vector_api.jar"],
    convertStrings=False,
    "--add-modules", "jdk.incubator.vector",
)
from com.portfoliosimulator import PortfolioCompute as _PC
```

JPype starts the JVM once per Python process; HotSpot JIT compiles the hot loop after
~100 iterations. A warmup run (already part of the benchmark protocol) ensures the timed
window measures steady-state throughput.

**Memory transfer:** JPype can pass Java `double[]` arrays backed by Python bytearray or
numpy buffers directly. Arrays are zero-copied via the Java `ByteBuffer` direct buffer API
— no serialisation overhead in the timed window.

**Key hypothesis (H-JV1):** The Vector API with 256-bit lanes (4 doubles per operation)
and `ForkJoinPool` parallelism will achieve 3.5–5× NumPy at N=1M — significantly narrowing
the JVM–native gap compared to pure auto-vectorised Java. HotSpot C2 tiered compilation
overhead should be negligible after warmup.

---

## Expected Performance Ranking (N=1M, hypothesis)

| Engine               | Expected throughput | vs NumPy | Notes |
|----------------------|---------------------|----------|-------|
| julia_loopvec        | 700K–1.1M/s        | ~4–6×    | `@turbo` polyhedral SIMD |
| fortran_openmp       | 750K–900K/s        | ~4–5×    | Same GCC backend as C++ |
| rust_rayon_nightly   | 700K–950K/s        | ~4–5×    | fadd_fast closes gap with C++ |
| cpp_openmp           | 826K/s (measured)  | 4.7×     | Phase 3 baseline |
| numba_parallel       | 920K/s (measured)  | 5.2×     | Phase 3 baseline |
| java_vector_api      | 400K–700K/s        | ~2–4×    | 256-bit SIMD + ForkJoin |
| go_goroutines        | 200K–400K/s        | ~1.5–2.5×| No SIMD; goroutine overhead |
| rust_rayon (stable)  | 465K/s (measured)  | 2.6×     | Phase 3 baseline |
| polars_engine        | 100K–200K/s        | ~0.5–1×  | Expression API overhead |
| duckdb_sql           | 10K–80K/s          | ~0.1–0.5×| Join + aggregation path |
| numpy_vectorised     | 178K/s (measured)  | 1×       | Phase 3 baseline |

These are hypotheses, not predictions — the purpose of the benchmark is to measure.

---

## Integration Architecture

All Phase 3b engines conform to the same interface as Phase 3:

```python
def compute(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Args:
        returns: (T, U) float64, C-order — daily log returns
        weights: (N, U) float64, C-order — portfolio weights (rows sum to ~1)
    Returns:
        out: (N, 2) float64 — [cumulative_return, annualised_sharpe]
    """
```

Native engines (FORTRAN, Rust nightly, Go, Java) are called via ctypes or JPype.
All results are validated against the `result_checksum` from `numpy_vectorised` within
tolerances of ±1e-10 (strict IEEE) or ±1e-6 (fast-math engines).

---

## Build Prerequisites Summary

### Phase 3b

| Engine              | New dependency         | Available check |
|---------------------|------------------------|-----------------|
| polars_engine       | polars                 | `python -c "import polars"` |
| duckdb_sql          | duckdb                 | `python -c "import duckdb"` |
| rust_rayon_nightly  | rustup nightly toolchain | `rustup toolchain list` |
| fortran_openmp      | gfortran               | `gfortran --version` |
| julia_loopvec       | julia binary + juliacall | `julia --version` |
| go_goroutines       | go 1.22+               | `go version` |
| java_vector_api     | java 21+ + jpype1      | `java -version` |

---

## Phase 3c — Float32 and Alternative Runtimes

**Date added:** 2026-02-20
**Status:** Implemented; preliminary N=1K runs complete; full sweep pending

### Motivation

Phase 3b established that the top single-machine engines (Numba, C++, FORTRAN, Julia)
achieve 750–920K portfolios/sec via parallel Welford + SIMD. Phase 3c investigates
three additional questions:

1. **Memory bandwidth vs compute**: Does halving the working set size (float32 vs float64)
   improve throughput at large N where L3 cache is saturated?
2. **ML framework overhead**: Can PyTorch (ATen backend) and JAX (XLA backend) match
   NumPy/OpenBLAS for this specific GEMM + statistics workload?
3. **BLAS-in-Rust**: Does faer's BLAS-backed GEMM approach (similar to Eigen in C++)
   offer a better trade-off than the manual Welford loop in `rust_rayon`?

### Engine Descriptions

#### `numpy_float32`

- **Mechanism**: Cast `(T, U)` returns and `(N, U)` weights to float32 before matmul;
  upcast result to float64 for statistics computation.
- **Key code**: `W = np.ascontiguousarray(weights, dtype=np.float32)`;
  `port_returns = (W @ R.T).astype(np.float64)`
- **Hypothesis**: At N≥100K where L3 is saturated, halving the data footprint should
  improve BLAS throughput. At small N the dtype cast overhead dominates.
- **Checksum behaviour**: Results differ from float64 engines by ~1e-6 (expected;
  documented in `notes` field of each result JSON).
- **Preliminary result**: 49K/s at N=1K (below NumPy float64 57K/s due to cast overhead)

#### `pytorch_cpu`

- **Mechanism**: `torch.from_numpy()` (zero-copy, no data movement) → `W @ R.T` via
  PyTorch's ATen backend → `.numpy()` zero-copy view for statistics.
- **Expected**: Matches `numpy_vectorised` — both call OpenBLAS DGEMM internally.
  Any difference reflects ATen dispatch overhead (typically <5%).
- **Install**: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`

#### `jax_cpu`

- **Mechanism**: Module-level `os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")`;
  `@jax.jit`-compiled kernel with `W @ R.T` + vectorised statistics.
- **Bessel correction**: `jnp.std` defaults to ddof=0; must apply
  `* jnp.sqrt(T / (T - 1))` to match the ddof=1 convention used by all other engines.
- **JIT warmup**: Compilation occurs on first call (~5–30 s); handled by benchmark warmup rep.
- **Expected**: Matches `numpy_vectorised` post-warmup; XLA may show advantages at
  specific matrix shapes via loop fusion.
- **Install**: `uv pip install "jax[cpu]"`

#### `cpp_eigen`

- **Mechanism**: `Eigen::Map<const MatrixXdRowMajor>` zero-copy wrapping of C arrays;
  `W * R.T` via Eigen's BLAS-backed GEMM; `#pragma omp parallel for` for per-row stats.
- **Compile flags**: same as `cpp_openmp`: `-O3 -march=native -ffast-math -funroll-loops`
- **Same ABI**: `extern "C" void compute_portfolio_metrics(r, w, results, n, t, u)`
- **Build**: `sudo apt install libeigen3-dev` + cmake (pending on this machine)
- **Expected**: Comparable to `cpp_openmp` at large N; Eigen's BLAS GEMM vs manual SIMD
  loop is the key comparison.

#### `rust_faer`

- **Mechanism**: `Mat::from_fn(t, u, |i, j| r_slice[i*u+j])` copies C-order array to
  faer's column-major layout; `matmul::matmul(..., Parallelism::Rayon(0))` for GEMM;
  `par_chunks_mut(2)` with Welford for per-portfolio statistics.
- **API note**: faer 0.20 has no `MatRef::from_raw_parts` or
  `from_column_major_slice` — `Mat::from_fn` is the only correct path and involves
  a layout-conversion copy (row-major → column-major).
- **Build**: `cargo build --release --manifest-path implementations/rust/faer/Cargo.toml`
  (stable toolchain; no nightly required)
- **Preliminary result**: 162K/s at N=1K — competitive with `numpy_vectorised` (58K/s)

### Phase 3c Build Prerequisites

| Engine         | New dependency            | Available check |
|----------------|---------------------------|-----------------|
| numpy_float32  | (none — in core deps)     | always available |
| pytorch_cpu    | torch (CPU-only wheel)    | `python -c "import torch"` |
| jax_cpu        | jax[cpu]                  | `python -c "import jax"` |
| cpp_eigen      | libeigen3-dev + cmake     | `dpkg -l libeigen3-dev` |
| rust_faer      | faer = "0.20" (Cargo.toml)| `cargo build --release ...faer/Cargo.toml` |
