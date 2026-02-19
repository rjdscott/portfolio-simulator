# Portfolio Return Simulator — Computational Benchmark Study

A reproducible benchmark suite measuring the computational intensity of computing
portfolio returns at massive scale: from 100 to 1,000,000,000 portfolios.

This research explores how storage format, compute engine, and system architecture
affect throughput, and provides a principled methodology for choosing the right tool
at each scale.

---

## Research Question

> What is the minimum time-to-result for computing annualised Sharpe ratios and
> cumulative returns across N portfolios drawn from the S&P 500, and how does the
> answer change as N grows from 100 to 10⁹?

---

## Hardware Environment

| Component       | Specification                                      |
|-----------------|----------------------------------------------------|
| CPU             | Intel Core i7-14700K (20 cores / 28 threads)       |
| RAM             | 64 GB                                              |
| Storage         | Local NVMe SSD                                     |
| OS              | Ubuntu 24.04 LTS (Linux 6.17)                      |
| BLAS            | OpenBLAS (scipy-openblas, DYNAMIC\_ARCH Haswell)   |

---

## Software Environment

### Python stack

| Package    | Version  | Role                                      |
|------------|----------|-------------------------------------------|
| Python     | 3.11.14  | Runtime                                   |
| uv         | 0.9.25   | Package and venv manager                  |
| numpy      | 2.4.2    | Baseline vectorised compute (OpenBLAS)    |
| pandas     | 3.0.1    | Data loading and row-loop baseline        |
| pyarrow    | 23.0.1   | Parquet I/O                               |
| numba      | 0.64.0   | JIT compiler (Phase 3)                    |
| llvmlite   | 0.46.0   | Numba LLVM backend                        |
| psutil     | —        | Telemetry (CPU %, RSS, I/O counters)      |
| cupy       | optional | GPU compute via CUDA (Phase 3)            |

### Native toolchain

| Tool        | Version  | Role                                      |
|-------------|----------|-------------------------------------------|
| GCC / g++   | 13.3.0   | C++ compiler                              |
| CMake       | 3.28.3   | Cross-platform C++ build system           |
| OpenMP      | 4.5      | C++ thread parallelism                    |
| Rust        | 1.93.1   | Rust compiler (via rustup)                |
| Rayon       | 1.11.0   | Rust data-parallelism library             |

---

## Data

- **Universe**: Top 100 S&P 500 stocks by market cap with ≥95% data completeness
- **Price history**: 2020-01-01 → 2024-12-31 (~1,257 trading days) via `yfinance`
- **Portfolio**: K=15 stocks, Dirichlet weights, global seed=42
- **Portfolio seed**: `(42 XOR portfolio_id) & 0xFFFFFFFF`

### Materialised portfolio files (on disk)

| N          | File size |
|------------|-----------|
| 100        | 64 KB     |
| 1,000      | 635 KB    |
| 100,000    | 62 MB     |
| 1,000,000  | 622 MB    |
| >10M       | Seeded generation only (not materialised) |

---

## Benchmark Phases

### Phase 1 — Compute baseline
Storage fixed at CSV. Compare Pandas row-loop vs NumPy/OpenBLAS matmul.

**Results** (warm cache):

| Implementation   | Storage       | N    | Throughput  |
|------------------|---------------|------|-------------|
| pandas_baseline  | csv_per_stock | 100  | 1,733/sec   |
| pandas_baseline  | csv_per_stock | 1K   | 13,189/sec  |
| numpy_vectorised | csv_per_stock | 100K | 156,874/sec |
| numpy_vectorised | csv_per_stock | 1M   | 174,721/sec |
| numpy_vectorised | csv_wide      | 1M   | 177,504/sec |

**Findings**:
- Storage format matters more than compute engine at small N: opening 100 individual
  CSV files is ~5–6× slower than reading one wide CSV at N=100, regardless of engine
- Storage impact disappears above N≈100K — compute dominates
- NumPy matmul is ~8× faster than Pandas row-loop at scale

---

### Phase 2 — Storage optimisation
Compute fixed at `numpy_vectorised`. Compare CSV vs Parquet variants.

**Results** (warm cache, N=100 shown; all formats converge at N≥100K):

| Storage                   | N=100 Throughput | vs baseline | Load time |
|---------------------------|-----------------|-------------|-----------|
| parquet_wide_uncompressed | 20,194/sec      | **14.1×**   | 4 ms      |
| parquet_wide_snappy       | 9,737/sec       | 6.8×        | 8 ms      |
| csv_wide                  | 7,733/sec       | 5.4×        | 12 ms     |
| parquet_wide_zstd         | 5,480/sec       | 3.8×        | 12 ms     |
| parquet_per_stock         | 1,390/sec       | ≈ baseline  | 71 ms     |
| all wide formats          | ~176K/sec       | converged   | 4–9 ms    |

**Findings**:
- Compression is a **cost** not a benefit on local NVMe for files <2 MB: uncompressed
  Parquet loads 3× faster than zstd because decompression dominates at that file size
- `parquet_wide_uncompressed` is the clear winner at small N (14.1× over baseline)
  and ties with all other wide formats at large N — making it optimal across all scales
- Per-stock file layout (100 individual files) causes the same ~70 ms I/O penalty
  whether reading CSV or Parquet — the file-open overhead, not format parsing, is the cost

---

### Phase 3 — Compute engine comparison
Storage fixed at `parquet_wide_uncompressed`. Vary the compute engine.

**Engines** (warm cache, 5 reps each, 2026-02-20):

| Engine           | Language | Parallelism                           |
|------------------|----------|---------------------------------------|
| numpy_vectorised | Python   | OpenBLAS auto-threaded DGEMM          |
| numba_parallel   | Python   | Numba `prange` (all cores), fastmath  |
| cpp_openmp       | C++      | OpenMP `#pragma omp parallel for`     |
| rust_rayon       | Rust     | Rayon `par_chunks_mut`, 8-wide accum. |
| cupy_gpu         | Python   | CUDA (optional — skipped, no GPU)     |

**Results** (portfolios/sec):

| Engine           | N=100  | N=1K    | N=100K    | N=1M      | vs NumPy (1M) |
|------------------|--------|---------|-----------|-----------|---------------|
| numba_parallel   | 25,641 | 193,125 | 846,224   | **919,968** | **5.2×**    |
| cpp_openmp       | 28,944 | 190,404 | **936,900** | 826,211 | **4.7×**    |
| rust_rayon       | 23,838 | 115,674 | 494,044   | 465,250   | **2.6×**    |
| numpy_vectorised | 20,194 | 57,887  | 170,007   | 177,572   | 1×            |

**Findings**:

1. **The "BLAS holds at large N" hypothesis is disproved.** Every Phase 3 engine
   beats NumPy/OpenBLAS at every scale from N=100 to N=1M. BLAS DGEMM allocates
   a full (N, T) output matrix and is memory-bandwidth bound; per-portfolio Welford
   loops hold weights in L1 and read the shared 1 MB returns matrix from L3,
   achieving better arithmetic intensity.

2. **Numba and C++ are neck-and-neck (~830–940K/s at large N).** C++ leads at
   N=100K (937K vs 846K) due to lower JIT dispatch overhead; Numba pulls ahead at
   N=1M (920K vs 826K) via `fastmath=True` LLVM loop fusion. Both run at 100% CPU
   on all 28 threads.

3. **C++ requires `-ffast-math` to be competitive.** Without it, GCC treats
   floating-point reduction as non-associative (strict IEEE 754) and the inner
   dot-product loop `Σ w[u] × r[u]` over U=100 stocks runs entirely scalar — making
   C++ *slower* than NumPy. Adding `-ffast-math`, `__restrict__`, and
   `#pragma omp simd reduction(+:port_r)` unlocks AVX2/FMA and gives the 4.7× gain.

4. **Rust stable Rayon achieves 2.6× NumPy** but trails C++/Numba by ~1.9×.
   Stable Rust has no `-ffast-math` equivalent at the language level. The workaround
   — 8 independent accumulators — allows LLVM to pack them into two AVX2 `ymm`
   registers and emit `vfmadd231pd`, but GCC with `-ffast-math` applies broader
   reassociation across the full reduction. Closing the gap would require nightly
   Rust (`std::intrinsics::fadd_fast`) or an explicit SIMD crate (`wide`, `std::simd`).

All engines use **Welford's online algorithm** for variance (ddof=1, numerically
stable, single pass) to ensure cross-implementation numerical agreement. Float
differences at ~1e-14 are expected due to different FP accumulation order and are
documented in each result's `result_checksum` field.

---

### Phase 4 — Distributed scale *(planned)*
PySpark local cluster and/or Dask for N≥10M portfolios. Seeded on-the-fly portfolio
generation replaces materialised weight files (>10M rows is impractical to store).

---

## Summary of Findings Across All Phases

| Insight | Detail |
|---------|--------|
| Storage format dominates at small N | Per-stock CSVs vs wide Parquet: **14× difference** at N=100 |
| All storage formats converge at large N | I/O is fixed cost; compute determines throughput above N≈100K |
| Compression hurts on NVMe for small files | Uncompressed Parquet 3× faster to load than zstd at <2 MB |
| BLAS is not optimal for this workload | Portfolio-parallel Welford beats DGEMM 5× at N=1M |
| C++ fast-math is non-negotiable | Scalar fallback without `-ffast-math` is slower than Python/NumPy |
| Numba is a practical near-C++ path | 5.2× NumPy with pure Python syntax; `cache=True` amortises JIT cost |
| Rust stable has a vectorisation gap | 2.6× NumPy; `fast-math` equivalent only available on nightly |

---

## Next Steps

### Phase 4 — Seeded generation + distributed compute (N=10M to 1B)
- Implement seeded on-the-fly portfolio generation (no materialisation above 10M)
- PySpark local cluster: partition N portfolios across cores, broadcast returns matrix
- Dask delayed graph: compare scheduling overhead vs Spark at intermediate N
- Target scales: 10M, 100M, 1B portfolios

### Phase 5 — GPU at scale (N≥1M)
- CuPy on CUDA GPU: PCIe transfer overhead expected to amortise above N≈500K
- Compare GPU matmul (CuPy `W @ R.T`) vs CPU-parallel Welford at N=1M, 10M
- Explore mixed strategy: GPU for compute, CPU threads for I/O prefetch

### Phase 6 — Cold-cache benchmarks
- Requires `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` between runs
- Quantify NVMe read latency contribution separately from compute
- Expected to significantly widen the per-stock vs wide-format gap at small N

### Improvements to existing phases
- **Rust nightly**: enable `std::intrinsics::fadd_fast` to close the ~1.9× gap vs C++
- **Kotlin/JVM**: add JVM baseline using `DoubleArray` and coroutines for Phase 3
- **BLAS tuning**: benchmark `OMP_NUM_THREADS` sweep to find OpenBLAS saturation point
- **Cold vs warm cache comparison**: quantify the page-cache warm-up effect for each
  storage format

---

## Quick Start

### Prerequisites

```bash
# Python venv (uv required)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate

# Core Python dependencies
uv pip install numpy pandas pyarrow psutil
```

### Phase 1 & 2

```bash
# Fetch price data
python scripts/fetch_data.py

# Generate portfolio weight files
python scripts/generate_portfolios.py --n 1_000_000

# Convert CSVs to Parquet (Phase 2)
python scripts/convert_to_parquet.py

# Run Phase 1 (CSV baselines)
python scripts/run_benchmark.py --phase 1

# Run Phase 2 (storage comparison)
python scripts/run_benchmark.py --phase 2

# Print aggregated report
python scripts/run_benchmark.py --report
```

### Phase 3 — native engine setup

```bash
# Install all Phase 3 deps and build native libraries (one command)
bash scripts/setup_phase3.sh

# Smoke tests (individual engines)
python scripts/run_benchmark.py --engine numba_parallel --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache
python scripts/run_benchmark.py --engine cpp_openmp     --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache
python scripts/run_benchmark.py --engine rust_rayon     --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache

# Full Phase 3 suite (16 configurations)
python scripts/run_benchmark.py --phase 3
```

**Manual build steps** (if `setup_phase3.sh` is not used):

```bash
# Numba
uv pip install "numba>=0.59.0"

# C++ OpenMP (requires cmake ≥3.16 and g++ with OpenMP 4.5)
cmake -S implementations/cpp/openmp -B implementations/cpp/openmp/build -DCMAKE_BUILD_TYPE=Release
cmake --build implementations/cpp/openmp/build --parallel

# Rust/Rayon (requires rustup / cargo)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
cd implementations/rust/rayon && cargo build --release

# CuPy (optional, requires CUDA 12.x GPU)
uv pip install cupy-cuda12x
```

---

## Repository Structure

```
portfolio-simulator/
├── data/
│   ├── raw/
│   │   ├── prices/              # Per-stock CSVs + prices_wide.csv
│   │   ├── metadata/            # universe.csv, data_manifest.json
│   │   └── portfolios/          # portfolios_<N>.csv weight files
│   └── parquet/                 # Parquet conversions (Phase 2)
├── src/
│   ├── data/
│   │   ├── fetch.py             # yfinance data acquisition
│   │   ├── validate.py          # Data quality checks
│   │   └── storage.py           # Unified format loader + Parquet conversion
│   ├── portfolio/
│   │   └── generator.py         # Seeded portfolio generator
│   ├── compute/
│   │   ├── baseline.py          # Pandas row-loop + NumPy matmul
│   │   ├── numba_parallel.py    # Numba JIT kernel (Phase 3)
│   │   └── cupy_gpu.py          # CuPy GPU engine (Phase 3)
│   └── benchmark/
│       ├── runner.py            # Benchmark orchestration + telemetry
│       └── report.py            # Results aggregation and reporting
├── implementations/
│   ├── cpp/
│   │   └── openmp/
│   │       ├── CMakeLists.txt            # CMake build (requires cmake ≥3.16)
│   │       ├── src/portfolio_compute.cpp # OpenMP + AVX2 kernel (-ffast-math)
│   │       ├── build/                    # Build output (.so) — gitignored
│   │       └── python/portfolio_openmp.py  # ctypes wrapper
│   └── rust/
│       └── rayon/
│           ├── Cargo.toml                # cdylib crate, rayon = "1.10" (resolves 1.11.0)
│           ├── .cargo/config.toml        # target-cpu=native (unlocks AVX2)
│           ├── src/lib.rs                # Rayon + 8-wide accumulator kernel
│           ├── target/                   # Cargo output — gitignored
│           └── python/portfolio_rayon.py # ctypes wrapper
├── scripts/
│   ├── fetch_data.py            # CLI: download price data
│   ├── generate_portfolios.py   # CLI: generate portfolio weight files
│   ├── convert_to_parquet.py    # CLI: CSV → Parquet conversion
│   ├── run_benchmark.py         # CLI: run benchmark configurations
│   └── setup_phase3.sh          # One-shot Phase 3 environment setup
├── results/                     # Benchmark outputs (JSON per run, summary.csv)
├── common/
│   └── schemas/
│       └── benchmark_result.schema.json
└── pyproject.toml
```

---

## Numerical Correctness

All implementations compute the same two metrics per portfolio:

- **Cumulative return**: `expm1(Σ log_returns @ weights)` — numerically stable
- **Annualised Sharpe**: `mean(port_returns) / std(port_returns, ddof=1) × √252`

All Phase 3 engines use **Welford's online algorithm** for variance to ensure
agreement with NumPy's `ddof=1`. The `result_checksum` field in each result JSON
(SHA-256 of the sorted results array) allows cross-implementation validation. Float
differences at ~1e-14 are expected due to different FP operation ordering.

---

## Reproducibility

- All randomness is seeded (`global_seed=42`, `portfolio_seed = (42 XOR id) & 0xFFFFFFFF`)
- `data/raw/metadata/` is committed (universe list, data manifest, validation report)
- `results/summary.csv` is committed; individual run JSONs are gitignored
- `implementations/rust/rayon/Cargo.lock` is committed (pins exact dependency versions)
- Build artefacts (`build/`, `target/`) are gitignored

---

## Contributing / Citation

If you use this benchmark in your research, please cite:

```
[Citation to be added upon publication]
```

See `RESEARCH.md` for the living research journal.
