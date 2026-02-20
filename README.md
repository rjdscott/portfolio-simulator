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
| duckdb     | 1.4.4    | SQL compute engine + result registry      |
| polars     | 1.38.1   | DataFrame engine (Rust/Rayon backend)     |
| psutil     | —        | Telemetry (CPU %, RSS, I/O counters)      |
| cupy       | optional | GPU compute via CUDA (Phase 3)            |
| torch      | optional | CPU matmul via PyTorch ATen               |
| jax        | optional | XLA-compiled CPU kernels                  |

### Native toolchain

| Tool        | Version  | Role                                      |
|-------------|----------|-------------------------------------------|
| GCC / g++   | 13.3.0   | C++ and FORTRAN compiler                  |
| CMake       | 3.28.3   | Cross-platform C++ / FORTRAN build system |
| OpenMP      | 4.5      | Thread parallelism (C++, FORTRAN)         |
| Rust        | 1.93.1   | Rust compiler (stable, via rustup)        |
| Rayon       | 1.11.0   | Rust data-parallelism library             |
| faer        | 0.20.2   | Rust BLAS-backed dense linear algebra     |
| Eigen3      | 3.4.0    | C++ header-only linear algebra (optional) |

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

**Core engines** (warm cache, 5 reps each, 2026-02-20):

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
   beats NumPy/OpenBLAS at every scale from N=100 to N=1M.

2. **Numba and C++ are neck-and-neck (~830–940K/s at large N).** C++ leads at
   N=100K (937K vs 846K) due to lower JIT dispatch overhead; Numba pulls ahead at
   N=1M (920K vs 826K) via `fastmath=True` LLVM loop fusion.

3. **C++ requires `-ffast-math` to be competitive.** Without it, GCC emits entirely
   scalar code for the inner dot-product loop. Adding `-ffast-math`, `__restrict__`,
   and `#pragma omp simd reduction(+:port_r)` unlocks AVX2/FMA.

4. **Rust stable Rayon achieves 2.6× NumPy** but trails C++/Numba by ~1.9×. Stable
   Rust has no `-ffast-math` equivalent at the language level.

---

### Phase 3b — Extended engine survey
Eleven additional compute engines spanning 7 languages / runtimes, all using
`parquet_wide_uncompressed` storage at scales 100 / 1K / 100K / 1M.

**Results** (portfolios/sec, warm cache, 5 reps):

| Engine              | N=100  | N=1K    | N=100K  | N=1M    | vs NumPy (1M) |
|---------------------|--------|---------|---------|---------|---------------|
| numba_parallel      | 25,641 | 193,125 | 846,224 | 919,968 | **5.2×**      |
| fortran_openmp      | 20,517 | 160,694 | 779,029 | 828,035 | **4.7×**      |
| cpp_openmp          | 28,944 | 190,404 | 936,900 | 826,211 | **4.7×**      |
| julia_loopvec       | 23,250 | 199,681 | 707,409 | 745,475 | **4.2×**      |
| rust_rayon_nightly  | 10,045 | 79,994  | 655,304 | 654,643 | **3.7×**      |
| rust_faer           | —      | 162,549 | —       | —       | —†            |
| java_vector_api     | 19,585 | 117,481 | 460,764 | 476,253 | **2.7×**      |
| rust_rayon          | 23,838 | 115,674 | 494,044 | 465,250 | **2.6×**      |
| go_goroutines       | 20,773 | 118,287 | 323,932 | 344,630 | **1.9×**      |
| numpy_float32       | —      | 49,408  | —       | —       | —†            |
| numpy_vectorised    | 20,194 | 57,887  | 170,007 | 177,572 | 1×            |
| polars_engine       | 1,943  | 22,219  | 105,927 | 83,811  | 0.47×         |
| duckdb_sql          | 884    | 1,119   | 11,360  | —‡      | —             |

† `rust_faer` and `numpy_float32` have been benchmarked at N=1K only; full sweep pending.
‡ DuckDB N=1M: the data-melt step (100M rows) is prohibitively slow; a native Parquet
pipeline (bypassing PyArrow) would avoid this overhead.

**Key findings**:
- **FORTRAN ties C++ OpenMP** (828K vs 826K at N=1M, 0.2% difference) — gfortran
  `-ffast-math -march=native` through the GCC backend produces identical AVX2 code
- **Julia matches C++ OpenMP at 1M** (745K) and actually leads at N=1K (200K vs 190K)
- **Rust nightly `fadd_fast` closes 41% of the Rust stable gap** (654K vs 465K)
- **Java Vector API (476K) is competitive with Rust stable (465K)** — HotSpot C2 JIT
  compiles `DoubleVector.SPECIES_256` to AVX2 after warmup
- **faer (Rust) achieves 162K/s at N=1K** in early testing — full sweep pending

---

### Phase 4 — Distributed scale
PySpark, Dask, and Ray for N≥1M portfolios. Seeded on-the-fly portfolio generation
replaces materialised weight files above N=10M.

---

## Result Registry (DuckDB)

All benchmark runs are persisted as individual JSON files in `results/<uuid>.json`
(immutable source of truth). A DuckDB registry (`results/registry.duckdb`) provides
a queryable index built incrementally from those JSONs.

```bash
# Ingest new JSONs and export publish-ready Parquet + CSV
python scripts/run_benchmark.py --export

# Or as part of the report command
python scripts/run_benchmark.py --report
```

Exports written to `results/exports/`:

| File | Description |
|------|-------------|
| `summary.parquet` | One row per run, all metadata + metrics |
| `telemetry.parquet` | Per-rep I/O, CPU%, RSS for every run |
| `summary.csv` | Same as summary.parquet in CSV format |
| `telemetry.csv` | Same as telemetry.parquet in CSV format |
| `comparison.csv` | Throughput pivot: (engine, storage) × scale |

**Deduplication**: Re-running the same configuration (same engine, storage, scale,
seed, CPU) marks the previous row `superseded=TRUE`; the `v_canonical` view always
shows only the latest non-superseded row per fingerprint.

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
| FORTRAN ≈ C++ at peak throughput | Same GCC backend + `-ffast-math` → identical AVX2 code |
| Julia ≈ C++ via `@turbo` + threads | 4.2× NumPy; wins at small N; no manual SIMD required |
| Rust nightly closes the fast-math gap | +41% over stable; `fadd_fast` unlocks `vfmadd231pd` reduction |
| JVM can do SIMD (with Vector API) | Java 476K/s (2.7×); comparable to Rust stable |

---

## Next Steps

### Phase 3 — Remaining engines
- Full sweep (all 4 scales) for `rust_faer`, `numpy_float32`, `pytorch_cpu`, `jax_cpu`, `cpp_eigen`
- `cpp_eigen` build requires `sudo apt install libeigen3-dev`

### Phase 4 — Distributed compute (N=1M to 1B)
- PySpark local cluster: partition N portfolios across cores, broadcast returns matrix
- Dask delayed graph: compare scheduling overhead vs Spark at intermediate N
- Ray actors: fine-grained task dispatch model
- Target scales: 1M, 10M, 100M, 1B portfolios

### Phase 5 — GPU at scale (N≥1M)
- CuPy on CUDA GPU: PCIe transfer overhead expected to amortise above N≈500K
- Mixed strategy: GPU for compute, CPU threads for I/O prefetch

### Phase 6 — Cold-cache benchmarks
- Requires `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` between runs
- Quantify NVMe read latency separately from compute

---

## Quick Start

### Prerequisites

```bash
# Python venv (uv required)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate

# Core Python dependencies
uv pip install numpy pandas pyarrow psutil duckdb polars
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

# Print aggregated report and update result registry
python scripts/run_benchmark.py --report

# Export publish-ready Parquet (results/exports/)
python scripts/run_benchmark.py --export
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

# Full Phase 3b suite (all 11 additional engines × 4 scales)
python scripts/run_benchmark.py --phase 3b
```

**Manual build steps** (if `setup_phase3.sh` is not used):

```bash
# Numba
uv pip install "numba>=0.59.0"

# C++ OpenMP
cmake -S implementations/cpp/openmp -B implementations/cpp/openmp/build -DCMAKE_BUILD_TYPE=Release
cmake --build implementations/cpp/openmp/build --parallel

# C++ Eigen (requires libeigen3-dev)
sudo apt install libeigen3-dev
cmake -S implementations/cpp/eigen -B implementations/cpp/eigen/build -DCMAKE_BUILD_TYPE=Release
cmake --build implementations/cpp/eigen/build --parallel

# Rust stable (rust_rayon engine)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
cargo build --release --manifest-path implementations/rust/rayon/Cargo.toml

# Rust faer (rust_faer engine)
cargo build --release --manifest-path implementations/rust/faer/Cargo.toml

# Rust nightly (rust_rayon_nightly engine)
rustup toolchain install nightly
cargo +nightly build --release --manifest-path implementations/rust/rayon_nightly/Cargo.toml

# PyTorch CPU
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# JAX CPU
uv pip install "jax[cpu]"

# CuPy (optional, requires CUDA 12.x)
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
│       └── prices_wide.arrow    # Arrow IPC zero-copy file (Phase 2.5)
├── src/
│   ├── data/
│   │   ├── fetch.py             # yfinance data acquisition
│   │   ├── validate.py          # Data quality checks
│   │   └── storage.py           # Unified format loader + Parquet/Arrow conversion
│   ├── portfolio/
│   │   └── generator.py         # Seeded portfolio generator
│   ├── compute/
│   │   ├── baseline.py          # Pandas row-loop + NumPy matmul (Phase 1/2)
│   │   ├── numba_parallel.py    # Numba JIT + prange, fastmath=True (Phase 3)
│   │   ├── cpp_openmp.py        # ctypes → libportfolio_openmp.so (Phase 3)
│   │   ├── cpp_eigen.py         # ctypes → libportfolio_eigen.so (Phase 3c)
│   │   ├── rust_rayon.py        # ctypes → libportfolio_rayon.so (Phase 3)
│   │   ├── rust_rayon_nightly.py # ctypes → libportfolio_rayon_nightly.so (Phase 3b)
│   │   ├── rust_faer.py         # ctypes → libportfolio_faer.so (Phase 3c)
│   │   ├── cupy_gpu.py          # CuPy GPU engine, graceful skip if no CUDA (Phase 3)
│   │   ├── numpy_float32.py     # float32 matmul — memory-bandwidth experiment (Phase 3c)
│   │   ├── pytorch_cpu.py       # PyTorch ATen matmul on CPU (Phase 3c)
│   │   ├── jax_cpu.py           # JAX JIT-compiled kernel on CPU (Phase 3c)
│   │   ├── polars_engine.py     # Polars expression API (Phase 3b)
│   │   ├── duckdb_sql.py        # DuckDB SQL aggregation (Phase 3b)
│   │   ├── fortran_openmp.py    # ctypes → libportfolio_fortran.so (Phase 3b)
│   │   ├── julia_loopvec.py     # juliacall in-process Julia runtime (Phase 3b)
│   │   ├── go_goroutines.py     # ctypes → libportfolio_go.so (Phase 3b)
│   │   ├── java_vector_api.py   # JPype in-process JVM, Vector API (Phase 3b)
│   │   ├── spark_local.py       # PySpark local cluster (Phase 4)
│   │   ├── dask_local.py        # Dask local scheduler (Phase 4)
│   │   └── ray_local.py         # Ray local cluster (Phase 4)
│   └── benchmark/
│       ├── runner.py            # Benchmark orchestration + telemetry
│       ├── report.py            # Results aggregation, print + CSV export
│       └── db.py                # DuckDB result registry (ingest, dedupe, Parquet export)
├── implementations/
│   ├── cpp/
│   │   ├── openmp/
│   │   │   ├── CMakeLists.txt           # CMake build (-ffast-math, OpenMP)
│   │   │   ├── src/portfolio_compute.cpp # OpenMP + AVX2 kernel
│   │   │   ├── build/                   # Build output (libportfolio_openmp.so)
│   │   │   └── python/portfolio_openmp.py
│   │   └── eigen/
│   │       ├── CMakeLists.txt           # CMake build (Eigen3 + OpenMP)
│   │       ├── src/portfolio_compute.cpp # Eigen::Map + OpenMP
│   │       ├── build/                   # Build output (libportfolio_eigen.so)
│   │       └── python/portfolio_eigen.py
│   ├── rust/
│   │   ├── rayon/
│   │   │   ├── Cargo.toml               # cdylib, rayon = "1.10"
│   │   │   ├── .cargo/config.toml       # target-cpu=native (AVX2)
│   │   │   ├── src/lib.rs               # 8-wide accumulator kernel
│   │   │   ├── target/                  # Cargo output (libportfolio_rayon.so)
│   │   │   └── python/portfolio_rayon.py
│   │   ├── rayon_nightly/
│   │   │   ├── Cargo.toml               # cdylib, nightly, fadd_fast
│   │   │   ├── .cargo/config.toml       # target-cpu=native
│   │   │   ├── src/lib.rs               # fadd_fast intrinsic kernel
│   │   │   └── target/                  # (libportfolio_rayon_nightly.so)
│   │   └── faer/
│   │       ├── Cargo.toml               # cdylib, faer = "0.20", rayon
│   │       ├── .cargo/config.toml       # target-cpu=native
│   │       ├── src/lib.rs               # faer GEMM + Rayon Welford
│   │       └── target/                  # (libportfolio_faer.so)
│   ├── fortran/
│   │   └── openmp/                      # FORTRAN ISO_C_BINDING + OpenMP
│   ├── go/
│   │   └── goroutines/                  # Go goroutine worker pool
│   └── java/
│       └── vector_api/                  # Java 21 Vector API + ForkJoinPool
├── scripts/
│   ├── fetch_data.py            # CLI: download price data
│   ├── generate_portfolios.py   # CLI: generate portfolio weight files
│   ├── convert_to_parquet.py    # CLI: CSV → Parquet + Arrow IPC conversion
│   ├── run_benchmark.py         # CLI: run benchmark configurations
│   ├── setup_phase3.sh          # One-shot Phase 3 environment setup
│   └── setup_phase3b.sh         # One-shot Phase 3b environment setup
├── results/
│   ├── *.json                   # Individual run results (immutable source of truth)
│   ├── registry.duckdb          # Queryable index of all run JSONs
│   ├── summary.csv              # Flat table of all runs (backward-compat)
│   ├── comparison.csv           # Throughput pivot (backward-compat)
│   └── exports/
│       ├── summary.parquet      # Publish-ready canonical dataset
│       ├── telemetry.parquet    # Per-rep telemetry
│       ├── summary.csv          # CSV version of summary
│       └── comparison.csv       # Throughput pivot in CSV
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

The `numpy_float32` engine uses float32 for the matmul step (halving the L3 cache
footprint) and upcasts to float64 before computing statistics; its checksum will
differ from float64 engines by ~1e-6 and is documented in the result notes.

---

## Reproducibility

- All randomness is seeded (`global_seed=42`, `portfolio_seed = (42 XOR id) & 0xFFFFFFFF`)
- `data/raw/metadata/` is committed (universe list, data manifest, validation report)
- `results/summary.csv` and `results/exports/` are committed; individual run JSONs are gitignored
- `implementations/rust/*/Cargo.lock` files are committed (pin exact dependency versions)
- Build artefacts (`build/`, `target/`) are gitignored

---

## Contributing / Citation

If you use this benchmark in your research, please cite:

```
[Citation to be added upon publication]
```

See `RESEARCH.md` for the living research journal and `RESULTS.md` for detailed results.
