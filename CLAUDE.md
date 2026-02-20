# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Python environment
```bash
# All Python commands must use the project venv
source .venv/bin/activate

# Install/update dependencies (uv, not pip)
uv pip install -r requirements.txt
uv pip install -e .                      # Editable install (registers CLI entry points)
uv pip install -e ".[dev,numba]"         # With optional extras

# CLI entry points (after editable install)
ps-fetch       # → scripts/fetch_data.py
ps-generate    # → scripts/generate_portfolios.py
ps-benchmark   # → scripts/run_benchmark.py
```

### Running benchmarks
```bash
# Single run: explicit config
python scripts/run_benchmark.py \
  --scale 1M \
  --storage parquet_wide_uncompressed \
  --engine cpp_openmp \
  --reps 5

# Full phase sweep (all configs in phase)
python scripts/run_benchmark.py --phase 3

# Report only (no benchmark, just summarise existing results)
python scripts/run_benchmark.py --report

# Available scale values: 100, 1K, 100K, 1M, 100M, 1B
# Available engines: numpy_vectorised, numba_parallel, cpp_openmp, rust_rayon, cupy_gpu
# Available storage: csv_per_stock, csv_wide, parquet_wide_uncompressed, parquet_wide_snappy, parquet_wide_zstd
```

### Data pipeline
```bash
python scripts/fetch_data.py                   # Download prices from Yahoo Finance
python scripts/generate_portfolios.py --n 1000000
python scripts/convert_to_parquet.py           # CSV → Parquet variants
python scripts/validate_data.py                # Data quality checks
```

### Building native libraries
```bash
# C++ (required for cpp_openmp engine)
cmake -S implementations/cpp/openmp \
      -B implementations/cpp/openmp/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build implementations/cpp/openmp/build --parallel
# Output: implementations/cpp/openmp/build/libportfolio_openmp.so

# Rust stable (required for rust_rayon engine)
$HOME/.cargo/bin/cargo build --release \
      --manifest-path implementations/rust/rayon/Cargo.toml
# Output: implementations/rust/rayon/target/release/libportfolio_rayon.so

# Rust nightly (required for rust_rayon_nightly engine)
$HOME/.cargo/bin/cargo +nightly build --release \
      --manifest-path implementations/rust/rayon_nightly/Cargo.toml
# Output: implementations/rust/rayon_nightly/target/release/libportfolio_rayon_nightly.so

# FORTRAN (required for fortran_openmp engine — needs gfortran)
# Install: sudo apt-get install gfortran
cmake -S implementations/fortran/openmp \
      -B implementations/fortran/openmp/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build implementations/fortran/openmp/build --parallel
# Output: implementations/fortran/openmp/build/libportfolio_fortran.so

# Go (required for go_goroutines engine)
# Install: sudo apt-get install golang-go
# Or user-local: curl -sL https://go.dev/dl/go1.22.12.linux-amd64.tar.gz | tar -xz -C ~/.local/go-sdk
make -C implementations/go/goroutines build
# Output: implementations/go/goroutines/build/libportfolio_go.so

# Java Vector API (required for java_vector_api engine — needs Java 21+)
make -C implementations/java/vector_api build
# Output: implementations/java/vector_api/dist/portfolio_vector_api.jar

# One-shot Phase 3 setup (installs Numba, builds C++ + Rust stable)
bash scripts/setup_phase3.sh

# One-shot Phase 3b setup (all 7 new engines)
bash scripts/setup_phase3b.sh
```

### Linting
```bash
ruff check src/ scripts/           # ruff is configured in pyproject.toml (line-length=100)
```

### Tests
```bash
pytest                             # testpaths = ["tests"], currently no tests exist
```

## Architecture

### Data flow
```
Yahoo Finance → fetch_data.py → data/raw/prices/ (per-stock CSVs + prices_wide.csv)
                              → data/raw/metadata/ (universe.csv, manifests)

generate_portfolios.py        → data/raw/portfolios/portfolios_<N>.csv
convert_to_parquet.py         → data/parquet/ (snappy / zstd / uncompressed variants)

scripts/run_benchmark.py
  └─ src/benchmark/runner.py (orchestration + telemetry)
       ├─ src/data/storage.py   (unified loader → returns matrix (T×U) + tickers)
       ├─ src/compute/*.py      (engine implementations)
       └─ results/<uuid>.json   (persisted result + hardware/software metadata)

scripts/run_benchmark.py --report
  └─ src/benchmark/report.py   → results/summary.csv, comparison.csv, telemetry.csv
```

### Compute engines (`src/compute/`)
All engines share the same interface: receive a `(T, U)` float64 returns matrix and a `(N, U)` float64/float32 weights matrix, return an `(N, 2)` float64 array of `[cumulative_return, annualised_sharpe]`.

**Phase 1–3 engines (original — do not modify):**

| Module | Engine key | Notes |
|--------|-----------|-------|
| `baseline.py` | `pandas_baseline` | Row-loop Pandas, N ≤ 10K only |
| `baseline.py` | `numpy_vectorised` | BLAS matmul (Phase 1/2 reference) |
| `numba_parallel.py` | `numba_parallel` | `@njit(parallel=True, fastmath=True, cache=True)`, Welford variance |
| `cpp_openmp.py` | `cpp_openmp` | ctypes → `libportfolio_openmp.so` |
| `rust_rayon.py` | `rust_rayon` | ctypes → `libportfolio_rayon.so` (stable Rust) |
| `cupy_gpu.py` | `cupy_gpu` | Optional; skipped gracefully if CUDA unavailable |

**Phase 3b engines (new — 7 additional languages/runtimes):**

| Module | Engine key | Integration | Build/install required |
|--------|-----------|-------------|------------------------|
| `polars_engine.py` | `polars_engine` | Pure Python | `uv pip install polars` |
| `duckdb_sql.py` | `duckdb_sql` | Pure Python | `uv pip install duckdb` |
| `rust_rayon_nightly.py` | `rust_rayon_nightly` | ctypes → `libportfolio_rayon_nightly.so` | `cargo +nightly build --release` |
| `fortran_openmp.py` | `fortran_openmp` | ctypes → `libportfolio_fortran.so` | cmake + gfortran |
| `julia_loopvec.py` | `julia_loopvec` | juliacall (in-process) | Julia + `uv pip install juliacall` |
| `go_goroutines.py` | `go_goroutines` | ctypes → `libportfolio_go.so` | `go build -buildmode=c-shared` |
| `java_vector_api.py` | `java_vector_api` | JPype (in-process JVM) | `make -C implementations/java/vector_api build` |

The Numba kernel warms up on first call (~30 s); `cache=True` skips on subsequent runs. Julia and Java JVMs start once per process — the existing warmup run handles JIT compilation.

### Benchmark runner (`src/benchmark/runner.py`)
- `run_benchmark(config)` is the main entry point. It runs `reps` repetitions, records per-rep telemetry (CPU%, I/O MB, RSS), and writes a result JSON to `results/<uuid>.json`.
- `TelemetryCollector` samples system metrics every 25 ms in a background thread.
- Result JSON includes full hardware + software fingerprint (CPU, RAM, GPU, Python/NumPy/BLAS versions, git SHA) for reproducibility.
- Checksums (`result_checksum`) are SHA-256 of sorted results, computed within (implementation, storage, scale, seed) for cross-engine numerical validation.

### Storage layer (`src/data/storage.py`)
Single function `load_returns(format, ...)` returns `(returns_matrix, tickers)`. All format variants are handled here. Parquet loaders use PyArrow internally; per-stock loaders open 100 individual CSV files (which is the reason for 5–6× slower load vs. wide format at small N).

### Portfolio generator (`src/portfolio/generator.py`)
Reproducible seeding: portfolio `i` uses seed `(42 XOR i) & 0xFFFFFFFF`. K=15 stocks drawn without replacement from U=100, weights from Dirichlet(α=1,...,1). Produces dense `(N, U)` matrix (zeros for non-selected stocks).

### Native library notes
- **C++ (`-ffast-math` is critical):** Without it, GCC treats the FP reduction as non-associative and emits scalar code. The `#pragma omp simd reduction(+:port_r)` directive on the inner loop plus `-ffast-math -funroll-loops` unlocks AVX2 FMA. All pointers are `__restrict__`.
- **Rust (8-wide accumulators):** Rust stable has no `-ffast-math` equivalent. The workaround is 8 independent accumulator variables per inner loop so LLVM can pack them into two ymm registers. `.cargo/config.toml` sets `target-cpu = native` to enable AVX2/FMA at the instruction level.

## Key conventions

- **Package manager:** `uv` only (not pip/pip3).
- **Engine name alias:** `"numpy_vectorised"` and `"numpy_matmul"` refer to the same engine in `runner.py`.
- **Scale shorthands:** CLI accepts `100`, `1K`, `100K`, `1M`, `100M`, `1B`; internally stored as integers.
- **N > 10M:** Portfolio weights are generated on-the-fly (seeded, not materialised on disk).
- **Metrics definition:** cumulative return = `expm1(Σ log_returns @ weights)`; Sharpe = `mean(daily_port_ret) / std(ddof=1) × sqrt(252)` with Rf=0.
- **Results schema:** `common/schemas/benchmark_result.schema.json` is the authoritative spec for the result JSON structure.
- **Committed vs. generated:** Raw price/portfolio CSVs and Parquet files are **not committed** (generated locally). Only `data/raw/metadata/` (universe.csv, manifests) and `results/summary.csv` are committed.
