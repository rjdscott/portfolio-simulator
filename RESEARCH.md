# Research Journal — Portfolio Return Simulator Benchmark

> This is a living document. Each entry records decisions made, hypotheses formed,
> experiments run, and conclusions reached. It is intended to accompany the eventual
> publication and provide full transparency into the research process.

---

## Project Charter

**Research goal**: Determine empirically how storage format and compute engine affect
the throughput of computing portfolio returns at scales from 100 to 10⁹ portfolios,
using real market data and reproducible methodology.

**Publication target**: Financial Data Engineering Forums / applied quantitative
research journals. Secondary: GitHub open-source release.

**Lead engineer**: Staff Data Engineer
**Date initiated**: 2026-02-19

---

## Hypotheses

| ID  | Hypothesis | Expected finding |
|-----|-----------|-----------------|
| H1  | CSV-per-stock is the worst storage format for cross-sectional operations | Yes — requires N file opens and row-wise parsing |
| H2  | A vectorised NumPy matmul (weight matrix × returns matrix) will outperform a Pandas loop by ≥10× at 1M scale | Yes — avoids Python overhead entirely |
| H3  | Parquet columnar storage will outperform CSV by ≥5× for the price loading step | To be confirmed |
| H4  | GPU (CuPy) will saturate at small N due to kernel launch overhead but dominate at N ≥ 100K | To be confirmed |
| H5  | For N ≥ 100M, no single-machine solution will maintain sub-minute latency; Spark required | To be confirmed |
| H6  | At N = 1B, memory bandwidth is the binding constraint, not FLOPS | To be confirmed |

---

## Experimental Design

### Unit of measurement
Wall-clock time (seconds) from first byte read to last result written.
We also record:
- Peak RSS memory (MB)
- CPU utilisation (%)
- GPU utilisation (%) where applicable
- I/O bytes read / written

### Statistical validity
Each benchmark configuration is run **5 times** with a 10-second warm-up (cold
cache dropped via `sync && echo 3 > /proc/sys/vm/drop_caches` between cold runs).
We report median, p10, and p90 to characterise variance.

### Controls
- Same random seed across all experiments (seed = 42)
- Same stock universe (committed constituent list)
- Same date range (2020-01-01 → 2024-12-31)
- Processes pinned to cores where possible (avoid NUMA effects)

---

## Data Provenance

| Dataset | Source | Fetch date | Notes |
|---------|--------|-----------|-------|
| S&P 500 constituents | Wikipedia (via pandas) | 2026-02-19 | Snapshot committed to repo |
| Daily adjusted-close prices | Yahoo Finance (yfinance) | TBD | Adjusted for splits and dividends |

### Known data quality issues to handle
- Stocks added/removed from S&P 500 during the study window (survivorship bias)
- Missing trading days (holidays, halts)
- Corporate actions creating discontinuities
- Tickers that changed (e.g. FB → META)

**Decision**: Use adjusted-close prices as provided by yfinance. Accept the current
constituent list as the universe (mild survivorship bias). Document this as a
limitation. For a production benchmark the correct approach would be point-in-time
constituents data; this is out of scope for Phase 1.

---

## Benchmark Configurations

### Phase 1 Configurations (this sprint)

| Config ID | Storage | Compute engine | Portfolio sizes |
|-----------|---------|---------------|-----------------|
| BL-001    | CSV per stock | Pandas row loop | 100, 1K |
| BL-002    | CSV per stock | NumPy matmul | 100, 1K, 100K, 1M |
| BL-003    | Combined long CSV | NumPy matmul | 100, 1K, 100K, 1M |

### Phase 2 Configurations (planned)
| Config ID | Storage | Compute engine | Notes |
|-----------|---------|---------------|-------|
| ST-001    | Parquet (per-stock) | NumPy matmul | Columnar read |
| ST-002    | Parquet (wide, date-indexed) | NumPy matmul | Single file |
| ST-003    | Arrow IPC | NumPy matmul | Zero-copy read |
| ST-004    | HDF5 | NumPy matmul | Traditional quant format |
| ST-005    | Zarr | NumPy matmul | Cloud-native chunked |

### Phase 3 Configurations (planned)
| Config ID | Storage | Compute engine | Notes |
|-----------|---------|---------------|-------|
| CP-001    | Arrow IPC | Numba (parallel CPU) | @njit(parallel=True) |
| CP-002    | Arrow IPC | CuPy (GPU) | Single GPU matmul |
| CP-003    | Arrow IPC | C++ (OpenMP + BLAS) | PyBind11 wrapper |

### Phase 4 Configurations (planned)
| Config ID | Storage | Compute engine | Notes |
|-----------|---------|---------------|-------|
| DS-001    | Parquet on local FS | PySpark (local[20]) | Spark on 20 cores |
| DS-002    | Parquet on local FS | Dask | Dask distributed |
| DS-003    | Parquet on local FS | Ray | Ray actors |

---

## Decisions Log

| Date       | Decision | Rationale |
|------------|---------|-----------|
| 2026-02-19 | Use S&P 500 as universe | ~500 stocks, well-known, freely available, reproducible |
| 2026-02-19 | Use yfinance for price data | Free, no API key, reproducible by any reader |
| 2026-02-19 | Portfolio = random K=30 stocks + Dirichlet weights | Realistic diversification constraint; Dirichlet gives non-degenerate weights |
| 2026-02-19 | Metrics = cumulative return + annualised Sharpe | Covers both simple and risk-adjusted performance; sufficient complexity to be interesting |
| 2026-02-19 | Start with 1M materialised portfolios in CSV | Right balance: fits in 64GB RAM, non-trivial to compute, establishes baseline |
| 2026-02-19 | 100M+ portfolios will be seeded, not materialised | At 100M × 30 stocks × 8 bytes = 24GB for weights alone; marginal with 64GB RAM. Seeded approach is the only option at 1B |

---

## Results

### Phase 1 — CSV Baseline (2026-02-19)

Machine: 20 physical / 28 logical cores, 67.2 GB RAM, warm cache.
All runs: 5 repetitions, 1 warmup, seed=42, K=15, universe=100 stocks.

| Config | Storage | N | Throughput (p/s) | Load time (s) |
|--------|---------|---|-----------------|--------------|
| BL-001 | csv_per_stock | 100 | 1,430 | 0.067 |
| BL-001 | csv_per_stock | 1K | 11,670 | 0.067 |
| BL-002 | csv_per_stock | 100K | 154,374 | 0.068 |
| BL-002 | csv_per_stock | 1M | 173,593 | 0.069 |
| BL-003 | csv_wide | 100 | 7,733 | 0.012 |
| BL-003 | csv_wide | 1K | 66,507 | 0.012 |
| BL-003 | csv_wide | 100K | 173,894 | 0.009 |
| BL-003 | csv_wide | 1M | 178,391 | 0.009 |

**Phase 1 findings**:
1. Storage format is the dominant factor at small N. Per-stock CSV (100 file opens)
   adds ~68ms of load overhead regardless of N — the OS must stat/open/read/close
   each file independently.
2. Wide CSV eliminates the file-open overhead (single parse), dropping load time
   from 79ms → 12ms (6.6x) at N=100. This translates to 5.4x overall speedup.
3. At N ≥ 100K, compute (BLAS matmul) dominates and all formats converge
   (~175K portfolios/sec). Storage format is irrelevant at scale — compute is the
   bottleneck.
4. RAM usage scales linearly: N=100K ≈ 3GB, N=1M ≈ 22GB (dominated by the 1M×100
   float32 weight matrix loaded into memory).
5. `io_read_mb = 0` in all runs (warm page cache). Cold-cache benchmarks require
   root to clear /proc/sys/vm/drop_caches — a key caveat for production scenarios.

---

### Phase 2 — Parquet Storage (2026-02-19)

Same machine, engine (numpy_vectorised), seed, K as Phase 1.
Parquet variants: per_stock (snappy), wide_snappy, wide_zstd, wide_uncompressed.

| Config | Storage | N | Throughput (p/s) | Load time (s) | vs BL-001 |
|--------|---------|---|-----------------|--------------|---------|
| ST-001 | parquet_per_stock | 100 | 1,390 | 0.071 | 0.97x |
| ST-001 | parquet_per_stock | 1M | 176,591 | 0.038 | 1.02x |
| ST-002 | parquet_wide_snappy | 100 | 9,737 | 0.008 | **6.8x** |
| ST-002 | parquet_wide_snappy | 1M | 175,988 | 0.005 | 1.01x |
| ST-003 | parquet_wide_zstd | 100 | 5,480 | 0.012 | 3.8x |
| ST-003 | parquet_wide_zstd | 1M | 176,507 | 0.004 | 1.02x |
| ST-004 | parquet_wide_uncompressed | 100 | **20,194** | **0.004** | **14.1x** |
| ST-004 | parquet_wide_uncompressed | 1M | 176,456 | 0.005 | 1.02x |

**Phase 2 key findings**:

1. **Parquet per-stock ≈ CSV per-stock at small N**: Columnar binary format provides
   no benefit when the bottleneck is OS file-open overhead (100 syscalls). Load
   time 71ms vs 79ms — marginal improvement. Both are dominated by file enumeration.

2. **Parquet wide (uncompressed) is the fastest loader at N=100**: 4ms vs 12ms for
   CSV wide — 3x faster for the pure I/O step. Translates to 14.1x overall speedup
   at N=100 vs the pandas/csv_per_stock baseline.

3. **Compression creates a non-trivial overhead at small data sizes**:
   - Uncompressed Parquet: 4ms load (fastest)
   - Snappy Parquet: 8ms (2x slower than uncompressed; CPU decompression visible)
   - zstd Parquet: 12ms (3x slower; higher compression = more CPU work at decode)
   - For 0.74–1.02 MB files, decompression overhead *exceeds* the I/O savings
   This reverses at large files (cloud/network I/O) where compression reduces
   bytes transferred at the cost of CPU. Not applicable here (local NVMe).

4. **At N ≥ 100K, all storage formats fully converge** (~175K p/s) — the matmul
   takes 570ms, dominating the 4–9ms load time completely.

5. **Storage size comparison** (price data for 100 stocks × 1,257 days):
   - csv_per_stock total: 3.50 MB
   - csv_wide: 2.19 MB
   - parquet_per_stock total: 1.37 MB (2.6x smaller)
   - parquet_wide_uncompressed: 1.02 MB (2.1x smaller than CSV wide)
   - parquet_wide_snappy: 0.74 MB (3.0x smaller)
   - parquet_wide_zstd: 0.49 MB (4.5x smaller)

6. **Hypothesis H3 (parquet ≥5x faster) is partially confirmed**: At small N and
   uncompressed format, we see 14x speedup. But at N ≥ 100K, storage format is
   irrelevant. The hypothesis needs to be stated per-regime.

---

### Open Questions Arising from Phase 2

- [ ] At what exact N does compute overtake I/O? (appears to be between N=1K and N=100K)
- [ ] Would Apache Arrow IPC (zero-copy memory mapping) eliminate load time entirely?
- [ ] How would cold-cache results differ? (need root for drop_caches)
- [ ] Does columnar Parquet help more when we read *subsets* of columns?
  (In our current workload we always read all 100 tickers — no column pruning benefit)

---

## Open Questions

- [ ] Should we include transaction costs in return calculation? (Out of scope for Phase 1)
- [ ] How to handle portfolios where a stock has missing data for part of the window?
- [ ] At what N does the portfolio CSV itself become the I/O bottleneck vs. the price data?
- [ ] Is it worth benchmarking columnar storage for the portfolio weight matrix itself?

---

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*.
- Apache Arrow documentation — zero-copy IPC format
- RAPIDS cuDF documentation — GPU DataFrame operations
- Apache Spark MLlib — distributed linear algebra
- Pedregosa et al. (2011) — benchmark methodology best practices

---

### Phase 3b — Extended Compute Engine Comparison (2026-02-20)

Extends Phase 3 with seven additional languages and runtimes, all on
`parquet_wide_uncompressed` storage at scales 100 / 1K / 100K / 1M.

| Config ID | Engine               | Language | Parallelism               | Status    |
|-----------|----------------------|----------|---------------------------|-----------|
| CP-007    | polars_engine        | Python   | Rayon (via Polars)        | ✅ complete |
| CP-008    | duckdb_sql           | Python   | DuckDB internal           | ✅ complete |
| CP-009    | rust_rayon_nightly   | Rust     | Rayon + fadd_fast         | ✅ complete |
| CP-010    | fortran_openmp       | FORTRAN  | OpenMP + -ffast-math      | ✅ complete |
| CP-011    | julia_loopvec        | Julia    | Threads.@threads + @turbo | ✅ complete |
| CP-012    | go_goroutines        | Go       | Goroutine worker pool     | ✅ complete |
| CP-013    | java_vector_api      | Java     | ForkJoinPool + Vector API | ✅ complete |

**Phase 3b Results** (portfolios/sec, warm cache, 5 reps, 2026-02-20):

| Engine              | N=100  | N=1K    | N=100K  | N=1M    | vs NumPy (1M) |
|---------------------|--------|---------|---------|---------|---------------|
| fortran_openmp      | 20,517 | 160,694 | 779,029 | 828,035 | **4.7×**      |
| julia_loopvec       | 23,250 | 199,681 | 707,409 | 745,475 | **4.2×**      |
| rust_rayon_nightly  | 10,045 | 79,994  | 655,304 | 654,643 | **3.7×**      |
| java_vector_api     | 19,585 | 117,481 | 460,764 | 476,253 | **2.7×**      |
| go_goroutines       | 20,773 | 118,287 | 323,932 | 344,630 | **1.9×**      |
| polars_engine       | 1,943  | 22,219  | 105,927 | 83,811  | 0.47×         |
| duckdb_sql          | 884    | 1,119   | 11,360  | —*      | —             |

*DuckDB N=1M: data-melt step creates 100M-row DataFrame (prohibitively slow).

**Phase 3b Hypothesis Status**

| ID    | Hypothesis | Status | Observation |
|-------|-----------|--------|-------------|
| H-PL1 | Polars ~50–150K/s, below NumPy | ✅ Confirmed | 84–106K/s at 100K–1M; degraded at 1M (lazy planning overhead) |
| H-DK1 | DuckDB ~10–80K/s | ✅ Confirmed | 11K/s at 100K; data-melt is the bottleneck |
| H-RN1 | Rust nightly ≈ C++ (~800–950K/s) | ⚠️ Partial | 655K at 1M — +41% over stable but still 25% below Numba |
| H-F1  | FORTRAN within ±5% of C++ | ✅ Confirmed | 828K vs 826K at 1M — 0.2% difference |
| H-JL1 | Julia ≈ Numba | ✅ Confirmed | 745K vs 920K at 1M; Julia leads at 1K (200K vs 193K) |
| H-GO1 | Go 200–400K/s | ✅ Confirmed | 345K at 1M — scalar, goroutine pool, upper bound of range |
| H-JV1 | Java Vector API 400–700K/s | ✅ Confirmed | 476K at 1M — HotSpot C2 AVX2 after warmup |

**Key findings from Phase 3b**:

1. **FORTRAN ≈ C++ (0.2% difference at N=1M)**: gfortran with `-ffast-math -march=native
   -fopenmp` through the GCC backend produces near-identical AVX2 code to g++. Critical:
   requires explicit-shape dummy args `r(U, T)` with `BIND(C)` — deferred-shape
   `C_F_POINTER` arrays prevent gfortran from proving stride-1 and fall back to scalar SSE2.

2. **Julia `@turbo` + `Threads.@threads` is the best scripted-language result**: 745K/s
   (4.2× NumPy) with no C extension — pure Julia code, LLVM backend, polyhedral SIMD.
   Julia leads at N=1K (200K vs 193K for Numba) due to lower dispatch overhead.

3. **Rust nightly `fadd_fast` closes 41% of the gap to C++/Numba**: 654K vs 465K.
   The `llvm.fadd.fast` attribute allows LLVM to vectorise the FP reduction the same
   way `-ffast-math` does for GCC. The remaining 25% gap vs Numba reflects broader
   loop-fusion opportunities that `-ffast-math` enables at the function level.

4. **Java Vector API is competitive**: 476K/s (2.7×) — comparable to Rust stable (465K)
   and Go in opposite direction. HotSpot C2 JIT compiles `DoubleVector.SPECIES_256`
   `fma` operations to AVX2 after warmup (~100 iterations).

5. **Go goroutines: 1.9× without SIMD**: scalar arithmetic, cooperative goroutine
   scheduler. The goroutine pool assigns N/28 portfolios per worker; scheduling
   overhead is negligible at this chunk size.

**Memory layout notes (CRITICAL for native engines)**

All returns and weight arrays passed to Phase 3b engines are C-order (row-major) numpy
float64 arrays. Engines that use column-major layouts (FORTRAN, Julia) must handle the
transposition explicitly via pointer arithmetic or index swapping — not array copying.

Full research notes: docs/benchmarks/phase3b_engines.md

---

### Phase 3c — Float32 and Alternative Runtimes (2026-02-20)

Additional engines targeting specific architectural questions: memory-bandwidth effects
(float32), ATen/XLA compute paths (PyTorch, JAX), Eigen BLAS (C++), and faer (Rust).

| Config ID | Engine               | Language | Question                            | Status        |
|-----------|----------------------|----------|-------------------------------------|---------------|
| CP-014    | numpy_float32        | Python   | Does float32 matmul beat float64?   | ⚠️ partial (1K only) |
| CP-015    | pytorch_cpu          | Python   | Can ATen matmul match NumPy BLAS?   | ⏳ pending full sweep |
| CP-016    | jax_cpu              | Python   | XLA-JIT vs OpenBLAS DGEMM           | ⏳ pending full sweep |
| CP-017    | cpp_eigen            | C++      | Header-only BLAS (Eigen) vs manual? | ⏳ pending build (needs libeigen3-dev) |
| CP-018    | rust_faer            | Rust     | faer GEMM vs manual Welford         | ⚠️ partial (1K only) |

**Preliminary results** (N=1K only; full sweep not yet run):

| Engine         | N=1K throughput | Notes |
|----------------|-----------------|-------|
| rust_faer      | 162,549/s       | faer GEMM + Rayon Welford; ~2.8× NumPy at 1K |
| numpy_float32  | 49,408/s        | float32 matmul; slower at 1K due to dtype conversion overhead |

**Hypotheses**:
- `numpy_float32` expected to match or exceed float64 at large N (halved L3 footprint)
  but will show overhead at small N due to `np.ascontiguousarray` dtype cast
- `pytorch_cpu` expected to match `numpy_vectorised` (both use OpenBLAS DGEMM internally)
- `jax_cpu` expected to match `numpy_vectorised` after JIT warmup; XLA may outperform
  at specific shapes
- `rust_faer` expected to approach `cpp_openmp` at large N — faer uses BLAS-backed
  GEMM like Eigen, but with Rust safety and Rayon parallelism for per-row statistics

---

### Phase 3d — DuckDB Result Registry (2026-02-20)

**Motivation**: With 12 engines × 4 scales = 48 Phase 3 configurations (and growing),
per-run JSON files are no longer sufficient for ad-hoc analysis. A queryable registry
is needed for: deduplication, superseded-run tracking, and publishable Parquet exports.

**Implementation**: `src/benchmark/db.py` — DuckDB registry with:

- **Immutable JSON source of truth**: `results/<uuid>.json` files unchanged
- **Incremental ingestion**: `ingest_all()` scans `results/*.json`, skips already-ingested
  paths (tracked by `json_path UNIQUE` in the `runs` table), ingests only new files
- **Deduplication**: `config_fingerprint` = SHA-256 of
  `(implementation, storage_format, scale, k, seed, cpu_model)`. Re-running the same
  config marks the old row `superseded=TRUE`; `v_canonical` view shows only the latest
- **Export**: `results/exports/summary.parquet`, `telemetry.parquet`, `summary.csv`,
  `comparison.csv` — all regenerated from the registry on each `--report` or `--export`
- **Baked views**: `v_canonical` (non-superseded), `v_phase3` (Phase 3 engines),
  `v_speedup` (speedup relative to numpy_vectorised at each scale)

```bash
python scripts/run_benchmark.py --export    # Ingest + export; no benchmark run
python scripts/run_benchmark.py --report    # Print summary + export automatically
```

