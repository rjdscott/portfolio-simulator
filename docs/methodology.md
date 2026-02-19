# Experimental Methodology

## 1. Research Objective

This benchmark study answers the following empirical question:

> Given a fixed dataset of daily equity returns for the S&P 500 over 5 years,
> how does wall-clock throughput (portfolios/second) scale as a function of
> (a) portfolio count N, (b) storage format, (c) compute engine, and
> (d) implementation language?

The study is designed to be reproducible by any practitioner with access to a
modern workstation. No proprietary data or software is required.

---

## 2. Data

### 2.1 Universe

The equity universe is the S&P 500 as of 2026-02-19. The exact constituent list
is committed to `data/raw/metadata/sp500_constituents.csv`. This creates mild
**survivorship bias** (we use the current index composition, not the historical
point-in-time composition), which is explicitly accepted as a limitation.

The S&P 500 was chosen over broader universes for three reasons:
1. Free, high-quality data is reliably available via Yahoo Finance
2. The 500-stock universe is large enough to make random portfolios meaningful
3. It is a universally recognised benchmark, making results interpretable

### 2.2 Price Data

- **Source**: Yahoo Finance via the `yfinance` Python library
- **Field**: Adjusted Close price (accounts for splits and dividends)
- **Frequency**: Daily (business days only)
- **Date range**: 2020-01-01 to 2024-12-31 (~1,258 trading days)
- **Fetch date**: Committed in `data/raw/metadata/data_manifest.json`

Missing values are handled as follows:
- If a stock has > 5% missing days, it is excluded from the universe
- Remaining missing values are forward-filled (carry last known price)
- This is documented in the data validation report at `data/raw/metadata/validation_report.json`

### 2.3 Returns

From the adjusted-close price series `P`, we compute log returns:

```
r_t = ln(P_t / P_{t-1})
```

Log returns are used because they are time-additive, which simplifies
cumulative return calculation to a simple sum. The price matrix is stored once
and the returns matrix derived in memory at benchmark runtime (not pre-stored),
so that price-load time is isolated from return-compute time.

---

## 3. Portfolio Definition

### 3.1 Composition

Each portfolio j is defined by:
- A subset S_j of K=30 stocks drawn uniformly at random from the universe
  (without replacement)
- A weight vector w_j ∈ ℝ^K with w_j > 0 and sum(w_j) = 1
  (Dirichlet(α=1) distribution, equivalent to uniform on the simplex)

K=30 was chosen to represent a diversified but realistic portfolio size.

### 3.2 Reproducibility

Portfolios are generated from a global seed (default: 42). For portfolio index j,
the RNG state is set to `seed + j`, making any portfolio independently
reproducible without generating all preceding portfolios.

For N ≤ 10M: portfolios are materialised to disk as a weight matrix CSV with
shape (N, K), with column headers being the ticker symbols.

For N > 10M: portfolios are generated in batches in memory from seeds.
The full weight matrix is never written to disk.

---

## 4. Return Metrics

For a portfolio with weight vector w and daily log-return matrix R (T×K):

**Portfolio daily log-returns**:
```
r_p = R @ w    (shape: T×1)
```

**Cumulative return** (total return over the study window):
```
CR = exp(sum(r_p)) - 1
```

**Annualised Sharpe ratio** (assuming risk-free rate = 0 for simplicity):
```
Sharpe = (mean(r_p) / std(r_p)) * sqrt(252)
```

Both metrics are computed for every portfolio. The output is an array of shape
(N, 2): [cumulative_return, sharpe_ratio].

---

## 5. Benchmark Protocol

### 5.1 What is timed

The benchmark clock starts when the first byte of data is requested from storage
and stops when the last result is written to the output array. Specifically:

1. **Load**: Read the price matrix from disk
2. **Transform**: Compute the log-returns matrix
3. **Portfolio I/O**: Load the portfolio weight matrix (if materialised)
4. **Compute**: Execute the matrix multiplication and metric calculation
5. **Collect**: Gather results into a final output array

Steps 1-5 are timed as a unit (end-to-end wall clock). Steps 1-2 are also
timed separately in a sub-benchmark to isolate storage performance.

### 5.2 Repetitions and warmup

Each configuration is run **5 times**. Before timing begins:
- OS page cache is dropped: `sync && echo 3 > /proc/sys/vm/drop_caches`
- A warm-up run (not timed) allows JIT compilers (Numba, JVM) to compile

### 5.3 Metrics recorded

| Metric | Unit | How measured |
|--------|------|-------------|
| Wall-clock time | seconds | `time.perf_counter()` / Rust `Instant` / C++ `chrono` |
| Peak RSS | MB | `/proc/self/status` or `ru_maxrss` |
| CPU utilisation | % | `psutil` snapshot at peak |
| GPU utilisation | % | `nvidia-smi` or CuPy event timing |
| I/O bytes read | MB | `psutil.disk_io_counters()` delta |
| Throughput | portfolios/sec | N / wall_clock_time |

### 5.4 Result schema

All implementations write results to `results/<run_id>.json` following the
schema defined in `common/schemas/benchmark_result.schema.json`.

---

## 6. Implementations

### 6.1 Python

| Variant | Key libraries | Parallelism |
|---------|-------------|-------------|
| `pandas_baseline` | pandas, pure Python loop | None |
| `numpy_vectorised` | numpy, matmul | BLAS (MKL/OpenBLAS) |
| `numba_parallel` | numba @njit(parallel=True) | OpenMP via Numba |
| `cupy_gpu` | CuPy | CUDA |

### 6.2 C++

| Variant | Key libraries | Parallelism |
|---------|-------------|-------------|
| `openmp` | Eigen or raw arrays, OpenMP | CPU multi-thread |
| `blas` | cblas_dgemm | BLAS SGEMM/DGEMM |
| `cuda` | CUDA cublasSgemm | GPU |

Build system: CMake 3.20+

### 6.3 Rust

| Variant | Key libraries | Parallelism |
|---------|-------------|-------------|
| `rayon` | rayon, ndarray | Rayon work-stealing |
| `ndarray` | ndarray, blas-src | BLAS via blas-src |

Build system: Cargo

### 6.4 Kotlin / JVM

| Variant | Key libraries | Parallelism |
|---------|-------------|-------------|
| `jvm` | kotlinx.coroutines, multik | JVM threads |
| `coroutines` | Kotlin coroutines | Coroutine-based parallelism |

Build system: Gradle

### 6.5 Distributed (Phase 4)

| Variant | Framework | Target scale |
|---------|----------|-------------|
| `spark_local` | PySpark local[20] | 100M–1B |
| `dask_local` | Dask distributed | 100M |
| `ray_local` | Ray | 100M |

---

## 7. Statistical Analysis

For each (implementation, scale) pair, we report:
- Median throughput (portfolios/sec)
- p10–p90 range (variability)
- Speedup ratio relative to `pandas_baseline` at 100 portfolios

We test whether differences between implementations are statistically
significant using the Wilcoxon signed-rank test (non-parametric, appropriate
for small sample sizes with unknown distributions).

---

## 8. Limitations

1. **Survivorship bias**: Current S&P 500 constituents only
2. **Single machine** (Phases 1–3): Results do not generalise to distributed environments without Phase 4 data
3. **Synthetic portfolios**: Real institutional portfolios have constraints (sector limits, turnover) not modelled here
4. **Risk-free rate**: Sharpe computed with Rf=0 for simplicity
5. **Single GPU**: No multi-GPU configurations tested
6. **No transaction costs**: Pure return calculation only
7. **JVM warmup**: JVM implementations given one warmup run; JIT compilation time is excluded from reported time
