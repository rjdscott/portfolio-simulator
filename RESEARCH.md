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

## Results (to be populated)

Results will be recorded here as experiments run, with a link to the detailed
JSON output in `results/`.

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
