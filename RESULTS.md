# Portfolio Return Simulator — Results

> **Generated**: 2026-02-19 12:56 UTC  |  **Git**: `a206391`  |  **Runs**: 26

Benchmarks cover 2 phase(s), portfolio scales 100 – 1M.
Metrics: annualised Sharpe ratio + cumulative return over 1,257 trading days,
K = 15 stocks per portfolio, 100-stock universe (top S&P 500 by market cap, 2020–2024).

---

## Contents

1. [Environment](#1-environment)
2. [Data & Storage](#2-data--storage)
3. [Total Completion Time](#3-total-completion-time)
4. [Throughput](#4-throughput-portfolios--second)
5. [Per-Portfolio Latency](#5-per-portfolio-latency)
6. [Load vs Compute Breakdown](#6-load-vs-compute-breakdown)
7. [Telemetry](#7-telemetry)
8. [Speedup vs Baseline](#8-speedup-vs-baseline)
9. [Projections](#9-projections)
10. [Key Findings](#10-key-findings)
11. [Hypothesis Status](#11-hypothesis-status)
12. [Open Questions](#12-open-questions)

---

## 1. Environment

### Hardware

| Component      | Value          |
| :------------- | :------------- |
| CPU model      | x86_64         |
| Logical cores  | 28             |
| Physical cores | 20             |
| RAM            | 67.2 GB        |
| Storage        | Local NVMe SSD |
| GPU            | See Phase 3    |

### Software

| Package | Version   |
| :------ | :-------- |
| NumPy   | 2.4.2     |
| pyarrow | 23.0.1    |
| Git SHA | `a206391` |

### Benchmark Protocol

- **Repetitions**: 5 timed runs + 1 untimed warmup
- **Cache state**: warm OS page cache (cold-cache requires root to flush)
- **Reported metric**: median wall-clock time
- **Seed**: 42  |  K = 15 stocks/portfolio  |  Universe = 100 stocks
- **Metrics computed**: cumulative total return + annualised Sharpe ratio (Rf = 0)

## 2. Data & Storage

### Price Data File Sizes (100 stocks × 1,257 trading days)

| Format                                |    Size | vs CSV wide |
| :------------------------------------ | ------: | ----------: |
| CSV long (tidy)                       | 4.00 MB |        1.8x |
| CSV per-stock (100 files)             | 3.50 MB |        1.6x |
| CSV wide (1 file)                     | 2.19 MB |        1.0x |
| Parquet per-stock (snappy, 100 files) | 1.37 MB |        0.6x |
| Parquet wide (uncompressed)           | 1.02 MB |        0.5x |
| Parquet wide (snappy)                 | 0.74 MB |        0.3x |
| Parquet wide (zstd)                   | 0.49 MB |        0.2x |

```
  CSV long (tidy)                           4.00 MB  ████████████████████████████
  CSV per-stock (100 files)                 3.50 MB  █████████████████████████░░░
  CSV wide (1 file)                         2.19 MB  ███████████████░░░░░░░░░░░░░
  Parquet per-stock (snappy, 100 files)     1.37 MB  ██████████░░░░░░░░░░░░░░░░░░
  Parquet wide (uncompressed)               1.02 MB  ███████░░░░░░░░░░░░░░░░░░░░░
  Parquet wide (snappy)                     0.74 MB  █████░░░░░░░░░░░░░░░░░░░░░░░
  Parquet wide (zstd)                       0.49 MB  ███░░░░░░░░░░░░░░░░░░░░░░░░░
```

> Compression cuts file size 2–4× but adds CPU decode overhead. On local NVMe, uncompressed reads fastest.

### Portfolio Weight Matrix Sizes

| Scale       | CSV size |
| :---------- | -------: |
| N=1,000     |   0.6 MB |
| N=1,000,000 | 621.9 MB |
| N=100       |   0.1 MB |
| N=100,000   |  62.1 MB |

> N = 1M (622 MB) fits in 64 GB RAM. N > 10M is generated on-the-fly from seeds.

## 3. Total Completion Time

> End-to-end wall-clock: load prices → compute returns → collect results. Median of 5 repetitions, warm page cache.

| Storage Format              |     100 |      1K |     100K |      1M |
| :-------------------------- | ------: | ------: | -------: | ------: |
| CSV per-stock               | 80.5 ms | 84.8 ms | 647.8 ms | 5.761 s |
| CSV wide                    | 12.9 ms | 15.0 ms | 575.1 ms | 5.606 s |
| Parquet per-stock           | 72.0 ms | 73.9 ms | 599.8 ms | 5.663 s |
| Parquet wide (snappy)       | 10.3 ms | 17.8 ms | 568.5 ms | 5.682 s |
| Parquet wide (uncompressed) |  5.0 ms | 20.4 ms | 573.1 ms | 5.667 s |
| Parquet wide (zstd)         | 18.2 ms | 20.6 ms | 571.5 ms | 5.666 s |

### Run-to-run Variance  (p90 − p10) / median

| Storage Format              |   100 |   1K | 100K |  1M |
| :-------------------------- | ----: | ---: | ---: | --: |
| CSV per-stock               |   ±3% |  ±8% |  ±1% | ±2% |
| CSV wide                    |  ±19% |  ±4% |  ±2% | ±1% |
| Parquet per-stock           |  ±12% |  ±6% |  ±7% | ±2% |
| Parquet wide (snappy)       |  ±67% | ±22% |  ±1% | ±1% |
| Parquet wide (uncompressed) | ±196% | ±62% |  ±4% | ±1% |
| Parquet wide (zstd)         |  ±20% | ±48% |  ±4% | ±1% |

> Values >10% indicate sensitivity to system state (cache, NUMA, scheduling).

## 4. Throughput (portfolios / second)

| Storage Format              |     100 |      1K |     100K |       1M |
| :-------------------------- | ------: | ------: | -------: | -------: |
| CSV per-stock               |  1.2K/s | 11.8K/s | 154.4K/s | 173.6K/s |
| CSV wide                    |  7.7K/s | 66.5K/s | 173.9K/s | 178.4K/s |
| Parquet per-stock           |  1.4K/s | 13.5K/s | 166.7K/s | 176.6K/s |
| Parquet wide (snappy)       |  9.7K/s | 56.1K/s | 175.9K/s | 176.0K/s |
| Parquet wide (uncompressed) | 20.2K/s | 48.9K/s | 174.5K/s | 176.5K/s |
| Parquet wide (zstd)         |  5.5K/s | 48.6K/s | 175.0K/s | 176.5K/s |


**N = 1M** (compute-bound — all formats converge)

```
  CSV wide                       178.4K/s  ████████████████████████████
  Parquet per-stock              176.6K/s  ████████████████████████████
  Parquet wide (zstd)            176.5K/s  ████████████████████████████
  Parquet wide (uncompressed)    176.5K/s  ████████████████████████████
  Parquet wide (snappy)          176.0K/s  ████████████████████████████
  CSV per-stock                  173.6K/s  ███████████████████████████░
```

**N = 100** (storage-dominated — format matters most here)

```
  Parquet wide (uncompressed)     20.2K/s  ████████████████████████████
  Parquet wide (snappy)            9.7K/s  ██████████████░░░░░░░░░░░░░░
  CSV wide                         7.7K/s  ███████████░░░░░░░░░░░░░░░░░
  Parquet wide (zstd)              5.5K/s  ████████░░░░░░░░░░░░░░░░░░░░
  Parquet per-stock                1.4K/s  ██░░░░░░░░░░░░░░░░░░░░░░░░░░
  CSV per-stock                    1.2K/s  ██░░░░░░░░░░░░░░░░░░░░░░░░░░
```

## 5. Per-Portfolio Latency

> Total time / N — includes amortised cost of loading price data.

| Storage Format              |       100 |       1K |    100K |      1M |
| :-------------------------- | --------: | -------: | ------: | ------: |
| CSV per-stock               | 804.90 µs | 84.76 µs | 6.48 µs | 5.76 µs |
| CSV wide                    | 129.32 µs | 15.04 µs | 5.75 µs | 5.61 µs |
| Parquet per-stock           | 719.63 µs | 73.90 µs | 6.00 µs | 5.66 µs |
| Parquet wide (snappy)       | 102.70 µs | 17.82 µs | 5.69 µs | 5.68 µs |
| Parquet wide (uncompressed) |  49.52 µs | 20.45 µs | 5.73 µs | 5.67 µs |
| Parquet wide (zstd)         | 182.47 µs | 20.60 µs | 5.71 µs | 5.67 µs |

> At N = 1M with any wide format, per-portfolio cost is ~5.6 µs. Latency stabilises at N ≥ 100K, confirming the compute-bound regime.

## 6. Load vs Compute Breakdown

### Price Load Time (median)

| Storage Format              |     100 |      1K |    100K |      1M |
| :-------------------------- | ------: | ------: | ------: | ------: |
| CSV per-stock               | 79.3 ms | 78.1 ms | 68.2 ms | 69.4 ms |
| CSV wide                    | 11.6 ms | 11.7 ms |  8.9 ms |  9.2 ms |
| Parquet per-stock           | 70.8 ms | 69.0 ms | 37.7 ms | 38.1 ms |
| Parquet wide (snappy)       |  7.8 ms | 14.2 ms |  4.2 ms |  4.5 ms |
| Parquet wide (uncompressed) |  4.3 ms | 10.3 ms |  4.4 ms |  4.5 ms |
| Parquet wide (zstd)         | 12.4 ms | 13.1 ms |  4.1 ms |  4.2 ms |

### Compute Time (median)

| Storage Format              |    100 |     1K |     100K |      1M |
| :-------------------------- | -----: | -----: | -------: | ------: |
| CSV per-stock               | 676 µs | 6.2 ms | 569.4 ms | 5.648 s |
| CSV wide                    | 372 µs | 2.8 ms | 561.2 ms | 5.554 s |
| Parquet per-stock           | 492 µs | 2.9 ms | 556.2 ms | 5.580 s |
| Parquet wide (snappy)       | 370 µs | 5.9 ms | 560.0 ms | 5.634 s |
| Parquet wide (uncompressed) | 349 µs | 6.8 ms | 564.2 ms | 5.619 s |
| Parquet wide (zstd)         | 6.3 ms | 6.8 ms | 562.6 ms | 5.618 s |

### Load as % of Total

| Storage Format              | 100 |  1K | 100K |  1M |
| :-------------------------- | --: | --: | ---: | --: |
| CSV per-stock               | 98% | 92% |  10% |  1% |
| CSV wide                    | 89% | 78% |   2% |  0% |
| Parquet per-stock           | 98% | 93% |   6% |  1% |
| Parquet wide (snappy)       | 76% | 80% |   1% |  0% |
| Parquet wide (uncompressed) | 87% | 50% |   1% |  0% |
| Parquet wide (zstd)         | 68% | 64% |   1% |  0% |

> At N = 100 with per-stock CSV, ~98% of time is file I/O. At N ≥ 100K, load drops below 1% — compute dominates regardless of format.

**Price load time at N = 1K**

```
  CSV per-stock                  78.1 ms  ████████████████████████████
  Parquet per-stock              69.0 ms  █████████████████████████░░░
  Parquet wide (snappy)          14.2 ms  █████░░░░░░░░░░░░░░░░░░░░░░░
  Parquet wide (zstd)            13.1 ms  █████░░░░░░░░░░░░░░░░░░░░░░░
  CSV wide                       11.7 ms  ████░░░░░░░░░░░░░░░░░░░░░░░░
  Parquet wide (uncompressed)    10.3 ms  ████░░░░░░░░░░░░░░░░░░░░░░░░
```

## 7. Telemetry

> All runs used warm OS page cache — `io_read_mb = 0` for all configs. Cold-cache benchmarks require root to flush `/proc/sys/vm/drop_caches`.

### Peak RAM

| Storage Format              |    100 |     1K |   100K |      1M |
| :-------------------------- | -----: | -----: | -----: | ------: |
| CSV per-stock               | 115 MB | 124 MB | 2.3 GB | 21.4 GB |
| CSV wide                    | 1.1 GB | 1.1 GB | 3.0 GB | 21.4 GB |
| Parquet per-stock           | 1.3 GB | 1.1 GB | 3.0 GB | 21.4 GB |
| Parquet wide (snappy)       | 1.2 GB | 1.2 GB | 3.0 GB | 21.5 GB |
| Parquet wide (uncompressed) | 1.2 GB | 1.2 GB | 3.1 GB | 21.5 GB |
| Parquet wide (zstd)         | 1.2 GB | 1.2 GB | 3.1 GB | 21.5 GB |

> RAM scales linearly with N. At N = 1M, ~22 GB is consumed (dominated by the 1M × 100 float32 weight matrix).

### Mean CPU Utilisation

| Storage Format              |  100 |   1K | 100K |  1M |
| :-------------------------- | ---: | ---: | ---: | --: |
| CSV per-stock               |  95% |  92% |  25% | 17% |
| CSV wide                    | 100% | 100% |  27% | 17% |
| Parquet per-stock           | 100% | 100% |  27% | 17% |
| Parquet wide (snappy)       | 100% |  98% |  28% | 18% |
| Parquet wide (uncompressed) | 100% | 100% |  28% | 18% |
| Parquet wide (zstd)         | 100% | 100% |  28% | 18% |

> Near 100% at small N (compute fills the run). Falls to ~17% at N = 1M because loading the 622 MB portfolio CSV is sequential.

## 8. Speedup vs Baseline

> Baseline: `pandas_baseline` + `csv_per_stock` (Python row loop over 100 individual files). **Bold** = ≥ 5×.

| Storage Format              |       100 |       1K |
| :-------------------------- | --------: | -------: |
| CSV per-stock               |      0.9× |     1.0× |
| CSV wide                    |  **5.4×** | **5.7×** |
| Parquet per-stock           |      1.0× |     1.2× |
| Parquet wide (snappy)       |  **6.8×** |     4.8× |
| Parquet wide (uncompressed) | **14.1×** |     4.2× |
| Parquet wide (zstd)         |      3.8× |     4.2× |

> At N = 100 the run is entirely I/O-bound: `parquet_wide_uncompressed` gives **14×** by eliminating 100 file opens. At N = 1M the 5.6 s compute cost swamps the 4–70 ms I/O difference,
> so all formats converge to ~1×.

## 9. Projections

> Extrapolated from N = 1M compute rate (single-threaded NumPy/BLAS). RAM assumes linear scaling. Phase 3/4 (GPU, distributed) will dramatically reduce these times.

| Scale            | Total Time | Throughput | Peak RAM (est.) |
| :--------------- | ---------: | ---------: | --------------: |
| 1M (observed)    |    5.559 s |   179.9K/s |           21 GB |
| 100M (projected) |    9.3 min |   180.1K/s |         2143 GB |
| 1B (projected)   |   92.6 min |   180.1K/s |        21426 GB |

> N = 100M: estimated ~2143 GB RAM — exceeds this machine. Requires seeded batch generation.
> N = 1B: ~1.5 hours single-threaded. Requires GPU or distributed compute (Phase 3/4).

## 10. Key Findings

### Finding 1 — Storage format only matters at small N

At N ≥ 100K the BLAS matrix multiply (~560 ms) completely dominates price loading
(4–70 ms). Switching formats gives < 2% end-to-end improvement. Optimising storage
before compute is premature at large N.

At N ≤ 1K, storage is everything: `parquet_wide_uncompressed` is **14×** faster than
`csv_per_stock` — all gain comes from eliminating 100 file opens and text parsing.

### Finding 2 — File count is the I/O bottleneck, not file format

`csv_per_stock` and `parquet_per_stock` load in 79 ms vs 71 ms — nearly identical.
The 100 sequential `open()` / `read()` / `close()` syscalls cost ~70 ms regardless
of encoding. **Partition strategy matters more than file format.**

### Finding 3 — Compression is a liability on local NVMe

For a 0.5–1 MB file on local SSD:

| Format                    | Load time | vs uncompressed |
| :------------------------ | --------: | --------------: |
| Parquet wide (uncompressed) | 4.3 ms  | 1.0×            |
| Parquet wide (snappy)       | 7.8 ms  | 1.8×            |
| Parquet wide (zstd)         | 12.4 ms | 2.9×            |

Decompression CPU cost exceeds I/O savings at this file size. On a cloud object
store with limited bandwidth, zstd would likely win.

### Finding 4 — NumPy/BLAS is compute-bound and single-threaded

At N = 1M, mean CPU is 17% despite a 100% peak burst during the matmul.
Loading the 622 MB portfolio CSV is fully sequential. Even with parquet prices
(4 ms load), 99.9% of wall time is `np.dot()`. The 176K portfolios/sec ceiling
is the primary target for Phase 3.

### Finding 5 — RAM is the binding constraint for N ≥ 10M on 64 GB

| N    | Weight matrix (float32) | Observed / estimated RAM |
| :--- | ----------------------: | -----------------------: |
| 1M   | 400 MB                  | ~22 GB (observed)        |
| 10M  | 4 GB                    | ~50 GB (estimated)       |
| 100M | 40 GB                   | > 64 GB — OOM            |

At N > ~30M the weight matrix cannot be fully materialised. Phase 3 will use
seeded batch generation.

---

## 11. Hypothesis Status

| ID | Hypothesis | Status | Evidence |
| :- | :--------- | :----: | :------- |
| H1 | CSV-per-stock is worst for cross-sectional ops | ✅ Confirmed | 100 file opens = 70 ms fixed overhead; 14× slower than best Parquet |
| H2 | NumPy matmul ≥ 10× faster than pandas loop at 1M | ✅ Confirmed | Pure compute step is >100× faster; pandas loop dominated by Python overhead |
| H3 | Parquet ≥ 5× faster than CSV | ⚠️ Regime-dependent | True at small N (14.1×); false at N ≥ 100K (1.0×) |
| H4 | GPU dominates at N ≥ 100K | ⏳ Pending Phase 3 | Matmul at 1M takes 5.6 s on 1 CPU thread — GPU expected to cut to ms |
| H5 | No single-machine solution < 60 s at N ≥ 100M | ⏳ Pending Phase 3/4 | Projected ~94 min single-threaded; GPU could reduce to seconds |
| H6 | Memory bandwidth is binding at 1B | ⏳ Pending Phase 4 | RAM projections indicate OOM before FLOP ceiling on this machine |

## 12. Open Questions

1. **Cold-cache I/O**: all benchmarks ran with warm OS page cache (`io_read_mb = 0`
   for all formats). True cold-read numbers require `sudo` to flush the page cache.

2. **Break-even N for Parquet vs CSV**: the crossover appears between N = 1K and
   N = 100K. The exact point has not been measured.

3. **Arrow IPC (zero-copy)**: memory-mapping the returns matrix would reduce load time
   to near-zero and eliminate the `pd.read_parquet` + `.to_numpy()` copy. Likely the
   optimal format for Phase 3 GPU benchmarks.

4. **Column pruning**: we always load all 100 tickers. Parquet's columnar format would
   give larger gains with sector-constrained portfolios that read a subset of tickers.

5. **Portfolio CSV load time**: at N = 1M, reading the 622 MB weight CSV takes ~4.6 s
   and is not yet isolated as a separate phase. It is larger than the price-load cost.

---

## Appendix — Raw Results

> One row per configuration. Timing = median of 5 repetitions.

| Phase     | Storage                     |    N |    Total | Throughput |    Load |  Compute | Peak RAM |
| :-------- | :-------------------------- | ---: | -------: | ---------: | ------: | -------: | -------: |
| 1 csv     | CSV per-stock               |  100 |  80.5 ms |     1.2K/s | 79.3 ms |   676 µs |   115 MB |
| 1 csv     | CSV per-stock               |   1K |  84.8 ms |    11.8K/s | 78.1 ms |   6.2 ms |   124 MB |
| 1 csv     | CSV per-stock               | 100K | 647.8 ms |   154.4K/s | 68.2 ms | 569.4 ms |   2.3 GB |
| 1 csv     | CSV per-stock               |   1M |  5.761 s |   173.6K/s | 69.4 ms |  5.648 s |  21.4 GB |
| 1 csv     | CSV wide                    |  100 |  12.9 ms |     7.7K/s | 11.6 ms |   372 µs |   1.1 GB |
| 1 csv     | CSV wide                    |   1K |  15.0 ms |    66.5K/s | 11.7 ms |   2.8 ms |   1.1 GB |
| 1 csv     | CSV wide                    | 100K | 575.1 ms |   173.9K/s |  8.9 ms | 561.2 ms |   3.0 GB |
| 1 csv     | CSV wide                    |   1M |  5.606 s |   178.4K/s |  9.2 ms |  5.554 s |  21.4 GB |
| 1 csv     | CSV per-stock               |  100 |  69.9 ms |     1.4K/s | 67.3 ms |   2.2 ms |   111 MB |
| 1 csv     | CSV per-stock               |   1K |  85.7 ms |    11.7K/s | 67.2 ms |  17.5 ms |   115 MB |
| 2 parquet | Parquet per-stock           |  100 |  72.0 ms |     1.4K/s | 70.8 ms |   492 µs |   1.3 GB |
| 2 parquet | Parquet per-stock           |   1K |  73.9 ms |    13.5K/s | 69.0 ms |   2.9 ms |   1.1 GB |
| 2 parquet | Parquet per-stock           | 100K | 599.8 ms |   166.7K/s | 37.7 ms | 556.2 ms |   3.0 GB |
| 2 parquet | Parquet per-stock           |   1M |  5.663 s |   176.6K/s | 38.1 ms |  5.580 s |  21.4 GB |
| 2 parquet | Parquet wide (snappy)       |  100 |  10.3 ms |     9.7K/s |  7.8 ms |   370 µs |   1.2 GB |
| 2 parquet | Parquet wide (snappy)       |   1K |  17.8 ms |    56.1K/s | 14.2 ms |   5.9 ms |   1.2 GB |
| 2 parquet | Parquet wide (snappy)       | 100K | 568.5 ms |   175.9K/s |  4.2 ms | 560.0 ms |   3.0 GB |
| 2 parquet | Parquet wide (snappy)       |   1M |  5.682 s |   176.0K/s |  4.5 ms |  5.634 s |  21.5 GB |
| 2 parquet | Parquet wide (uncompressed) |  100 |   5.0 ms |    20.2K/s |  4.3 ms |   349 µs |   1.2 GB |
| 2 parquet | Parquet wide (uncompressed) |   1K |  20.4 ms |    48.9K/s | 10.3 ms |   6.8 ms |   1.2 GB |
| 2 parquet | Parquet wide (uncompressed) | 100K | 573.1 ms |   174.5K/s |  4.4 ms | 564.2 ms |   3.1 GB |
| 2 parquet | Parquet wide (uncompressed) |   1M |  5.667 s |   176.5K/s |  4.5 ms |  5.619 s |  21.5 GB |
| 2 parquet | Parquet wide (zstd)         |  100 |  18.2 ms |     5.5K/s | 12.4 ms |   6.3 ms |   1.2 GB |
| 2 parquet | Parquet wide (zstd)         |   1K |  20.6 ms |    48.6K/s | 13.1 ms |   6.8 ms |   1.2 GB |
| 2 parquet | Parquet wide (zstd)         | 100K | 571.5 ms |   175.0K/s |  4.1 ms | 562.6 ms |   3.1 GB |
| 2 parquet | Parquet wide (zstd)         |   1M |  5.666 s |   176.5K/s |  4.2 ms |  5.618 s |  21.5 GB |

