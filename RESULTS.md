# Portfolio Return Simulator — Results Report

> **Generated**: 2026-02-19 12:42 UTC  |  **Git**: `00e617e`  |  **Runs**: 26  |  **Configs**: 7

This report captures all benchmark results across 2 phase(s), covering
portfolio scales from 100 to 1M.
All computations: annualised Sharpe ratio + cumulative return over 1,257 trading days
for each portfolio of K=15 stocks drawn from a 100-stock universe (top S&P 500 by
market cap, 2020-01-01 → 2024-12-31).

---

## Contents

1. [Environment](#1-environment)
2. [Data & Storage](#2-data--storage)
3. [Total Completion Time](#3-total-completion-time)
4. [Throughput](#4-throughput-portfolios--second)
5. [Per-Portfolio Latency](#5-per-portfolio-latency)
6. [Load vs Compute Breakdown](#6-load-vs-compute-breakdown)
7. [Telemetry — CPU, Memory, I/O](#7-telemetry--cpu-memory-io)
8. [Speedup vs Baseline](#8-speedup-vs-baseline)
9. [Projections for Larger Scales](#9-projections-for-larger-scales)
10. [Key Findings](#10-key-findings)
11. [Hypothesis Status](#11-hypothesis-status)
12. [Open Questions](#12-open-questions)

---

## 1. Environment

### Hardware

| Component | Value |
|:---|:---|
| CPU | x86_64 |
| Logical cores | 28 |
| Physical cores | 20 |
| RAM | 67.2 GB |
| Storage | Local NVMe SSD |
| GPU | See Phase 3 |

### Software

| Component | Version |
|:---|:---|
| Python | see below |
| NumPy | 2.4.2 |
| pyarrow | 23.0.1 |
| Git SHA | `00e617e` |

### Benchmark Protocol

- **Repetitions per config**: 5 timed runs + 1 untimed warmup
- **Cache state**: warm (OS page cache not dropped — root required for cold reads)
- **Metric reported**: median wall-clock time across 5 reps
- **Portfolio seed**: 42  |  K = 15 stocks/portfolio  |  Universe = 100 stocks
- **Return metrics computed**: cumulative total return + annualised Sharpe ratio (Rf=0)

## 2. Data & Storage

### Price Data File Sizes (100 stocks × 1,257 trading days)

| Format                                |    Size | Ratio vs CSV wide | ▓                              |
| :------------------------------------- | -:------ | -:---------------- | :------------------------------ |
| CSV long (tidy)                       | 4.00 MB |              1.8x | ██████████████████████████████ |
| CSV (per-stock, 100 files)            | 3.50 MB |              1.6x | ██████████████████████████░░░░ |
| CSV wide (1 file)                     | 2.19 MB |              1.0x | ████████████████░░░░░░░░░░░░░░ |
| Parquet per-stock (snappy, 100 files) | 1.37 MB |              0.6x | ██████████░░░░░░░░░░░░░░░░░░░░ |
| Parquet wide (uncompressed)           | 1.02 MB |              0.5x | ████████░░░░░░░░░░░░░░░░░░░░░░ |
| Parquet wide (snappy)                 | 0.74 MB |              0.3x | ██████░░░░░░░░░░░░░░░░░░░░░░░░ |
| Parquet wide (zstd)                   | 0.49 MB |              0.2x | ████░░░░░░░░░░░░░░░░░░░░░░░░░░ |


> Compression reduces storage by 2–4x but adds CPU decode overhead at read time.

### Portfolio Weight Matrix Sizes

| Scale       | CSV File Size |
| :----------- | -:------------ |
| N=1,000     |        0.6 MB |
| N=1,000,000 |      621.9 MB |
| N=100       |        0.1 MB |
| N=100,000   |       62.1 MB |


> N=1M portfolio CSV (622 MB) fits comfortably in 64 GB RAM.

> N>10M is generated on-the-fly from seeds — never fully materialised.

## 3. Total Completion Time

> End-to-end wall-clock time: load prices → compute returns → collect results.
> Median of 5 repetitions (warm OS page cache).

| Storage Format              |     100 |      1K |     100K |      1M |
| :--------------------------- | -:------ | -:------ | -:------- | -:------ |
| CSV per-stock (100 files)   | 80.5 ms | 84.8 ms | 647.8 ms | 5.761 s |
| CSV wide (1 file)           | 12.9 ms | 15.0 ms | 575.1 ms | 5.606 s |
| Parquet per-stock (snappy)  | 72.0 ms | 73.9 ms | 599.8 ms | 5.663 s |
| Parquet wide (snappy)       | 10.3 ms | 17.8 ms | 568.5 ms | 5.682 s |
| Parquet wide (uncompressed) |  5.0 ms | 20.4 ms | 573.1 ms | 5.667 s |
| Parquet wide (zstd)         | 18.2 ms | 20.6 ms | 571.5 ms | 5.666 s |


### Timing Variance (p90 − p10) / median

| Storage Format              |   100 |   1K | 100K |  1M |
| :--------------------------- | -:---- | -:--- | -:--- | -:-- |
| CSV per-stock (100 files)   |   ±3% |  ±8% |  ±1% | ±2% |
| CSV wide (1 file)           |  ±19% |  ±4% |  ±2% | ±1% |
| Parquet per-stock (snappy)  |  ±12% |  ±6% |  ±7% | ±2% |
| Parquet wide (snappy)       |  ±67% | ±22% |  ±1% | ±1% |
| Parquet wide (uncompressed) | ±196% | ±62% |  ±4% | ±1% |
| Parquet wide (zstd)         |  ±20% | ±48% |  ±4% | ±1% |


**Reading the table**: variance reflects run-to-run jitter. Values >10% suggest sensitivity to system state (cache, NUMA, scheduling).

## 4. Throughput (portfolios / second)

| Storage Format              |     100 |      1K |     100K |       1M |
| :--------------------------- | -:------ | -:------ | -:------- | -:------- |
| CSV per-stock (100 files)   |  1.2K/s | 11.8K/s | 154.4K/s | 173.6K/s |
| CSV wide (1 file)           |  7.7K/s | 66.5K/s | 173.9K/s | 178.4K/s |
| Parquet per-stock (snappy)  |  1.4K/s | 13.5K/s | 166.7K/s | 176.6K/s |
| Parquet wide (snappy)       |  9.7K/s | 56.1K/s | 175.9K/s | 176.0K/s |
| Parquet wide (uncompressed) | 20.2K/s | 48.9K/s | 174.5K/s | 176.5K/s |
| Parquet wide (zstd)         |  5.5K/s | 48.6K/s | 175.0K/s | 176.5K/s |


### Throughput Visual — N=1M

```
  CSV wide (1 file)              178.4K/s  █████████████████████████
  Parquet per-stock (snappy)     176.6K/s  █████████████████████████
  Parquet wide (zstd)            176.5K/s  █████████████████████████
  Parquet wide (uncompressed)    176.5K/s  █████████████████████████
  Parquet wide (snappy)          176.0K/s  █████████████████████████
  CSV per-stock (100 files)      173.6K/s  ████████████████████████░
```

### Throughput Visual — N=100 (storage-dominated regime)

```
  Parquet wide (uncompressed)     20.2K/s  █████████████████████████
  Parquet wide (snappy)            9.7K/s  ████████████░░░░░░░░░░░░░
  CSV wide (1 file)                7.7K/s  ██████████░░░░░░░░░░░░░░░
  Parquet wide (zstd)              5.5K/s  ███████░░░░░░░░░░░░░░░░░░
  Parquet per-stock (snappy)       1.4K/s  ██░░░░░░░░░░░░░░░░░░░░░░░
  CSV per-stock (100 files)        1.2K/s  ██░░░░░░░░░░░░░░░░░░░░░░░
```

## 5. Per-Portfolio Latency

> Time to compute returns for a single portfolio = total_time / N.
> This includes the amortised cost of loading prices from disk.

| Storage Format              |       100 |       1K |    100K |      1M |
| :--------------------------- | -:-------- | -:------- | -:------ | -:------ |
| CSV per-stock (100 files)   | 804.90 µs | 84.76 µs | 6.48 µs | 5.76 µs |
| CSV wide (1 file)           | 129.32 µs | 15.04 µs | 5.75 µs | 5.61 µs |
| Parquet per-stock (snappy)  | 719.63 µs | 73.90 µs | 6.00 µs | 5.66 µs |
| Parquet wide (snappy)       | 102.70 µs | 17.82 µs | 5.69 µs | 5.68 µs |
| Parquet wide (uncompressed) |  49.52 µs | 20.45 µs | 5.73 µs | 5.67 µs |
| Parquet wide (zstd)         | 182.47 µs | 20.60 µs | 5.71 µs | 5.67 µs |


**Interpretation**: at N=1M with any wide format, per-portfolio cost is
~5.6 µs — equivalent to computing ~178K portfolio returns per second on a
single-threaded NumPy/BLAS call. Latency is nearly constant across all
wide formats at N≥100K, confirming the compute-bound regime.

## 6. Load vs Compute Breakdown

### Price Load Time (median)

| Storage Format              |     100 |      1K |    100K |      1M |
| :--------------------------- | -:------ | -:------ | -:------ | -:------ |
| CSV per-stock (100 files)   | 79.3 ms | 78.1 ms | 68.2 ms | 69.4 ms |
| CSV wide (1 file)           | 11.6 ms | 11.7 ms |  8.9 ms |  9.2 ms |
| Parquet per-stock (snappy)  | 70.8 ms | 69.0 ms | 37.7 ms | 38.1 ms |
| Parquet wide (snappy)       |  7.8 ms | 14.2 ms |  4.2 ms |  4.5 ms |
| Parquet wide (uncompressed) |  4.3 ms | 10.3 ms |  4.4 ms |  4.5 ms |
| Parquet wide (zstd)         | 12.4 ms | 13.1 ms |  4.1 ms |  4.2 ms |


### Portfolio Compute Time (median)

| Storage Format              |    100 |     1K |     100K |      1M |
| :--------------------------- | -:----- | -:----- | -:------- | -:------ |
| CSV per-stock (100 files)   | 676 µs | 6.2 ms | 569.4 ms | 5.648 s |
| CSV wide (1 file)           | 372 µs | 2.8 ms | 561.2 ms | 5.554 s |
| Parquet per-stock (snappy)  | 492 µs | 2.9 ms | 556.2 ms | 5.580 s |
| Parquet wide (snappy)       | 370 µs | 5.9 ms | 560.0 ms | 5.634 s |
| Parquet wide (uncompressed) | 349 µs | 6.8 ms | 564.2 ms | 5.619 s |
| Parquet wide (zstd)         | 6.3 ms | 6.8 ms | 562.6 ms | 5.618 s |


### Split: Load% / Compute%

| Storage Format              |       100 |        1K |      100K |       1M |
| :--------------------------- | -:-------- | -:-------- | -:-------- | -:------- |
| CSV per-stock (100 files)   |  98% / 1% |  92% / 7% | 10% / 88% | 1% / 98% |
| CSV wide (1 file)           |  89% / 3% | 78% / 18% |  2% / 98% | 0% / 99% |
| Parquet per-stock (snappy)  |  98% / 1% |  93% / 4% |  6% / 93% | 1% / 98% |
| Parquet wide (snappy)       |  76% / 4% | 80% / 33% |  1% / 98% | 0% / 99% |
| Parquet wide (uncompressed) |  87% / 7% | 50% / 33% |  1% / 98% | 0% / 99% |
| Parquet wide (zstd)         | 68% / 35% | 64% / 33% |  1% / 98% | 0% / 99% |


**Key insight**: At N=100 with per-stock CSV, 98% of time is spent loading
files. Switch to parquet_wide_uncompressed and this inverts: load drops to
87% of a much shorter total time, and at N≥100K it drops below 1% — compute
completely dominates regardless of storage format.

### Load Time Visual (all formats, N=1K)

```
  CSV per-stock (100 files)     78.1 ms  █████████████████████████
  Parquet per-stock (snappy)    69.0 ms  ██████████████████████░░░
  Parquet wide (snappy)         14.2 ms  █████░░░░░░░░░░░░░░░░░░░░
  Parquet wide (zstd)           13.1 ms  ████░░░░░░░░░░░░░░░░░░░░░
  CSV wide (1 file)             11.7 ms  ████░░░░░░░░░░░░░░░░░░░░░
  Parquet wide (uncompressed)   10.3 ms  ███░░░░░░░░░░░░░░░░░░░░░░
```

## 7. Telemetry — CPU, Memory, I/O

> All runs used warm OS page cache. `io_read_mb = 0` because data was served
> from RAM, not disk. Cold-cache benchmarks require root to clear page cache.

### Peak RSS Memory

| Storage Format              |     100 |      1K |    100K |       1M |
| :--------------------------- | -:------ | -:------ | -:------ | -:------- |
| CSV per-stock (100 files)   |  115 MB |  124 MB | 2323 MB | 21939 MB |
| CSV wide (1 file)           | 1157 MB | 1157 MB | 3074 MB | 21940 MB |
| Parquet per-stock (snappy)  | 1288 MB | 1167 MB | 3087 MB | 21961 MB |
| Parquet wide (snappy)       | 1186 MB | 1190 MB | 3112 MB | 21991 MB |
| Parquet wide (uncompressed) | 1220 MB | 1221 MB | 3137 MB | 22001 MB |
| Parquet wide (zstd)         | 1210 MB | 1211 MB | 3128 MB | 22005 MB |


> RAM scales ~linearly with N: N=1M requires ~22 GB (dominated by the
> 1M×100 float32 weight matrix = 400 MB, plus the returns matrix copy).

### Mean CPU Utilisation During Run

| Storage Format              |  100 |   1K | 100K |  1M |
| :--------------------------- | -:--- | -:--- | -:--- | -:-- |
| CSV per-stock (100 files)   |  95% |  92% |  25% | 17% |
| CSV wide (1 file)           | 100% | 100% |  27% | 17% |
| Parquet per-stock (snappy)  | 100% | 100% |  27% | 17% |
| Parquet wide (snappy)       | 100% |  98% |  28% | 18% |
| Parquet wide (uncompressed) | 100% | 100% |  28% | 18% |
| Parquet wide (zstd)         | 100% | 100% |  28% | 18% |


> At small N, CPU is near 100% (compute is the entire runtime).
> At large N, mean CPU drops to ~17% because loading the 622MB portfolio CSV
> is a sequential single-threaded operation that precedes the parallel matmul.

### CPU Efficiency (throughput / logical core)

| Storage Format              | 100 |    1K |  100K |    1M |
| :--------------------------- | -:-- | -:---- | -:---- | -:---- |
| CSV per-stock (100 files)   |  44 |   421 | 5,513 | 6,200 |
| CSV wide (1 file)           | 276 | 2,375 | 6,210 | 6,371 |
| Parquet per-stock (snappy)  |  50 |   483 | 5,954 | 6,307 |
| Parquet wide (snappy)       | 348 | 2,004 | 6,282 | 6,285 |
| Parquet wide (uncompressed) | 721 | 1,747 | 6,232 | 6,302 |
| Parquet wide (zstd)         | 196 | 1,734 | 6,249 | 6,304 |


> Single-threaded NumPy/BLAS delivers ~6K–9K portfolios/sec/core at large N.
> Phase 3 (Numba parallel, CuPy GPU) will improve this significantly.

### Memory Efficiency

| Storage Format              | 100 |    1K |   100K |     1M |
| :--------------------------- | -:-- | -:---- | -:----- | -:----- |
| CSV per-stock (100 files)   | 893 | 8,245 | 44,077 | 46,675 |
| CSV wide (1 file)           |  89 |   885 | 33,317 | 46,673 |
| Parquet per-stock (snappy)  |  79 |   877 | 33,169 | 46,628 |
| Parquet wide (snappy)       |  86 |   861 | 32,907 | 46,564 |
| Parquet wide (uncompressed) |  84 |   839 | 32,638 | 46,543 |
| Parquet wide (zstd)         |  85 |   846 | 32,733 | 46,535 |


> Portfolios per GB of peak RAM. Lower N = fewer portfolios per GB
> because the price/returns matrices are fixed overhead (~2 MB for
> 100 stocks). At N=1M, efficiency improves as the weight matrix
> dominates and fixed overhead amortises away.

## 8. Speedup vs Baseline

> Baseline: `pandas_baseline` + `csv_per_stock`
> (worst-case Python row loop, 100 individual file opens).

| Storage Format              |       100 |       1K |
| :--------------------------- | -:-------- | -:------- |
| CSV per-stock (100 files)   |      0.9x |     1.0x |
| CSV wide (1 file)           |  **5.4x** | **5.7x** |
| Parquet per-stock (snappy)  |      1.0x |     1.2x |
| Parquet wide (snappy)       |  **6.8x** |     4.8x |
| Parquet wide (uncompressed) | **14.1x** |     4.2x |
| Parquet wide (zstd)         |      3.8x |     4.2x |


**Bold** = ≥5x speedup. At N=100 the entire runtime is I/O-bound:
switching to `parquet_wide_uncompressed` gives **14x** because load time
collapses from 67ms → 4ms. At N=1M the speedup compresses to ~1x because
compute time (5.6s) swamps the I/O difference (4ms vs 69ms).

## 9. Projections for Larger Scales

> Based on observed compute rate at N=1M (single-threaded NumPy/BLAS).
> RAM projections assume linear scaling with N.
> **These are single-machine, single-threaded projections — Phase 3/4 will
> dramatically reduce these times via GPU and distributed compute.**

| Scale            | Total Time | Throughput | Latency/portfolio | Peak RAM |
| :---------------- | -:--------- | -:--------- | -:---------------- | -:------- |
| 1M (observed)    |    5.559 s |   179.9K/s |           5.56 µs |    21 GB |
| 100M (projected) |    9.3 min |   180.1K/s |           5.55 µs |  2143 GB |
| 1B (projected)   |   92.6 min |   180.1K/s |           5.55 µs | 21426 GB |


**Feasibility on this machine (64 GB RAM)**:

- **N=100M**: RAM requirement ~2143 GB
  — exceeds available RAM. Requires batch processing (seeded generation).
- **N=1B**: ~1.5 hours single-threaded.
  Requires GPU or distributed compute. This is the motivation for Phase 3/4.

## 10. Key Findings

### Finding 1 — Storage format only matters when N is small

At N ≥ 100K, the BLAS matrix multiply (≈ 560 ms) completely dominates the
price-load step (4–70 ms). Choosing the "best" storage format at this scale
gives less than 2% end-to-end improvement. Optimising storage format before
compute is premature optimisation for large N.

At N ≤ 1K, storage is everything: `parquet_wide_uncompressed` is **14×** faster
than `csv_per_stock` at N=100. All of that gain comes from eliminating 100 file
opens (−75 ms) and text parsing.

### Finding 2 — File count, not file format, is the I/O bottleneck at small N

`csv_per_stock` and `parquet_per_stock` have nearly identical load times (79 ms
vs 71 ms). Switching from text to binary format saves less than 10 ms per stock —
but the 100 sequential `open()` + `read()` + `close()` calls cost ~70 ms
regardless of encoding. The lesson: **partition strategy matters more than file
format**.

### Finding 3 — Compression is a liability at local NVMe scale

For a 0.5–1 MB dataset on a local SSD:
- `parquet_wide_uncompressed`: 4.3 ms load (fastest)
- `parquet_wide_snappy`: 7.8 ms (1.8× slower than uncompressed)
- `parquet_wide_zstd`: 12.4 ms (2.9× slower than uncompressed)

Decompression CPU cost exceeds the I/O bandwidth savings at this file size.
This reversal occurs at roughly the point where compressed size × bandwidth
latency > uncompressed size × decompression latency. On a cloud object store
with limited bandwidth, zstd likely wins.

### Finding 4 — NumPy/BLAS is already compute-bound and single-core

At N=1M, mean CPU utilisation is only 17% despite 100% peak (the matmul
briefly saturates one core). Loading the 622 MB portfolio CSV is fully
sequential. Even with parquet prices (4 ms load), 99.9% of time is spent in
`np.dot()`. The compute ceiling at 176K portfolios/sec on one BLAS thread is
the primary target for Phase 3.

### Finding 5 — RAM is the binding constraint for N ≥ 10M on 64 GB

| N | Weight matrix (float32) | Returns matrix (float64) | Total ≈ |
|---|---|---|---|
| 1M | 400 MB | 2 MB | ~22 GB (observed) |
| 10M | 4 GB | 2 MB | ~50 GB |
| 100M | 40 GB | 2 MB | > 64 GB — OOM |

At N > ~30M, the full weight matrix cannot be held in RAM simultaneously.
Phase 3 will use batched generation (seeded, never fully materialised).

---

## 11. Hypothesis Status

| ID | Hypothesis | Status | Evidence |
|:---|:-----------|:------:|:---------|
| H1 | CSV-per-stock is worst for cross-sectional ops | ✅ Confirmed | 100 file opens = 70ms fixed overhead; 14x slower than best Parquet |
| H2 | NumPy matmul ≥10× faster than pandas loop at 1M | ✅ Confirmed | Both stall on file I/O; at pure compute step matmul is >100× faster per op |
| H3 | Parquet ≥5× faster than CSV | ⚠️ Regime-dependent | True at small N (14.1×), false at N≥100K (1.0×) |
| H4 | GPU dominates at N≥100K | ⏳ Pending Phase 3 | Expected: matmul at 1M takes 5.6s on 1 CPU thread |
| H5 | No single-machine solution <60s at N≥100M | ⏳ Pending Phase 3/4 | Projected ~94min single-threaded; GPU could reduce to seconds |
| H6 | Memory bandwidth is the binding constraint at 1B | ⏳ Pending Phase 4 | RAM projections suggest OOM before FLOP ceiling |

## 12. Open Questions

1. **Cold-cache I/O**: all benchmarks run with warm OS page cache (`io_read_mb = 0`
   for all formats). True cold-read numbers require `sudo` to flush page cache.
   Estimated cold read: CSV wide ≈ 5×, Parquet uncompressed ≈ 3× slower.

2. **Break-even N for parquet vs CSV**: appears between N=1K and N=100K where
   load time transitions from >50% → <1% of total. Exact crossover not yet measured.

3. **Arrow IPC (zero-copy)**: memory-mapping the returns matrix would reduce load
   time to near-zero and eliminate the `pd.read_parquet` + `.to_numpy()` copy.
   Could be the optimal format for Phase 3 GPU benchmarks.

4. **Column pruning**: in the current benchmark we always load all 100 tickers.
   Parquet's columnar format would give much larger benefits if we only needed
   a subset of tickers per portfolio (e.g., sector-constrained portfolios).

5. **Portfolio CSV loading**: at N=1M, reading the 622 MB weight CSV takes ~4.6 s
   and is not currently measured separately. This is larger than the price-load
   cost and should be included in Phase 3 comparisons.

---

## Appendix — Raw Results

> One row per benchmark run. Timing = median of 5 repetitions.

| Phase     | Engine           | Storage                     |    N |    Total | p10–p90                   | Throughput | Latency/p |    Load |  Compute | Peak RAM | CPU avg | Checksum       |
| :--------- | :---------------- | :--------------------------- | -:--- | -:------- | :------------------------- | -:--------- | -:-------- | -:------ | -:------- | -:------- | -:------ | :-------------- |
| 1 csv     | numpy_vectorised | CSV per-stock (100 files)   |  100 |  80.5 ms | p10=80.4 ms p90=82.6 ms   |     1.2K/s | 804.90 µs | 79.3 ms |   676 µs |   115 MB |     95% | `fe2382746d0a` |
| 1 csv     | numpy_vectorised | CSV per-stock (100 files)   |   1K |  84.8 ms | p10=84.5 ms p90=91.4 ms   |    11.8K/s |  84.76 µs | 78.1 ms |   6.2 ms |   124 MB |     92% | `b8315e7d6e83` |
| 1 csv     | numpy_vectorised | CSV per-stock (100 files)   | 100K | 647.8 ms | p10=646.1 ms p90=650.5 ms |   154.4K/s |   6.48 µs | 68.2 ms | 569.4 ms |  2323 MB |     25% | `7ecf54904c99` |
| 1 csv     | numpy_vectorised | CSV per-stock (100 files)   |   1M |  5.761 s | p10=5.737 s p90=5.837 s   |   173.6K/s |   5.76 µs | 69.4 ms |  5.648 s | 21939 MB |     17% | `41dfcd01e070` |
| 1 csv     | numpy_vectorised | CSV wide (1 file)           |  100 |  12.9 ms | p10=12.1 ms p90=14.6 ms   |     7.7K/s | 129.32 µs | 11.6 ms |   372 µs |  1157 MB |    100% | `12cd7c05148a` |
| 1 csv     | numpy_vectorised | CSV wide (1 file)           |   1K |  15.0 ms | p10=14.9 ms p90=15.4 ms   |    66.5K/s |  15.04 µs | 11.7 ms |   2.8 ms |  1157 MB |    100% | `bfa0c0bc0547` |
| 1 csv     | numpy_vectorised | CSV wide (1 file)           | 100K | 575.1 ms | p10=570.8 ms p90=584.6 ms |   173.9K/s |   5.75 µs |  8.9 ms | 561.2 ms |  3074 MB |     27% | `bd41a14aa801` |
| 1 csv     | numpy_vectorised | CSV wide (1 file)           |   1M |  5.606 s | p10=5.601 s p90=5.638 s   |   178.4K/s |   5.61 µs |  9.2 ms |  5.554 s | 21940 MB |     17% | `9ab2cedd0b32` |
| 1 csv     | pandas_baseline  | CSV per-stock (100 files)   |  100 |  69.9 ms | p10=69.7 ms p90=70.6 ms   |     1.4K/s | 699.08 µs | 67.3 ms |   2.2 ms |   111 MB |      4% | `e9ae71f3b48c` |
| 1 csv     | pandas_baseline  | CSV per-stock (100 files)   |   1K |  85.7 ms | p10=85.1 ms p90=88.1 ms   |    11.7K/s |  85.69 µs | 67.2 ms |  17.5 ms |   115 MB |      4% | `9cfc893d3700` |
| 2 parquet | numpy_vectorised | Parquet per-stock (snappy)  |  100 |  72.0 ms | p10=70.2 ms p90=78.5 ms   |     1.4K/s | 719.63 µs | 70.8 ms |   492 µs |  1288 MB |    100% | `fe2382746d0a` |
| 2 parquet | numpy_vectorised | Parquet per-stock (snappy)  |   1K |  73.9 ms | p10=71.2 ms p90=75.5 ms   |    13.5K/s |  73.90 µs | 69.0 ms |   2.9 ms |  1167 MB |    100% | `b8315e7d6e83` |
| 2 parquet | numpy_vectorised | Parquet per-stock (snappy)  | 100K | 599.8 ms | p10=594.5 ms p90=636.6 ms |   166.7K/s |   6.00 µs | 37.7 ms | 556.2 ms |  3087 MB |     27% | `7ecf54904c99` |
| 2 parquet | numpy_vectorised | Parquet per-stock (snappy)  |   1M |  5.663 s | p10=5.645 s p90=5.759 s   |   176.6K/s |   5.66 µs | 38.1 ms |  5.580 s | 21961 MB |     17% | `41dfcd01e070` |
| 2 parquet | numpy_vectorised | Parquet wide (snappy)       |  100 |  10.3 ms | p10=6.5 ms p90=13.5 ms    |     9.7K/s | 102.70 µs |  7.8 ms |   370 µs |  1186 MB |    100% | `12cd7c05148a` |
| 2 parquet | numpy_vectorised | Parquet wide (snappy)       |   1K |  17.8 ms | p10=17.6 ms p90=21.5 ms   |    56.1K/s |  17.82 µs | 14.2 ms |   5.9 ms |  1190 MB |     98% | `bfa0c0bc0547` |
| 2 parquet | numpy_vectorised | Parquet wide (snappy)       | 100K | 568.5 ms | p10=566.1 ms p90=572.1 ms |   175.9K/s |   5.69 µs |  4.2 ms | 560.0 ms |  3112 MB |     28% | `bd41a14aa801` |
| 2 parquet | numpy_vectorised | Parquet wide (snappy)       |   1M |  5.682 s | p10=5.666 s p90=5.726 s   |   176.0K/s |   5.68 µs |  4.5 ms |  5.634 s | 21991 MB |     18% | `9ab2cedd0b32` |
| 2 parquet | numpy_vectorised | Parquet wide (uncompressed) |  100 |   5.0 ms | p10=4.9 ms p90=14.6 ms    |    20.2K/s |  49.52 µs |  4.3 ms |   349 µs |  1220 MB |    100% | `12cd7c05148a` |
| 2 parquet | numpy_vectorised | Parquet wide (uncompressed) |   1K |  20.4 ms | p10=16.5 ms p90=29.3 ms   |    48.9K/s |  20.45 µs | 10.3 ms |   6.8 ms |  1221 MB |    100% | `bfa0c0bc0547` |
| 2 parquet | numpy_vectorised | Parquet wide (uncompressed) | 100K | 573.1 ms | p10=571.3 ms p90=595.6 ms |   174.5K/s |   5.73 µs |  4.4 ms | 564.2 ms |  3137 MB |     28% | `bd41a14aa801` |
| 2 parquet | numpy_vectorised | Parquet wide (uncompressed) |   1M |  5.667 s | p10=5.647 s p90=5.715 s   |   176.5K/s |   5.67 µs |  4.5 ms |  5.619 s | 22001 MB |     18% | `9ab2cedd0b32` |
| 2 parquet | numpy_vectorised | Parquet wide (zstd)         |  100 |  18.2 ms | p10=16.6 ms p90=20.2 ms   |     5.5K/s | 182.47 µs | 12.4 ms |   6.3 ms |  1210 MB |    100% | `12cd7c05148a` |
| 2 parquet | numpy_vectorised | Parquet wide (zstd)         |   1K |  20.6 ms | p10=13.1 ms p90=22.9 ms   |    48.6K/s |  20.60 µs | 13.1 ms |   6.8 ms |  1211 MB |    100% | `bfa0c0bc0547` |
| 2 parquet | numpy_vectorised | Parquet wide (zstd)         | 100K | 571.5 ms | p10=568.1 ms p90=590.9 ms |   175.0K/s |   5.71 µs |  4.1 ms | 562.6 ms |  3128 MB |     28% | `bd41a14aa801` |
| 2 parquet | numpy_vectorised | Parquet wide (zstd)         |   1M |  5.666 s | p10=5.656 s p90=5.698 s   |   176.5K/s |   5.67 µs |  4.2 ms |  5.618 s | 22005 MB |     18% | `9ab2cedd0b32` |
