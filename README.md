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

| Component | Specification |
|-----------|---------------|
| CPU       | Intel Core i7 (20 logical cores) |
| RAM       | 64 GB |
| GPU       | (see `docs/environment.md` for details) |
| OS        | Linux |
| Storage   | Local NVMe SSD |

---

## Data

- **Universe**: S&P 500 constituents (snapshot committed to `data/raw/metadata/`)
- **Price history**: 5 years of daily adjusted-close prices via `yfinance`
- **Baseline storage**: CSV (one file per stock + combined long-format)
- **Date range**: 2020-01-01 → 2024-12-31

---

## Portfolio Definition

Each portfolio is defined by:
- A **random draw** of K stocks (default K=30) from the S&P 500 universe
- **Random weights** summing to 1.0 (Dirichlet-distributed)
- A **global seed** + **portfolio index** for full reproducibility

At scale ≥ 100M, portfolios are generated on-the-fly from seeds and never fully
materialised, making the benchmark memory-bounded rather than disk-bounded.

---

## Benchmark Scales

| Scale       | Portfolios       | Strategy                         |
|-------------|------------------|----------------------------------|
| Tiny        | 100              | Fully materialised CSV           |
| Small       | 1,000            | Fully materialised CSV           |
| Medium      | 100,000          | Fully materialised CSV/Parquet   |
| Large       | 1,000,000        | Materialised (baseline phase)    |
| X-Large     | 100,000,000      | Seeded generation + batching     |
| Extreme     | 1,000,000,000    | Distributed (Spark) + batching   |

---

## Benchmark Phases

### Phase 1 — Baseline (this repository, current)
Storage: CSV on disk (worst case)
Compute: Pandas / NumPy

### Phase 2 — Storage Optimisation
Compare: CSV vs Parquet vs Arrow IPC vs HDF5 vs Zarr

### Phase 3 — Compute Optimisation
Compare: NumPy → Numba (CPU) → CuPy (GPU) → C++ (OpenMP/BLAS)

### Phase 4 — Distributed Scale
PySpark on local cluster vs Dask vs Ray

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch S&P 500 price data (stores to data/raw/)
python scripts/fetch_data.py

# 3. Generate 1M portfolio weight matrix (stores to data/raw/portfolios/)
python scripts/generate_portfolios.py --n 1_000_000

# 4. Run the baseline benchmark
python scripts/run_benchmark.py --scale 1M --engine csv_pandas
```

---

## Repository Structure

```
portfolio-simulator/
├── data/
│   ├── raw/
│   │   ├── prices/          # One CSV per stock (baseline storage)
│   │   ├── metadata/        # S&P 500 constituent list, data manifest
│   │   └── portfolios/      # Generated portfolio weight CSVs
│   ├── parquet/             # Phase 2: Parquet storage
│   ├── arrow/               # Phase 2: Arrow IPC storage
│   └── hdf5/                # Phase 2: HDF5 storage
├── src/
│   ├── data/
│   │   ├── fetch.py         # yfinance data acquisition
│   │   └── validate.py      # Data quality and completeness checks
│   ├── portfolio/
│   │   └── generator.py     # Seeded portfolio generator
│   ├── compute/
│   │   └── baseline.py      # Pandas/NumPy return computation
│   └── benchmark/
│       ├── runner.py        # Benchmark orchestration
│       └── report.py        # Results serialisation and reporting
├── scripts/
│   ├── fetch_data.py        # CLI: download price data
│   ├── generate_portfolios.py  # CLI: generate portfolio weight files
│   └── run_benchmark.py     # CLI: run a benchmark configuration
├── docs/
│   ├── methodology.md       # Experimental design and statistical rigour
│   ├── data-dictionary.md   # Schema definitions for all datasets
│   └── environment.md       # Hardware/software environment capture
├── notebooks/
│   └── 01_data_exploration.ipynb
├── results/                 # Benchmark outputs (JSON + CSV)
├── docker/                  # Spark cluster configs (Phase 4)
├── RESEARCH.md              # Living research journal
└── pyproject.toml
```

---

## Reproducibility

All randomness is seeded. The exact S&P 500 constituent list used is committed to
`data/raw/metadata/sp500_constituents.csv`. Price data can be re-fetched with
`scripts/fetch_data.py --start 2020-01-01 --end 2024-12-31`.

Results are stored in `results/` as timestamped JSON with full configuration
metadata embedded, making them diffable and archivable.

---

## Contributing / Citation

If you use this benchmark in your research, please cite:

```
[Citation to be added upon publication]
```

See `RESEARCH.md` for the living research journal and `docs/methodology.md` for
the full experimental design.
