#!/usr/bin/env python3
"""
CLI: Run a benchmark configuration.

Examples
--------
# Baseline: NumPy matmul, wide CSV, 1M portfolios
python scripts/run_benchmark.py --scale 1M --storage csv_wide --engine numpy_matmul

# Run all Phase 1 configurations (CSV baseline)
python scripts/run_benchmark.py --phase 1

# Run all Phase 2 configurations (Parquet storage comparison)
python scripts/run_benchmark.py --phase 2

# Run both phases and print combined report
python scripts/run_benchmark.py --phase 1 --phase 2 --report

# Print report of all existing results
python scripts/run_benchmark.py --report
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import runner, report

SCALE_MAP = {
    "100": 100,
    "1K": 1_000,
    "100K": 100_000,
    "1M": 1_000_000,
    "100M": 100_000_000,
    "1B": 1_000_000_000,
}

PHASE1_CONFIGS = [
    # BL-001: Pandas row loop, per-stock CSV (tiny scales only)
    {"implementation": "pandas_baseline", "language": "python",
     "storage_format": "csv_per_stock", "portfolio_scale": 100,
     "portfolio_k": 15, "universe_size": 100, "seed": 42},
    {"implementation": "pandas_baseline", "language": "python",
     "storage_format": "csv_per_stock", "portfolio_scale": 1_000,
     "portfolio_k": 15, "universe_size": 100, "seed": 42},

    # BL-002: NumPy matmul, per-stock CSV
    {"implementation": "numpy_vectorised", "language": "python",
     "storage_format": "csv_per_stock", "portfolio_scale": 100,
     "portfolio_k": 15, "universe_size": 100, "seed": 42},
    {"implementation": "numpy_vectorised", "language": "python",
     "storage_format": "csv_per_stock", "portfolio_scale": 1_000,
     "portfolio_k": 15, "universe_size": 100, "seed": 42},
    {"implementation": "numpy_vectorised", "language": "python",
     "storage_format": "csv_per_stock", "portfolio_scale": 100_000,
     "portfolio_k": 15, "universe_size": 100, "seed": 42},
    {"implementation": "numpy_vectorised", "language": "python",
     "storage_format": "csv_per_stock", "portfolio_scale": 1_000_000,
     "portfolio_k": 15, "universe_size": 100, "seed": 42},

    # BL-003: NumPy matmul, wide CSV
    {"implementation": "numpy_vectorised", "language": "python",
     "storage_format": "csv_wide", "portfolio_scale": 100,
     "portfolio_k": 15, "universe_size": 100, "seed": 42},
    {"implementation": "numpy_vectorised", "language": "python",
     "storage_format": "csv_wide", "portfolio_scale": 1_000,
     "portfolio_k": 15, "universe_size": 100, "seed": 42},
    {"implementation": "numpy_vectorised", "language": "python",
     "storage_format": "csv_wide", "portfolio_scale": 100_000,
     "portfolio_k": 15, "universe_size": 100, "seed": 42},
    {"implementation": "numpy_vectorised", "language": "python",
     "storage_format": "csv_wide", "portfolio_scale": 1_000_000,
     "portfolio_k": 15, "universe_size": 100, "seed": 42},
]

# ---------------------------------------------------------------------------
# Phase 2: Storage optimisation — Parquet (snappy, zstd, uncompressed)
# Same compute engine (numpy_vectorised) as Phase 1 for a clean I/O comparison.
# ---------------------------------------------------------------------------
_BASE = {"implementation": "numpy_vectorised", "language": "python",
         "portfolio_k": 15, "universe_size": 100, "seed": 42}

PHASE2_CONFIGS = []
for _storage in (
    "parquet_per_stock",
    "parquet_wide_snappy",
    "parquet_wide_zstd",
    "parquet_wide_uncompressed",
):
    for _scale in (100, 1_000, 100_000, 1_000_000):
        PHASE2_CONFIGS.append({**_BASE, "storage_format": _storage, "portfolio_scale": _scale})


ALL_PHASES = {1: PHASE1_CONFIGS, 2: PHASE2_CONFIGS}


def _run_phase(phase_num: int, configs: list[dict], args) -> None:
    log = logging.getLogger(__name__)
    log.info(f"Running Phase {phase_num}: {len(configs)} configurations...")
    for cfg in configs:
        try:
            runner.run_benchmark(
                config=cfg,
                repetitions=args.reps,
                warmup=not args.no_warmup,
                drop_cache=not args.no_drop_cache,
            )
        except FileNotFoundError as e:
            log.error(str(e))


def main():
    parser = argparse.ArgumentParser(description="Run portfolio benchmark configurations.")
    parser.add_argument("--scale", choices=list(SCALE_MAP.keys()),
                        help="Portfolio scale shorthand (e.g. 1M)")
    parser.add_argument("--storage",
                        choices=list({
                            "csv_wide", "csv_per_stock",
                            "parquet_wide_snappy", "parquet_wide_zstd",
                            "parquet_wide_uncompressed", "parquet_per_stock",
                            "arrow_ipc", "hdf5",
                        }),
                        default="csv_wide")
    parser.add_argument("--engine",
                        choices=["numpy_vectorised", "numpy_matmul",
                                 "pandas_baseline", "numba_parallel", "cupy_gpu"],
                        default="numpy_vectorised")
    parser.add_argument("--reps", type=int, default=5,
                        help="Number of timed repetitions (default: 5)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup run")
    parser.add_argument("--no-drop-cache", action="store_true",
                        help="Do not drop OS page cache between runs")
    parser.add_argument("--phase", type=int, action="append",
                        choices=[1, 2], dest="phases",
                        help="Run all configurations in a phase (repeatable: --phase 1 --phase 2)")
    parser.add_argument("--report", action="store_true",
                        help="Print aggregated report of all results and exit")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if args.report and not args.phases and not args.scale:
        report.run()
        return

    if args.phases:
        for p in sorted(set(args.phases)):
            _run_phase(p, ALL_PHASES[p], args)
        report.run()
        return

    if not args.scale:
        parser.error("--scale is required (unless using --phase or --report)")

    scale = SCALE_MAP[args.scale]
    config = {
        "implementation": args.engine,
        "language": "python",
        "storage_format": args.storage,
        "portfolio_scale": scale,
        "portfolio_k": 15,
        "universe_size": 100,
        "seed": 42,
    }

    runner.run_benchmark(
        config=config,
        repetitions=args.reps,
        warmup=not args.no_warmup,
        drop_cache=not args.no_drop_cache,
    )


if __name__ == "__main__":
    main()
