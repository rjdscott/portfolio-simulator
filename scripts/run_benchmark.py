#!/usr/bin/env python3
"""
CLI: Run a benchmark configuration.

Examples
--------
# Baseline: NumPy matmul, wide CSV, 1M portfolios
python scripts/run_benchmark.py --scale 1M --storage csv_wide --engine numpy_matmul

# Slow baseline: Pandas row loop, per-stock CSVs, 1K portfolios only
python scripts/run_benchmark.py --scale 1K --storage csv_per_stock --engine pandas_baseline

# Run all Phase 1 configurations
python scripts/run_benchmark.py --phase 1

# Print report of all results
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


def main():
    parser = argparse.ArgumentParser(description="Run portfolio benchmark configurations.")
    parser.add_argument("--scale", choices=list(SCALE_MAP.keys()),
                        help="Portfolio scale shorthand (e.g. 1M)")
    parser.add_argument("--storage",
                        choices=["csv_wide", "csv_per_stock", "csv_long",
                                 "parquet_wide", "arrow_ipc", "hdf5"],
                        default="csv_wide")
    parser.add_argument("--engine",
                        choices=["numpy_matmul", "pandas_baseline",
                                 "numba_parallel", "cupy_gpu"],
                        default="numpy_matmul")
    parser.add_argument("--reps", type=int, default=5,
                        help="Number of timed repetitions (default: 5)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup run")
    parser.add_argument("--no-drop-cache", action="store_true",
                        help="Do not drop OS page cache between runs")
    parser.add_argument("--phase", type=int, choices=[1],
                        help="Run all configurations in a phase")
    parser.add_argument("--report", action="store_true",
                        help="Print aggregated report of all results and exit")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if args.report:
        report.run()
        return

    if args.phase == 1:
        logging.getLogger(__name__).info(
            f"Running all {len(PHASE1_CONFIGS)} Phase 1 configurations..."
        )
        for cfg in PHASE1_CONFIGS:
            try:
                runner.run_benchmark(
                    config=cfg,
                    repetitions=args.reps,
                    warmup=not args.no_warmup,
                    drop_cache=not args.no_drop_cache,
                )
            except FileNotFoundError as e:
                logging.getLogger(__name__).error(str(e))
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
