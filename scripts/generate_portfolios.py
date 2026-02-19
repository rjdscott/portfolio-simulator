#!/usr/bin/env python3
"""
CLI: Generate portfolio weight matrix and store as CSV.

Examples
--------
# 1 million portfolios (default)
python scripts/generate_portfolios.py --n 1_000_000

# 100K portfolios with custom seed
python scripts/generate_portfolios.py --n 100_000 --seed 123

# All benchmark scales
python scripts/generate_portfolios.py --all-scales
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import generator

BENCHMARK_SCALES = [100, 1_000, 100_000, 1_000_000]
# Note: 100M and 1B are NOT materialised — they use seeded generation at runtime


def main():
    parser = argparse.ArgumentParser(
        description="Generate portfolio weight matrix CSV files."
    )
    parser.add_argument("--n", type=lambda x: int(x.replace("_", "")),
                        default=1_000_000,
                        help="Number of portfolios (default: 1_000_000). Use _ as separator.")
    parser.add_argument("--k", type=int, default=generator.DEFAULT_K,
                        help=f"Stocks per portfolio (default: {generator.DEFAULT_K})")
    parser.add_argument("--seed", type=int, default=generator.GLOBAL_SEED,
                        help=f"Global random seed (default: {generator.GLOBAL_SEED})")
    parser.add_argument("--chunk", type=int, default=generator.WRITE_CHUNK,
                        help=f"Write batch size (default: {generator.WRITE_CHUNK:,})")
    parser.add_argument("--all-scales", action="store_true",
                        help=f"Generate CSVs for all materialised scales: {BENCHMARK_SCALES}")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    universe = generator.load_universe_tickers()
    logging.getLogger(__name__).info(f"Universe: {len(universe)} tickers")

    if args.all_scales:
        for scale in BENCHMARK_SCALES:
            generator.materialise_to_csv(
                n=scale, universe=universe, k=args.k,
                global_seed=args.seed, chunk_size=args.chunk,
            )
    else:
        generator.materialise_to_csv(
            n=args.n, universe=universe, k=args.k,
            global_seed=args.seed, chunk_size=args.chunk,
        )


if __name__ == "__main__":
    main()
