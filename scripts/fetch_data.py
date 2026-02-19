#!/usr/bin/env python3
"""
CLI: Download S&P 500 price data.

Examples
--------
# Default: top 100 stocks, 2020-01-01 to 2024-12-31
python scripts/fetch_data.py

# Custom date range and universe size
python scripts/fetch_data.py --start 2019-01-01 --end 2024-12-31 --universe-size 100
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import fetch


def main():
    parser = argparse.ArgumentParser(
        description="Download S&P 500 price data for the benchmark universe."
    )
    parser.add_argument("--start", default=fetch.STUDY_START,
                        help=f"Start date YYYY-MM-DD (default: {fetch.STUDY_START})")
    parser.add_argument("--end", default=fetch.STUDY_END,
                        help=f"End date YYYY-MM-DD (default: {fetch.STUDY_END})")
    parser.add_argument("--universe-size", type=int, default=fetch.UNIVERSE_SIZE,
                        help=f"Number of stocks in universe (default: {fetch.UNIVERSE_SIZE})")
    parser.add_argument("--min-completeness", type=float, default=fetch.MIN_COMPLETENESS,
                        help=f"Min data completeness 0-1 (default: {fetch.MIN_COMPLETENESS})")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    fetch.run(
        universe_size=args.universe_size,
        start=args.start,
        end=args.end,
        min_completeness=args.min_completeness,
    )


if __name__ == "__main__":
    main()
