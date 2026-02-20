#!/usr/bin/env python3
"""
CLI: Convert price data to Parquet format (Phase 2) and Arrow IPC (Phase 2.5).

Creates:
  data/parquet/prices_wide_snappy.parquet
  data/parquet/prices_wide_zstd.parquet
  data/parquet/prices_wide_uncompressed.parquet
  data/parquet/per_stock/<TICKER>.parquet   (snappy)
  data/parquet/prices_wide.arrow            (with --arrow)

Examples
--------
# All Parquet formats (default)
python scripts/convert_to_parquet.py

# Also convert to Arrow IPC
python scripts/convert_to_parquet.py --arrow

# Specific compression only
python scripts/convert_to_parquet.py --compression snappy zstd

# Show storage size comparison
python scripts/convert_to_parquet.py --report
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.storage import convert_all, convert_wide_to_arrow_ipc, storage_size_report


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV price data to Parquet and/or Arrow IPC format."
    )
    parser.add_argument(
        "--compression", nargs="+",
        choices=["snappy", "zstd", "none"],
        default=["snappy", "zstd", "none"],
        help="Compression codec(s) for wide Parquet (default: all three)"
    )
    parser.add_argument(
        "--arrow", action="store_true",
        help="Also convert to Arrow IPC format (data/parquet/prices_wide.arrow)"
    )
    parser.add_argument(
        "--arrow-only", action="store_true",
        help="Convert to Arrow IPC only (skip Parquet)"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print storage size comparison and exit (no conversion)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if args.report:
        sizes = storage_size_report()
        print("\n--- Storage Format Size Comparison ---")
        print(f"{'Format':<40} {'Size (MB)':>10}")
        print("-" * 52)
        for label, mb in sorted(sizes.items()):
            if mb > 0:
                print(f"{label:<40} {mb:>10.2f}")
        print()
        return

    if args.arrow_only:
        convert_wide_to_arrow_ipc()
    else:
        convert_all(compressions=args.compression, arrow=args.arrow)

    print("\n--- Storage Size Report (post-conversion) ---")
    sizes = storage_size_report()
    print(f"{'Format':<40} {'Size (MB)':>10}")
    print("-" * 52)
    for label, mb in sorted(sizes.items()):
        if mb > 0:
            print(f"{label:<40} {mb:>10.2f}")
    print()


if __name__ == "__main__":
    main()
