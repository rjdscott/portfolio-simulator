#!/usr/bin/env python3
"""CLI: Run data validation checks after fetch_data.py."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import validate

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    report = validate.run()
    sys.exit(0 if report["passed"] else 1)
