"""
Data validation module.

Checks price data completeness, detects anomalies, and writes a validation
report to data/raw/metadata/validation_report.json.

Run after fetch.py. Any validation failures are logged as warnings and
recorded in the report but do not halt execution (the benchmark is designed
to work on the already-filtered universe).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PRICES_DIR = RAW_DIR / "prices"
META_DIR = RAW_DIR / "metadata"


def load_wide_prices() -> pd.DataFrame:
    path = PRICES_DIR / "prices_wide.csv"
    if not path.exists():
        raise FileNotFoundError(f"prices_wide.csv not found at {path}. Run fetch.py first.")
    df = pd.read_csv(path, index_col="date", parse_dates=False)
    return df


def check_missing_values(prices: pd.DataFrame) -> dict:
    missing_pct = (prices.isna().mean() * 100).round(4)
    tickers_with_missing = missing_pct[missing_pct > 0].to_dict()
    return {
        "any_missing": bool(tickers_with_missing),
        "tickers_with_missing": tickers_with_missing,
        "max_missing_pct": float(missing_pct.max()),
    }


def check_zero_prices(prices: pd.DataFrame) -> dict:
    """Detect any zero or negative prices (data error indicator)."""
    zero_or_neg = (prices <= 0).any()
    bad_tickers = zero_or_neg[zero_or_neg].index.tolist()
    return {"tickers_with_zero_or_negative": bad_tickers}


def check_extreme_daily_moves(prices: pd.DataFrame, threshold: float = 0.50) -> dict:
    """Flag single-day price moves exceeding `threshold` (default 50%)."""
    returns = prices.pct_change().abs()
    extreme = (returns > threshold).any()
    flagged = extreme[extreme].index.tolist()
    details = {}
    for ticker in flagged:
        dates = returns.index[returns[ticker] > threshold].tolist()
        details[ticker] = [str(d) for d in dates]
    return {
        "threshold_pct": threshold * 100,
        "tickers_with_extreme_moves": flagged,
        "details": details,
    }


def check_price_continuity(prices: pd.DataFrame) -> dict:
    """Check that trading day index is monotonically increasing with no gaps."""
    if len(prices) == 0:
        return {"ok": True}
    index = pd.to_datetime(prices.index)
    is_monotonic = index.is_monotonic_increasing
    return {
        "index_monotonic_increasing": bool(is_monotonic),
        "n_trading_days": len(prices),
        "first_date": str(prices.index[0]),
        "last_date": str(prices.index[-1]),
    }


def check_return_statistics(prices: pd.DataFrame) -> dict:
    """Summary statistics of log returns for sanity checking."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    stats = {
        "mean_daily_log_return": float(log_returns.mean().mean()),
        "std_daily_log_return": float(log_returns.std().mean()),
        "min_daily_log_return": float(log_returns.min().min()),
        "max_daily_log_return": float(log_returns.max().max()),
        "annualised_vol_avg": float(log_returns.std().mean() * np.sqrt(252)),
    }
    return stats


def run() -> dict:
    """
    Run all validation checks on the downloaded price data.

    Returns the validation report as a dict and writes it to
    data/raw/metadata/validation_report.json.
    """
    log.info("Running data validation...")
    prices = load_wide_prices()
    universe_tickers = prices.columns.tolist()

    report = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "universe_size": len(universe_tickers),
        "tickers": universe_tickers,
        "checks": {},
    }

    checks = {
        "missing_values": check_missing_values(prices),
        "zero_or_negative_prices": check_zero_prices(prices),
        "extreme_daily_moves": check_extreme_daily_moves(prices),
        "price_continuity": check_price_continuity(prices),
        "return_statistics": check_return_statistics(prices),
    }

    report["checks"] = checks

    # Summarise warnings
    warnings = []
    if checks["missing_values"]["any_missing"]:
        warnings.append(f"Missing values detected: {checks['missing_values']['tickers_with_missing']}")
    if checks["zero_or_negative_prices"]["tickers_with_zero_or_negative"]:
        warnings.append(f"Zero/negative prices: {checks['zero_or_negative_prices']['tickers_with_zero_or_negative']}")
    if checks["extreme_daily_moves"]["tickers_with_extreme_moves"]:
        warnings.append(f"Extreme daily moves (>50%): {checks['extreme_daily_moves']['tickers_with_extreme_moves']}")
    if not checks["price_continuity"]["index_monotonic_increasing"]:
        warnings.append("Price index is not monotonically increasing")

    report["warnings"] = warnings
    report["passed"] = len(warnings) == 0

    for w in warnings:
        log.warning(w)

    if report["passed"]:
        log.info("All validation checks passed.")
    else:
        log.warning(f"{len(warnings)} validation warning(s). See report for details.")

    # Write report
    META_DIR.mkdir(parents=True, exist_ok=True)
    path = META_DIR / "validation_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Validation report written to {path}")

    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run()
