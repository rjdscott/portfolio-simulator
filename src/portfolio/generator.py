"""
Portfolio weight generator.

Generates N random portfolios from a fixed universe, each consisting of K stocks
drawn uniformly at random (without replacement) with Dirichlet(α=1) weights.

The generator is fully reproducible: portfolio j is always generated from
seed = global_seed XOR j, so any portfolio can be reproduced independently
without generating all preceding portfolios.

For N ≤ 10M:
    Generates and writes a dense weight matrix CSV to disk.
    Shape: (N, len(universe)) — zeros for non-selected stocks.

For N > 10M (future):
    Use generate_batch() iteratively. The full matrix is never materialised.

Output
------
data/raw/portfolios/portfolios_<N>.csv
    Columns: portfolio_id, <TICKER_1>, ..., <TICKER_U>
    Rows: N portfolios
    Values: portfolio weight (0.0 for non-selected stocks)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PORTFOLIOS_DIR = ROOT / "data" / "raw" / "portfolios"
META_DIR = ROOT / "data" / "raw" / "metadata"

GLOBAL_SEED = 42
DEFAULT_K = 15          # stocks per portfolio
WRITE_CHUNK = 100_000   # rows per CSV write (controls peak memory during write)


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def _portfolio_weights(
    universe: list[str],
    k: int,
    rng: np.random.Generator,
) -> tuple[list[str], np.ndarray]:
    """
    Draw K stocks from the universe and assign Dirichlet weights.

    Parameters
    ----------
    universe : list of ticker symbols (the full universe)
    k        : number of stocks to include
    rng      : NumPy random generator (caller controls seed)

    Returns
    -------
    selected_tickers : list of K tickers
    weights          : 1-D array of K weights summing to 1.0
    """
    indices = rng.choice(len(universe), size=k, replace=False)
    selected = [universe[i] for i in sorted(indices)]
    # Dirichlet(1, ..., 1) = uniform on the simplex
    weights = rng.dirichlet(np.ones(k))
    return selected, weights


def generate_single(
    portfolio_id: int,
    universe: list[str],
    k: int = DEFAULT_K,
    global_seed: int = GLOBAL_SEED,
) -> dict[str, float]:
    """
    Generate a single portfolio deterministically from its ID.

    Parameters
    ----------
    portfolio_id : integer index (0-based)
    universe     : list of ticker symbols
    k            : stocks per portfolio
    global_seed  : global seed mixed with portfolio_id

    Returns
    -------
    dict mapping ticker → weight for the K selected stocks.
    All other tickers implicitly have weight 0.
    """
    # XOR global seed with portfolio_id for independent reproducibility
    seed = (global_seed ^ portfolio_id) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    tickers, weights = _portfolio_weights(universe, k, rng)
    return dict(zip(tickers, weights))


def generate_batch(
    start_id: int,
    end_id: int,
    universe: list[str],
    k: int = DEFAULT_K,
    global_seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of portfolios as a dense weight matrix.

    Parameters
    ----------
    start_id : first portfolio ID (inclusive)
    end_id   : last portfolio ID (exclusive)
    universe : list of ticker symbols (length U)
    k        : stocks per portfolio
    global_seed : global seed

    Returns
    -------
    ids     : 1-D int array of shape (batch_size,)
    weights : 2-D float32 array of shape (batch_size, U)
              Row i corresponds to portfolio start_id + i.
              Non-selected stocks have weight 0.0.
    """
    U = len(universe)
    batch_size = end_id - start_id
    weight_matrix = np.zeros((batch_size, U), dtype=np.float32)

    ticker_to_idx = {t: i for i, t in enumerate(universe)}

    for local_i in range(batch_size):
        pid = start_id + local_i
        seed = (global_seed ^ pid) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        indices = rng.choice(U, size=k, replace=False)
        weights = rng.dirichlet(np.ones(k)).astype(np.float32)
        weight_matrix[local_i, indices] = weights

    ids = np.arange(start_id, end_id, dtype=np.int64)
    return ids, weight_matrix


# ---------------------------------------------------------------------------
# Streaming generator (memory-efficient iteration)
# ---------------------------------------------------------------------------

def iter_batches(
    n: int,
    universe: list[str],
    k: int = DEFAULT_K,
    global_seed: int = GLOBAL_SEED,
    batch_size: int = WRITE_CHUNK,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield (ids, weight_matrix) batches for all N portfolios.

    Memory usage per batch: batch_size × U × 4 bytes (float32).
    For batch_size=100K and U=100: ~40 MB per batch.
    """
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        yield generate_batch(start, end, universe, k=k, global_seed=global_seed)
        start = end


# ---------------------------------------------------------------------------
# Materialise to disk (CSV — worst format baseline)
# ---------------------------------------------------------------------------

def materialise_to_csv(
    n: int,
    universe: list[str],
    k: int = DEFAULT_K,
    global_seed: int = GLOBAL_SEED,
    chunk_size: int = WRITE_CHUNK,
) -> Path:
    """
    Generate N portfolios and write them to a dense CSV weight matrix.

    This is the baseline (worst format) storage — CSV requires full
    parse/serialise on every read.

    File layout:
        portfolio_id, AAPL, MSFT, ..., <U tickers>
        0,            0.0,  0.12, ..., 0.0

    Parameters
    ----------
    n         : number of portfolios
    universe  : list of universe ticker symbols (length U)
    k         : stocks per portfolio
    global_seed : global seed for reproducibility
    chunk_size  : rows per write batch (controls peak RAM)

    Returns
    -------
    Path to the written CSV file.
    """
    PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PORTFOLIOS_DIR / f"portfolios_{n:_}.csv"

    log.info(f"Generating {n:,} portfolios (K={k}) from universe of {len(universe)} stocks...")
    log.info(f"Output: {out_path}")

    header = ["portfolio_id"] + universe
    total_batches = (n + chunk_size - 1) // chunk_size

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with tqdm(total=n, desc=f"Generating portfolios", unit=" portfolios") as pbar:
            for ids, weights in iter_batches(n, universe, k, global_seed, chunk_size):
                batch_size = len(ids)
                # Concatenate [id | weights] as object array for CSV writing
                id_col = ids.reshape(-1, 1).astype(object)
                weight_cols = weights.astype(object)
                rows = np.concatenate([id_col, weight_cols], axis=1)
                writer.writerows(rows)
                pbar.update(batch_size)

    size_mb = out_path.stat().st_size / 1_048_576
    log.info(f"Portfolio CSV written: {size_mb:.1f} MB ({n:,} rows × {len(universe) + 1} cols)")
    return out_path


# ---------------------------------------------------------------------------
# Load utilities (for use by benchmark runners)
# ---------------------------------------------------------------------------

def load_portfolio_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a materialised portfolio CSV into (ids, weight_matrix).

    Returns
    -------
    ids    : int64 array of shape (N,)
    weights: float32 array of shape (N, U)
    """
    df = pd.read_csv(path, index_col="portfolio_id")
    ids = df.index.to_numpy(dtype=np.int64)
    weights = df.to_numpy(dtype=np.float32)
    return ids, weights


def load_universe_tickers(meta_dir: Path = META_DIR) -> list[str]:
    """Load the universe ticker list from the committed universe CSV."""
    path = meta_dir / "universe.csv"
    if not path.exists():
        raise FileNotFoundError(f"universe.csv not found at {path}. Run fetch.py first.")
    df = pd.read_csv(path)
    return df["ticker"].tolist()


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate portfolio weight matrix CSV")
    parser.add_argument("--n", type=int, default=1_000_000,
                        help="Number of portfolios to generate (default: 1,000,000)")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"Stocks per portfolio (default: {DEFAULT_K})")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED,
                        help=f"Global random seed (default: {GLOBAL_SEED})")
    parser.add_argument("--chunk", type=int, default=WRITE_CHUNK,
                        help=f"Rows per write batch (default: {WRITE_CHUNK:,})")
    args = parser.parse_args()

    universe = load_universe_tickers()
    log.info(f"Loaded universe: {len(universe)} tickers")

    materialise_to_csv(
        n=args.n,
        universe=universe,
        k=args.k,
        global_seed=args.seed,
        chunk_size=args.chunk,
    )
