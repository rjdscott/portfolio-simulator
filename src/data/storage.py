"""
Storage layer — format conversion and loaders for all benchmark storage formats.

Supported formats
-----------------
CSV (baseline / worst case):
    csv_per_stock   — one file per ticker, text, no compression
    csv_wide        — single wide file (date rows × ticker columns)
    csv_long        — single tidy file (date, ticker, adj_close)

Parquet (Phase 2):
    parquet_per_stock           — one Parquet file per ticker (snappy)
    parquet_wide_snappy         — single wide Parquet, snappy compression
    parquet_wide_zstd           — single wide Parquet, zstd compression
    parquet_wide_uncompressed   — single wide Parquet, no compression

Arrow IPC (Phase 2.5):
    arrow_ipc     — Apache Arrow IPC file format (zero-copy mmap, uncompressed)

Future formats (Phase 2+):
    hdf5          — HDF5 (traditional quant format)
    zarr          — chunked, cloud-native

Each loader returns:
    returns : np.ndarray of shape (T, U) — log returns, float64
    tickers : list[str] — ticker symbols in column order
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PRICES_DIR = RAW_DIR / "prices"
PARQUET_DIR = ROOT / "data" / "parquet"

StorageFormat = Literal[
    "csv_per_stock",
    "csv_wide",
    "csv_long",
    "parquet_per_stock",
    "parquet_wide_snappy",
    "parquet_wide_zstd",
    "parquet_wide_uncompressed",
    "arrow_ipc",
]


# ---------------------------------------------------------------------------
# Shared: log-returns from prices array
# ---------------------------------------------------------------------------

def _log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from a 2-D price array (T, U). Drops first NaN row."""
    return np.log(prices[1:] / prices[:-1])


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_csv_wide(path: Path | None = None) -> tuple[np.ndarray, list[str]]:
    if path is None:
        path = PRICES_DIR / "prices_wide.csv"
    df = pd.read_csv(path, index_col="date")
    tickers = df.columns.tolist()
    returns = _log_returns(df.to_numpy(dtype=np.float64))
    log.debug(f"csv_wide loaded: {returns.shape} from {path.stat().st_size / 1e6:.2f} MB")
    return returns, tickers


def load_csv_per_stock(prices_dir: Path | None = None) -> tuple[np.ndarray, list[str]]:
    if prices_dir is None:
        prices_dir = PRICES_DIR
    csv_files = sorted(f for f in prices_dir.glob("*.csv")
                       if f.stem not in ("prices_wide", "prices_long"))
    if not csv_files:
        raise FileNotFoundError(f"No per-stock CSVs in {prices_dir}")
    log.debug(f"csv_per_stock: loading {len(csv_files)} files from {prices_dir}")
    series = []
    tickers = []
    for f in csv_files:
        df = pd.read_csv(f, index_col="date")
        series.append(df["adj_close"].values)
        tickers.append(f.stem)
    prices = np.column_stack(series).astype(np.float64)
    return _log_returns(prices), tickers


# ---------------------------------------------------------------------------
# Parquet conversion
# ---------------------------------------------------------------------------

def convert_wide_to_parquet(
    src: Path | None = None,
    compression: Literal["snappy", "zstd", "none"] = "snappy",
    row_group_size: int = 65_536,
) -> Path:
    """
    Convert the wide-format price CSV to a Parquet file.

    Parameters
    ----------
    src             : path to prices_wide.csv (default: data/raw/prices/prices_wide.csv)
    compression     : Parquet compression codec
    row_group_size  : rows per row group (controls seek granularity)

    Returns the output Parquet file path.
    """
    if src is None:
        src = PRICES_DIR / "prices_wide.csv"

    codec_suffix = "uncompressed" if compression == "none" else compression
    out_path = PARQUET_DIR / f"prices_wide_{codec_suffix}.parquet"
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Converting wide CSV → Parquet ({compression}): {out_path}")
    df = pd.read_csv(src, index_col="date")
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(
        table,
        out_path,
        compression=compression if compression != "none" else None,
        row_group_size=row_group_size,
        write_statistics=True,
        use_dictionary=False,   # disable dict-encoding for float columns
    )
    in_mb = src.stat().st_size / 1_048_576
    out_mb = out_path.stat().st_size / 1_048_576
    ratio = in_mb / out_mb
    log.info(f"  CSV: {in_mb:.2f} MB  →  Parquet: {out_mb:.2f} MB  (ratio {ratio:.1f}x)")
    return out_path


def convert_per_stock_to_parquet(
    prices_dir: Path | None = None,
    compression: Literal["snappy", "zstd", "none"] = "snappy",
) -> Path:
    """
    Convert per-stock price CSVs to per-stock Parquet files.

    Returns the directory containing the Parquet files.
    """
    if prices_dir is None:
        prices_dir = PRICES_DIR

    out_dir = PARQUET_DIR / "per_stock"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(f for f in prices_dir.glob("*.csv")
                       if f.stem not in ("prices_wide", "prices_long"))
    log.info(f"Converting {len(csv_files)} per-stock CSVs → Parquet ({compression}): {out_dir}")

    total_in_mb = 0.0
    total_out_mb = 0.0
    for f in csv_files:
        df = pd.read_csv(f, index_col="date")
        table = pa.Table.from_pandas(df, preserve_index=True)
        out_path = out_dir / f"{f.stem}.parquet"
        pq.write_table(
            table, out_path,
            compression=compression if compression != "none" else None,
            use_dictionary=False,
        )
        total_in_mb += f.stat().st_size / 1_048_576
        total_out_mb += out_path.stat().st_size / 1_048_576

    ratio = total_in_mb / total_out_mb
    log.info(f"  Total CSV: {total_in_mb:.2f} MB  →  Parquet: {total_out_mb:.2f} MB  (ratio {ratio:.1f}x)")
    return out_dir


def convert_wide_to_arrow_ipc(
    src: Path | None = None,
) -> Path:
    """
    Convert the wide-format price CSV to an Arrow IPC file (uncompressed).

    The Arrow IPC file format supports memory-mapping (zero-copy read) via
    pa.memory_map(). The file is written to data/parquet/prices_wide.arrow.

    Returns the output Arrow IPC file path.
    """
    if src is None:
        src = PRICES_DIR / "prices_wide.csv"

    out_path = PARQUET_DIR / "prices_wide.arrow"
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Converting wide CSV → Arrow IPC (uncompressed): {out_path}")
    df = pd.read_csv(src, index_col="date")
    table = pa.Table.from_pandas(df, preserve_index=True)

    with pa_ipc.new_file(str(out_path), table.schema) as writer:
        writer.write_table(table)

    in_mb = src.stat().st_size / 1_048_576
    out_mb = out_path.stat().st_size / 1_048_576
    ratio = in_mb / out_mb
    log.info(f"  CSV: {in_mb:.2f} MB  →  Arrow IPC: {out_mb:.2f} MB  (ratio {ratio:.1f}x)")
    return out_path


def convert_all(
    compressions: list[str] | None = None,
    arrow: bool = False,
) -> None:
    """Convert price data to all configured Parquet variants (and optionally Arrow IPC)."""
    if compressions is None:
        compressions = ["snappy", "zstd", "none"]

    log.info("=== Phase 2: Parquet conversion ===")
    for codec in compressions:
        convert_wide_to_parquet(compression=codec)

    # Per-stock: snappy only (compression differences are less interesting per-file)
    convert_per_stock_to_parquet(compression="snappy")

    if arrow:
        log.info("=== Phase 2.5: Arrow IPC conversion ===")
        convert_wide_to_arrow_ipc()

    log.info("All conversions complete.")
    _print_parquet_manifest()


def _print_parquet_manifest() -> None:
    """Log a summary of all Parquet and Arrow IPC files created."""
    log.info("\n--- Data file manifest ---")
    total_mb = 0.0
    for pattern in ("*.parquet", "*.arrow"):
        for f in sorted(PARQUET_DIR.rglob(pattern)):
            mb = f.stat().st_size / 1_048_576
            total_mb += mb
            log.info(f"  {f.relative_to(ROOT)}: {mb:.2f} MB")
    log.info(f"  Total: {total_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Parquet loaders
# ---------------------------------------------------------------------------

def load_parquet_wide(
    compression: Literal["snappy", "zstd", "uncompressed"] = "snappy",
) -> tuple[np.ndarray, list[str]]:
    """Load wide-format Parquet into a returns matrix."""
    codec = "uncompressed" if compression == "none" else compression
    path = PARQUET_DIR / f"prices_wide_{codec}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {path}. Run: python scripts/convert_to_parquet.py"
        )
    # Use pyarrow directly for zero-copy-friendly read
    table = pq.read_table(path)
    df = table.to_pandas()
    if "date" in df.columns:
        df = df.set_index("date")
    tickers = df.columns.tolist()
    returns = _log_returns(df.to_numpy(dtype=np.float64))
    log.debug(f"parquet_wide_{codec} loaded: {returns.shape} from {path.stat().st_size / 1e6:.2f} MB")
    return returns, tickers


def load_parquet_per_stock() -> tuple[np.ndarray, list[str]]:
    """Load per-stock Parquet files and assemble the returns matrix."""
    per_stock_dir = PARQUET_DIR / "per_stock"
    if not per_stock_dir.exists():
        raise FileNotFoundError(
            f"Per-stock Parquet dir not found: {per_stock_dir}. "
            "Run: python scripts/convert_to_parquet.py"
        )
    pq_files = sorted(per_stock_dir.glob("*.parquet"))
    if not pq_files:
        raise FileNotFoundError(f"No .parquet files in {per_stock_dir}")

    log.debug(f"parquet_per_stock: loading {len(pq_files)} files")
    series = []
    tickers = []
    for f in pq_files:
        table = pq.read_table(f)
        col = table.column("adj_close").to_pylist()
        series.append(np.array(col, dtype=np.float64))
        tickers.append(f.stem)

    prices = np.column_stack(series)
    return _log_returns(prices), tickers


# ---------------------------------------------------------------------------
# Arrow IPC loader
# ---------------------------------------------------------------------------

def load_arrow_ipc() -> tuple[np.ndarray, list[str]]:
    """
    Load the wide-format price data from an Arrow IPC file using memory-mapping.

    Memory-mapping (pa.memory_map) allows Arrow to read data directly from the
    OS virtual-memory mapping without an extra buffer copy — the zero-copy path.
    """
    path = PARQUET_DIR / "prices_wide.arrow"
    if not path.exists():
        raise FileNotFoundError(
            f"Arrow IPC file not found: {path}. "
            "Run: python scripts/convert_to_parquet.py --arrow"
        )
    mmap = pa.memory_map(str(path), "r")
    table = pa_ipc.open_file(mmap).read_all()
    df = table.to_pandas()
    if "date" in df.columns:
        df = df.set_index("date")
    tickers = df.columns.tolist()
    returns = _log_returns(df.to_numpy(dtype=np.float64))
    log.debug(f"arrow_ipc loaded: {returns.shape} from {path.stat().st_size / 1e6:.2f} MB")
    return returns, tickers


# ---------------------------------------------------------------------------
# Unified loader dispatch
# ---------------------------------------------------------------------------

def load_returns(
    storage: StorageFormat,
    **kwargs,
) -> tuple[np.ndarray, list[str]]:
    """
    Load the returns matrix from the specified storage format.

    Parameters
    ----------
    storage : storage format identifier (see module docstring)

    Returns
    -------
    returns : (T, U) float64 ndarray of log returns
    tickers : list of U ticker symbols
    """
    dispatch = {
        "csv_wide":                  load_csv_wide,
        "csv_per_stock":             load_csv_per_stock,
        "parquet_wide_snappy":       lambda: load_parquet_wide("snappy"),
        "parquet_wide_zstd":         lambda: load_parquet_wide("zstd"),
        "parquet_wide_uncompressed": lambda: load_parquet_wide("uncompressed"),
        "parquet_per_stock":         load_parquet_per_stock,
        "arrow_ipc":                 load_arrow_ipc,
    }
    if storage not in dispatch:
        raise ValueError(
            f"Unknown storage format: '{storage}'. "
            f"Valid options: {list(dispatch)}"
        )
    return dispatch[storage]()


# ---------------------------------------------------------------------------
# Storage size reporter
# ---------------------------------------------------------------------------

def storage_size_report() -> dict[str, float]:
    """Return a dict of {storage_label: size_mb} for all available data files."""
    report: dict[str, float] = {}

    def _mb(p: Path) -> float:
        return p.stat().st_size / 1_048_576 if p.exists() else 0.0

    # CSV
    report["csv_per_stock_total"] = sum(
        f.stat().st_size for f in PRICES_DIR.glob("*.csv")
        if f.stem not in ("prices_wide", "prices_long")
    ) / 1_048_576
    report["csv_wide"] = _mb(PRICES_DIR / "prices_wide.csv")
    report["csv_long"] = _mb(PRICES_DIR / "prices_long.csv")

    # Parquet
    for codec in ("snappy", "zstd", "uncompressed"):
        p = PARQUET_DIR / f"prices_wide_{codec}.parquet"
        report[f"parquet_wide_{codec}"] = _mb(p)

    per_stock_dir = PARQUET_DIR / "per_stock"
    if per_stock_dir.exists():
        report["parquet_per_stock_total"] = sum(
            f.stat().st_size for f in per_stock_dir.glob("*.parquet")
        ) / 1_048_576

    # Arrow IPC
    report["arrow_ipc"] = _mb(PARQUET_DIR / "prices_wide.arrow")

    return report
