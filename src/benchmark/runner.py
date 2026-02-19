"""
Benchmark runner — orchestrates timed runs of compute/storage configurations.

Usage (from project root):
    python -m src.benchmark.runner --config bl_001

Or via the CLI script:
    python scripts/run_benchmark.py --scale 1M --storage wide_csv --engine numpy_matmul
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import psutil

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
PRICES_DIR = ROOT / "data" / "raw" / "prices"
PORTFOLIOS_DIR = ROOT / "data" / "raw" / "portfolios"


# ---------------------------------------------------------------------------
# Hardware / software introspection
# ---------------------------------------------------------------------------

def capture_hardware() -> dict:
    cpu_freq = psutil.cpu_freq()
    return {
        "cpu_model": platform.processor() or _read_cpuinfo_model(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "gpu_model": _detect_gpu(),
        "gpu_vram_gb": None,  # populated separately if GPU run
        "storage_type": "nvme",  # update manually if different
    }


def _read_cpuinfo_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.machine()


def _detect_gpu() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out.split("\n")[0] if out else None
    except Exception:
        return None


def capture_software() -> dict:
    sw: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "kernel_version": platform.version(),
        "python_version": platform.python_version(),
        "implementation_version": _git_sha(),
    }
    for pkg, key in [
        ("numpy", "numpy_version"),
        ("pandas", "pandas_version"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(pkg)
            sw[key] = getattr(mod, "__version__", None)
        except ImportError:
            sw[key] = None

    for pkg, key in [
        ("numba", "numba_version"),
        ("cupy", "cupy_version"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(pkg)
            sw[key] = getattr(mod, "__version__", None)
        except ImportError:
            sw[key] = None

    return sw


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

def _peak_rss_mb() -> float:
    """Current process peak RSS in MB."""
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1_048_576


def drop_page_cache() -> None:
    """
    Drop the OS page cache so that disk reads are cold.

    Requires root. If not available, log a warning — benchmarks will be warm.
    """
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        log.info("OS page cache dropped.")
    except PermissionError:
        log.warning(
            "Could not drop page cache (requires root). "
            "Disk-read timings may reflect warm cache. "
            "Run: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
        )
    except FileNotFoundError:
        log.warning("drop_caches not available (non-Linux OS). Skipping.")


# ---------------------------------------------------------------------------
# Single timed run
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that records elapsed wall-clock time."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def timed_run(
    fn: Callable[[], tuple],
    drop_cache: bool = False,
) -> tuple[float, tuple]:
    """
    Execute fn() with wall-clock timing.

    Parameters
    ----------
    fn         : zero-argument callable that returns a result tuple
    drop_cache : whether to drop page cache before running

    Returns
    -------
    elapsed : wall-clock seconds
    result  : whatever fn() returned
    """
    if drop_cache:
        drop_page_cache()

    with Timer() as t:
        result = fn()

    return t.elapsed, result


# ---------------------------------------------------------------------------
# Full benchmark: N repetitions
# ---------------------------------------------------------------------------

def run_benchmark(
    config: dict,
    repetitions: int = 5,
    warmup: bool = True,
    drop_cache: bool = True,
) -> dict:
    """
    Run a benchmark configuration N times and collect timings.

    Parameters
    ----------
    config : benchmark configuration dict. Must contain:
        - implementation : str
        - language        : str
        - storage_format  : str
        - portfolio_scale : int
        - portfolio_k     : int
        - universe_size   : int
        - seed            : int
    repetitions : number of timed runs
    warmup      : if True, run once before timing (allows JIT to compile)
    drop_cache  : if True, drop OS page cache before each timed run

    Returns
    -------
    Full result dict conforming to benchmark_result.schema.json
    """
    from src.compute.baseline import (
        load_returns_from_wide_csv,
        load_returns_from_per_stock_csvs,
        compute_numpy_matmul,
        compute_pandas_row_loop,
        result_checksum,
    )

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    scale = config["portfolio_scale"]
    storage = config["storage_format"]
    engine = config["implementation"]

    portfolio_path = PORTFOLIOS_DIR / f"portfolios_{scale:_}.csv"
    if not portfolio_path.exists():
        raise FileNotFoundError(
            f"Portfolio CSV not found: {portfolio_path}\n"
            f"Run: python scripts/generate_portfolios.py --n {scale}"
        )

    log.info(f"Starting benchmark: {engine} | {storage} | N={scale:,}")

    # --- Load portfolios (outside main timer — done once) ---
    log.info("Loading portfolio weights...")
    port_df = pd.read_csv(portfolio_path, index_col="portfolio_id")

    def _make_run_fn(storage_format: str, engine_name: str):
        """Return a zero-argument callable that performs one full timed run."""
        def run_fn():
            # Phase 1: Load prices
            t0 = time.perf_counter()
            if storage_format == "csv_wide":
                returns, tickers = load_returns_from_wide_csv()
            elif storage_format == "csv_per_stock":
                returns, tickers = load_returns_from_per_stock_csvs()
            else:
                raise ValueError(f"Unknown storage format: {storage_format}")
            t_load = time.perf_counter() - t0

            # Phase 2: Align portfolio weights with ticker order
            t1 = time.perf_counter()
            weights = port_df.reindex(columns=tickers, fill_value=0.0).to_numpy(dtype=np.float32)
            t_align = time.perf_counter() - t1

            # Phase 3: Compute metrics
            t2 = time.perf_counter()
            if engine_name in ("numpy_matmul", "numpy_vectorised"):
                results = compute_numpy_matmul(returns, weights)
            elif engine_name == "pandas_baseline":
                results = compute_pandas_row_loop(returns, weights, tickers)
            else:
                raise ValueError(f"Unknown engine: {engine_name}")
            t_compute = time.perf_counter() - t2

            t_total = t_load + t_align + t_compute
            return results, t_load, t_align, t_compute, t_total

        return run_fn

    run_fn = _make_run_fn(storage, engine)

    # --- Warmup pass (not timed) ---
    if warmup:
        log.info("Warmup run (not timed)...")
        run_fn()

    # --- Timed repetitions ---
    timings_load = []
    timings_compute = []
    timings_total = []
    checksum = None
    peak_ram = 0.0

    for rep in range(repetitions):
        log.info(f"  Repetition {rep + 1}/{repetitions}...")
        elapsed, (results, t_load, t_align, t_compute, t_total) = timed_run(
            run_fn, drop_cache=drop_cache
        )
        timings_load.append(round(t_load, 6))
        timings_compute.append(round(t_compute, 6))
        timings_total.append(round(t_total, 6))
        peak_ram = max(peak_ram, _peak_rss_mb())

        if checksum is None:
            checksum = result_checksum(results)

    throughput = scale / float(np.median(timings_total))

    result_doc = {
        "run_id": run_id,
        "timestamp": started_at,
        "config": {
            **config,
            "repetitions": repetitions,
            "batch_size": None,
        },
        "hardware": capture_hardware(),
        "software": capture_software(),
        "timings_sec": {
            "load_prices": timings_load,
            "compute_metrics": timings_compute,
            "total": timings_total,
        },
        "summary": {
            "median_total_sec": round(float(np.median(timings_total)), 6),
            "p10_total_sec": round(float(np.percentile(timings_total, 10)), 6),
            "p90_total_sec": round(float(np.percentile(timings_total, 90)), 6),
            "throughput_portfolios_per_sec": round(throughput, 2),
            "peak_ram_mb": round(peak_ram, 1),
            "peak_gpu_vram_mb": None,
            "io_bytes_read_mb": None,
            "cpu_utilisation_pct": None,
            "gpu_utilisation_pct": None,
        },
        "result_checksum": checksum,
        "notes": None,
    }

    # --- Persist result ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(result_doc, f, indent=2)
    log.info(f"Result written to {out_path}")
    log.info(
        f"Median: {result_doc['summary']['median_total_sec']:.3f}s | "
        f"Throughput: {throughput:,.0f} portfolios/sec"
    )

    return result_doc
