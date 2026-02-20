"""
Benchmark runner — orchestrates timed runs of compute/storage configurations.

Each benchmark run collects:
  - Wall-clock time, broken down by phase (load, align, compute)
  - Per-process I/O bytes read (via psutil.Process.io_counters)
  - CPU utilisation (background sampling thread at 100 ms intervals)
  - Peak RSS memory (background sampling thread)
  - Full per-repetition telemetry stored in the result JSON

Usage:
    python scripts/run_benchmark.py --scale 1M --storage parquet_wide_snappy --engine numpy_vectorised
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
PORTFOLIOS_DIR = ROOT / "data" / "raw" / "portfolios"

TELEMETRY_INTERVAL_SEC = 0.025  # CPU / RSS sampling interval (25 ms)


# ---------------------------------------------------------------------------
# Hardware / software introspection
# ---------------------------------------------------------------------------

def capture_hardware() -> dict:
    freq = psutil.cpu_freq()
    return {
        "cpu_model": platform.processor() or _read_cpuinfo_model(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_freq_mhz_max": round(freq.max, 0) if freq else None,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "gpu_model": _detect_gpu(),
        "gpu_vram_gb": None,
        "storage_type": "nvme",
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
        # Parallelism settings for Phase 3 engines
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
    }
    for pkg, key in [
        ("numpy", "numpy_version"),
        ("pandas", "pandas_version"),
        ("pyarrow", "pyarrow_version"),
        ("numba", "numba_version"),
        ("cupy", "cupy_version"),
    ]:
        try:
            mod = __import__(pkg)
            sw[key] = getattr(mod, "__version__", None)
        except ImportError:
            sw[key] = None

    # BLAS implementation
    try:
        import numpy as np
        blas_info = np.__config__.blas_opt_info  # type: ignore[attr-defined]
        sw["blas_implementation"] = str(blas_info.get("libraries", ["unknown"])[0])
    except Exception:
        sw["blas_implementation"] = None

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
# Page cache management
# ---------------------------------------------------------------------------

def drop_page_cache() -> bool:
    """
    Drop the OS page cache (cold-cache benchmark).
    Requires root. Returns True if successful, False otherwise.
    """
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        log.debug("OS page cache dropped (cold read).")
        return True
    except PermissionError:
        log.warning(
            "Cannot drop page cache (requires root) — reads will be warm. "
            "For cold benchmarks: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
        )
        return False
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Telemetry collector
# ---------------------------------------------------------------------------

class TelemetryCollector:
    """
    Collects system telemetry during a benchmark run.

    Spawns a background thread that samples CPU utilisation and process RSS
    at a fixed interval. Also snapshots per-process I/O counters before/after
    to measure actual bytes read from disk.

    Usage:
        tc = TelemetryCollector()
        tc.start()
        # ... run workload ...
        tc.stop()
        summary = tc.summary()
    """

    def __init__(self, interval_sec: float = TELEMETRY_INTERVAL_SEC):
        self._interval = interval_sec
        self._proc = psutil.Process(os.getpid())
        self._cpu_samples: list[float] = []
        self._rss_samples: list[float] = []   # bytes
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._io_start: psutil._common.pio | None = None
        self._io_end: psutil._common.pio | None = None
        self._t_start: float = 0.0
        self._t_end: float = 0.0
        self._cache_dropped: bool = False

    def _sample_loop(self) -> None:
        """Background thread: sample CPU% and RSS at fixed intervals."""
        # prime the cpu_percent call (first call always returns 0.0)
        psutil.cpu_percent(interval=None)
        while not self._stop_event.wait(self._interval):
            self._cpu_samples.append(psutil.cpu_percent(interval=None))
            try:
                self._rss_samples.append(self._proc.memory_info().rss)
            except psutil.NoSuchProcess:
                break

    def start(self, drop_cache: bool = False) -> None:
        """Start telemetry collection. Optionally drop page cache first."""
        if drop_cache:
            self._cache_dropped = drop_page_cache()
        # Snapshot I/O counters
        try:
            self._io_start = self._proc.io_counters()
        except (AttributeError, psutil.AccessDenied):
            self._io_start = None

        # Force an initial CPU% reading (primes the psutil accumulator)
        psutil.cpu_percent(interval=None)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._t_start = time.perf_counter()
        self._thread.start()

    def stop(self) -> None:
        """Stop collection. Call immediately after the workload finishes."""
        self._t_end = time.perf_counter()
        # Force a final sample so very fast runs have at least one data point
        try:
            self._cpu_samples.append(psutil.cpu_percent(interval=None))
            self._rss_samples.append(self._proc.memory_info().rss)
        except Exception:
            pass
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._io_end = self._proc.io_counters()
        except (AttributeError, psutil.AccessDenied):
            self._io_end = None

    @property
    def elapsed_sec(self) -> float:
        return self._t_end - self._t_start

    def summary(self) -> dict:
        """Return a summary dict of all collected telemetry."""
        # I/O
        if self._io_start and self._io_end:
            io_read_mb = (self._io_end.read_bytes - self._io_start.read_bytes) / 1_048_576
            io_read_count = self._io_end.read_count - self._io_start.read_count
        else:
            io_read_mb = None
            io_read_count = None

        # CPU
        cpu_mean = round(float(np.mean(self._cpu_samples)), 1) if self._cpu_samples else None
        cpu_peak = round(float(max(self._cpu_samples)), 1) if self._cpu_samples else None
        cpu_p90 = round(float(np.percentile(self._cpu_samples, 90)), 1) if self._cpu_samples else None

        # RSS (memory)
        if self._rss_samples:
            rss_peak_mb = round(max(self._rss_samples) / 1_048_576, 1)
            rss_mean_mb = round(float(np.mean(self._rss_samples)) / 1_048_576, 1)
        else:
            rss_peak_mb = None
            rss_mean_mb = None

        return {
            "elapsed_sec": round(self.elapsed_sec, 6),
            "wall_clock_ms": round(self.elapsed_sec * 1000, 3),
            "cache_dropped": self._cache_dropped,
            "io_read_mb": round(io_read_mb, 3) if io_read_mb is not None else None,
            "io_read_syscalls": io_read_count,
            "cpu_mean_pct": cpu_mean,
            "cpu_peak_pct": cpu_peak,
            "cpu_p90_pct": cpu_p90,
            "cpu_samples": len(self._cpu_samples),
            "rss_peak_mb": rss_peak_mb,
            "rss_mean_mb": rss_mean_mb,
        }


# ---------------------------------------------------------------------------
# Phase timers (fine-grained breakdown within a single run)
# ---------------------------------------------------------------------------

class PhaseTimer:
    """Records wall-clock elapsed time for named phases within a run."""

    def __init__(self):
        self._phases: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        elapsed = time.perf_counter() - self._starts.pop(name)
        self._phases[name] = round(elapsed, 6)
        return elapsed

    def as_dict(self) -> dict[str, float]:
        return dict(self._phases)


# ---------------------------------------------------------------------------
# Core run function
# ---------------------------------------------------------------------------

def _make_run_fn(storage: str, engine: str, port_df: pd.DataFrame):
    """
    Build a zero-argument callable that executes one full benchmark repetition.

    Returns (results_array, phase_timings_dict).
    """
    from src.data.storage import load_returns
    from src.compute.baseline import (
        compute_numpy_matmul,
        compute_pandas_row_loop,
    )

    def run_fn():
        pt = PhaseTimer()

        # Phase: load prices from storage
        pt.start("load_prices")
        returns, tickers = load_returns(storage)
        pt.stop("load_prices")

        # Phase: align portfolio weights with the loaded ticker order
        pt.start("align_weights")
        weights = port_df.reindex(columns=tickers, fill_value=0.0).to_numpy(dtype=np.float32)
        pt.stop("align_weights")

        # Phase: compute portfolio metrics
        pt.start("compute_metrics")
        if engine in ("numpy_matmul", "numpy_vectorised"):
            results = compute_numpy_matmul(returns, weights)
        elif engine == "pandas_baseline":
            results = compute_pandas_row_loop(returns, weights, tickers)
        elif engine == "numba_parallel":
            from src.compute.numba_parallel import compute_numba_parallel
            results = compute_numba_parallel(returns, weights)
        elif engine == "cpp_openmp":
            from src.compute.cpp_openmp import compute_cpp_openmp
            results = compute_cpp_openmp(returns, weights)
        elif engine == "rust_rayon":
            from src.compute.rust_rayon import compute_rust_rayon
            results = compute_rust_rayon(returns, weights)
        elif engine == "cupy_gpu":
            from src.compute.cupy_gpu import compute_cupy_gpu
            results = compute_cupy_gpu(returns, weights)
        # Phase 3b engines
        elif engine == "polars_engine":
            from src.compute.polars_engine import compute_polars_engine
            results = compute_polars_engine(returns, weights)
        elif engine == "duckdb_sql":
            from src.compute.duckdb_sql import compute_duckdb_sql
            results = compute_duckdb_sql(returns, weights)
        elif engine == "rust_rayon_nightly":
            from src.compute.rust_rayon_nightly import compute_rust_rayon_nightly
            results = compute_rust_rayon_nightly(returns, weights)
        elif engine == "fortran_openmp":
            from src.compute.fortran_openmp import compute_fortran_openmp
            results = compute_fortran_openmp(returns, weights)
        elif engine == "julia_loopvec":
            from src.compute.julia_loopvec import compute_julia_loopvec
            results = compute_julia_loopvec(returns, weights)
        elif engine == "go_goroutines":
            from src.compute.go_goroutines import compute_go_goroutines
            results = compute_go_goroutines(returns, weights)
        elif engine == "java_vector_api":
            from src.compute.java_vector_api import compute_java_vector_api
            results = compute_java_vector_api(returns, weights)
        # Phase 3c engines (C++ micro-arch optimisations)
        elif engine == "cpp_openmp_unroll":
            from src.compute.cpp_openmp_unroll import compute_cpp_openmp_unroll
            results = compute_cpp_openmp_unroll(returns, weights)
        elif engine == "cpp_openmp_tile4":
            from src.compute.cpp_openmp_tile4 import compute_cpp_openmp_tile4
            results = compute_cpp_openmp_tile4(returns, weights)
        elif engine == "cpp_openmp_clang":
            from src.compute.cpp_openmp_clang import compute_cpp_openmp_clang
            results = compute_cpp_openmp_clang(returns, weights)
        elif engine == "numpy_float32":
            from src.compute.numpy_float32 import compute_numpy_float32
            results = compute_numpy_float32(returns, weights)
        elif engine == "pytorch_cpu":
            from src.compute.pytorch_cpu import compute_pytorch_cpu
            results = compute_pytorch_cpu(returns, weights)
        elif engine == "jax_cpu":
            from src.compute.jax_cpu import compute_jax_cpu
            results = compute_jax_cpu(returns, weights)
        elif engine == "cpp_eigen":
            from src.compute.cpp_eigen import compute_cpp_eigen
            results = compute_cpp_eigen(returns, weights)
        elif engine == "rust_faer":
            from src.compute.rust_faer import compute_rust_faer
            results = compute_rust_faer(returns, weights)
        else:
            raise ValueError(f"Unknown engine: '{engine}'")
        pt.stop("compute_metrics")

        phases = pt.as_dict()
        phases["total"] = sum(phases.values())
        return results, phases

    return run_fn


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    config: dict,
    repetitions: int = 5,
    warmup: bool = True,
    drop_cache: bool = True,
) -> dict:
    """
    Run a benchmark configuration N times, collecting full telemetry per rep.

    Parameters
    ----------
    config : dict with keys:
        implementation, language, storage_format, portfolio_scale,
        portfolio_k, universe_size, seed
    repetitions : timed runs (median reported)
    warmup      : untimed pre-run (important for JIT compilers)
    drop_cache  : drop OS page cache before each rep (requires root)

    Returns
    -------
    Full result document (dict), also written to results/<uuid>.json
    """
    from src.compute.baseline import result_checksum

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    scale = config["portfolio_scale"]
    storage = config["storage_format"]
    engine = config["implementation"]

    portfolio_path = PORTFOLIOS_DIR / f"portfolios_{scale:_}.csv"
    if not portfolio_path.exists():
        raise FileNotFoundError(
            f"Portfolio CSV not found: {portfolio_path}\n"
            f"Generate with: python scripts/generate_portfolios.py --n {scale}"
        )

    log.info("=" * 65)
    log.info(f"Benchmark: {engine} | {storage} | N={scale:,} | reps={repetitions}")
    log.info("=" * 65)

    # Load portfolio weights once (not timed — same for all reps)
    log.info(f"Loading portfolio weights from {portfolio_path.name}...")
    port_df = pd.read_csv(portfolio_path, index_col="portfolio_id")
    log.info(f"  Weights shape: {port_df.shape[0]:,} portfolios × {port_df.shape[1]} stocks")

    run_fn = _make_run_fn(storage, engine, port_df)

    # Warmup (not timed): allows JIT compilers and BLAS thread pools to initialise
    if warmup:
        log.info("Warmup run (not timed)...")
        run_fn()

    # Timed repetitions
    per_rep_telemetry: list[dict] = []
    timings_load: list[float] = []
    timings_compute: list[float] = []
    timings_total: list[float] = []
    checksum: str | None = None

    for rep in range(repetitions):
        log.info(f"  Rep {rep + 1}/{repetitions}...")
        tc = TelemetryCollector()
        tc.start(drop_cache=drop_cache)

        results, phases = run_fn()

        tc.stop()
        tel = tc.summary()

        # Record timings
        timings_load.append(phases.get("load_prices", 0.0))
        timings_compute.append(phases.get("compute_metrics", 0.0))
        timings_total.append(phases["total"])

        if checksum is None:
            checksum = result_checksum(results)

        # Store per-rep telemetry
        per_rep_telemetry.append({
            "rep": rep,
            "phases_sec": phases,
            **tel,
        })

        log.info(
            f"    elapsed={phases['total']:.3f}s | "
            f"load={phases.get('load_prices', 0):.3f}s | "
            f"compute={phases.get('compute_metrics', 0):.3f}s | "
            f"io={tel.get('io_read_mb', '?')} MB | "
            f"cpu_peak={tel.get('cpu_peak_pct', '?')}% | "
            f"rss_peak={tel.get('rss_peak_mb', '?')} MB"
        )

    # Aggregate summary across reps
    n_portfolios = scale
    median_total = float(np.median(timings_total))
    throughput = n_portfolios / median_total

    summary = {
        "median_total_sec": round(median_total, 6),
        "p10_total_sec": round(float(np.percentile(timings_total, 10)), 6),
        "p90_total_sec": round(float(np.percentile(timings_total, 90)), 6),
        "median_load_sec": round(float(np.median(timings_load)), 6),
        "median_compute_sec": round(float(np.median(timings_compute)), 6),
        "throughput_portfolios_per_sec": round(throughput, 2),
        "peak_ram_mb": max(
            (r.get("rss_peak_mb") or 0.0) for r in per_rep_telemetry
        ) or None,
        "mean_io_read_mb": _safe_median([r.get("io_read_mb") for r in per_rep_telemetry]),
        "mean_cpu_pct": _safe_median([r.get("cpu_mean_pct") for r in per_rep_telemetry]),
        "peak_cpu_pct": _safe_max([r.get("cpu_peak_pct") for r in per_rep_telemetry]),
        "peak_gpu_vram_mb": None,
        "gpu_utilisation_pct": None,
    }

    result_doc = {
        "run_id": run_id,
        "timestamp": started_at,
        "config": {
            **config,
            "repetitions": repetitions,
            "warmup": warmup,
            "drop_cache_attempted": drop_cache,
            "batch_size": None,
        },
        "hardware": capture_hardware(),
        "software": capture_software(),
        "timings_sec": {
            "load_prices": timings_load,
            "compute_metrics": timings_compute,
            "total": timings_total,
        },
        "telemetry_per_rep": per_rep_telemetry,
        "summary": summary,
        "result_checksum": checksum,
        "notes": None,
    }

    # Persist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(result_doc, f, indent=2, default=_json_default)

    log.info(
        f"  → {run_id[:8]}  "
        f"median={median_total:.3f}s  "
        f"throughput={throughput:,.0f} portfolios/sec  "
        f"io={summary.get('mean_io_read_mb', '?')} MB  "
        f"cpu_peak={summary.get('peak_cpu_pct', '?')}%"
    )
    log.info(f"  Result: {out_path}")

    return result_doc


# ---------------------------------------------------------------------------
# Phase 4: Distributed benchmark entry point
# ---------------------------------------------------------------------------

def run_distributed_benchmark(
    config: dict,
    repetitions: int = 3,
    warmup: bool = True,
    drop_cache: bool = False,
) -> dict:
    """
    Run a distributed compute benchmark (Phase 4: Spark, Dask, Ray).

    Differences from run_benchmark():
    - Portfolio weights are generated on-the-fly (seeded, not from CSV).
    - Price data is always loaded from parquet_wide_uncompressed.
    - align_weights phase is omitted (workers align internally).
    - For N > 10M: returns aggregate_stats dict instead of full results array;
      result_checksum covers first 10,000 portfolios only.

    Parameters
    ----------
    config : dict with keys:
        implementation, language, storage_format, portfolio_scale,
        portfolio_k, universe_size, seed, portfolio_source, batch_size
    repetitions : timed runs (median reported)
    warmup      : untimed pre-run (important for framework startup)
    drop_cache  : drop OS page cache before each rep

    Returns
    -------
    Full result document (dict), also written to results/<uuid>.json
    """
    from src.compute.baseline import result_checksum
    from src.data.storage import load_returns
    from src.portfolio.generator import generate_batch, load_universe_tickers

    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    scale = config["portfolio_scale"]
    engine = config["implementation"]
    k = config.get("portfolio_k", 15)
    seed = config.get("seed", 42)
    batch_size = config.get("batch_size", 100_000)
    storage = "parquet_wide_uncompressed"  # always for Phase 4

    log.info("=" * 65)
    log.info(f"Distributed benchmark: {engine} | N={scale:,} | reps={repetitions}")
    log.info("=" * 65)

    # Load universe tickers (from metadata)
    universe = load_universe_tickers()
    log.info(f"Universe: {len(universe)} tickers")

    # Load price returns once (not timed — same for all reps)
    log.info("Loading price data from parquet_wide_uncompressed...")
    returns, tickers = load_returns(storage)
    log.info(f"  Returns shape: {returns.shape}")

    # Dispatch function for the engine
    def _dispatch(ret, tick, univ, n, k_, seed_, bs):
        if engine == "spark_local":
            from src.compute.spark_local import compute_spark_local
            return compute_spark_local(ret, tick, univ, n, k_, seed_, bs)
        elif engine == "dask_local":
            from src.compute.dask_local import compute_dask_local
            return compute_dask_local(ret, tick, univ, n, k_, seed_, bs)
        elif engine == "ray_local":
            from src.compute.ray_local import compute_ray_local
            return compute_ray_local(ret, tick, univ, n, k_, seed_, bs)
        else:
            raise ValueError(f"Unknown distributed engine: '{engine}'")

    # Warmup: initialise framework and JIT (untimed)
    if warmup:
        log.info("Warmup run (not timed)...")
        # Use a small N for warmup to keep it fast
        warmup_n = min(scale, 10_000)
        _dispatch(returns, tickers, universe, warmup_n, k, seed, batch_size)

    # Timed repetitions
    per_rep_telemetry: list[dict] = []
    timings_load: list[float] = []
    timings_compute: list[float] = []
    timings_total: list[float] = []
    checksum: str | None = None
    aggregate_stats: dict | None = None
    notes: str | None = None

    for rep in range(repetitions):
        log.info(f"  Rep {rep + 1}/{repetitions}...")
        tc = TelemetryCollector()
        tc.start(drop_cache=drop_cache)
        pt = PhaseTimer()

        # Phase: load prices
        pt.start("load_prices")
        _returns, _tickers = load_returns(storage)
        pt.stop("load_prices")

        # Phase: distributed compute (includes portfolio generation)
        pt.start("compute_metrics")
        results = _dispatch(_returns, _tickers, universe, scale, k, seed, batch_size)
        pt.stop("compute_metrics")

        tc.stop()
        tel = tc.summary()

        phases = pt.as_dict()
        phases["total"] = sum(phases.values())

        timings_load.append(phases.get("load_prices", 0.0))
        timings_compute.append(phases.get("compute_metrics", 0.0))
        timings_total.append(phases["total"])

        # Checksum and aggregate stats (on first rep only)
        if checksum is None:
            if isinstance(results, dict):
                # N > 10M: aggregate stats only
                aggregate_stats = results
                # Checksum sample: generate first 10K portfolios
                sample_ids, sample_w = generate_batch(0, 10_000, universe, k=k, global_seed=seed)
                ticker_idx = [universe.index(t) for t in tickers]
                sample_w_aligned = sample_w[:, ticker_idx].astype(np.float64)
                sample_pr = sample_w_aligned @ returns.T
                sample_cum = np.expm1(sample_pr.sum(axis=1))
                _std_r = sample_pr.std(axis=1, ddof=1)
                sample_shr = np.where(
                    _std_r > 0,
                    sample_pr.mean(axis=1) / _std_r * np.sqrt(252),
                    0.0,
                )
                sample_results = np.column_stack([sample_cum, sample_shr])
                checksum = result_checksum(sample_results)
                notes = "checksum_sample_n=10000"
            else:
                checksum = result_checksum(results)

        per_rep_telemetry.append({
            "rep": rep,
            "phases_sec": phases,
            **tel,
        })

        log.info(
            f"    elapsed={phases['total']:.3f}s | "
            f"load={phases.get('load_prices', 0):.3f}s | "
            f"compute={phases.get('compute_metrics', 0):.3f}s | "
            f"io={tel.get('io_read_mb', '?')} MB | "
            f"cpu_peak={tel.get('cpu_peak_pct', '?')}% | "
            f"rss_peak={tel.get('rss_peak_mb', '?')} MB"
        )

    # Aggregate summary
    median_total = float(np.median(timings_total))
    throughput = scale / median_total

    summary = {
        "median_total_sec": round(median_total, 6),
        "p10_total_sec": round(float(np.percentile(timings_total, 10)), 6),
        "p90_total_sec": round(float(np.percentile(timings_total, 90)), 6),
        "median_load_sec": round(float(np.median(timings_load)), 6),
        "median_compute_sec": round(float(np.median(timings_compute)), 6),
        "throughput_portfolios_per_sec": round(throughput, 2),
        "peak_ram_mb": max(
            (r.get("rss_peak_mb") or 0.0) for r in per_rep_telemetry
        ) or None,
        "mean_io_read_mb": _safe_median([r.get("io_read_mb") for r in per_rep_telemetry]),
        "mean_cpu_pct": _safe_median([r.get("cpu_mean_pct") for r in per_rep_telemetry]),
        "peak_cpu_pct": _safe_max([r.get("cpu_peak_pct") for r in per_rep_telemetry]),
        "peak_gpu_vram_mb": None,
        "gpu_utilisation_pct": None,
    }

    result_doc = {
        "run_id": run_id,
        "timestamp": started_at,
        "config": {
            **config,
            "storage_format": storage,
            "repetitions": repetitions,
            "warmup": warmup,
            "drop_cache_attempted": drop_cache,
        },
        "hardware": capture_hardware(),
        "software": capture_software(),
        "timings_sec": {
            "load_prices": timings_load,
            "compute_metrics": timings_compute,
            "total": timings_total,
        },
        "telemetry_per_rep": per_rep_telemetry,
        "summary": summary,
        "result_checksum": checksum,
        "aggregate_stats": aggregate_stats,
        "notes": notes,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(result_doc, f, indent=2, default=_json_default)

    log.info(
        f"  → {run_id[:8]}  "
        f"median={median_total:.3f}s  "
        f"throughput={throughput:,.0f} portfolios/sec  "
        f"io={summary.get('mean_io_read_mb', '?')} MB  "
        f"cpu_peak={summary.get('peak_cpu_pct', '?')}%"
    )
    log.info(f"  Result: {out_path}")

    return result_doc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_median(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return round(float(np.median(clean)), 2) if clean else None


def _safe_max(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return round(float(max(clean)), 1) if clean else None


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")
