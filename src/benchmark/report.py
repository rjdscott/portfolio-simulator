"""
Result aggregation and reporting.

Reads all JSON result files from results/ and produces:
  - results/summary.csv        — flat table of every run
  - results/comparison.csv     — pivot: (implementation, storage) × scale → throughput
  - results/telemetry.csv      — per-rep telemetry (I/O, CPU, RSS) for every run
  - Printed human-readable summary

Run directly:
    python -m src.benchmark.report
    python scripts/run_benchmark.py --report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

PHASE_LABELS = {
    "1": "CSV baseline",
    "2": "Parquet (Phase 2)",
    "3": "Compute opt (Phase 3)",
    "4": "Distributed (Phase 4)",
}


# ---------------------------------------------------------------------------
# Load all result JSON files
# ---------------------------------------------------------------------------

def load_all_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all result JSON files.

    Returns
    -------
    summary_df  : one row per run, flat columns
    telemetry_df: one row per (run, rep), with all per-rep telemetry
    """
    summary_rows = []
    telemetry_rows = []

    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name in ("summary.csv", "comparison.csv", "telemetry.csv"):
            continue
        try:
            with open(path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"Skipping {path.name}: {e}")
            continue

        cfg = doc.get("config", {})
        hw = doc.get("hardware", {})
        sw = doc.get("software", {})
        smry = doc.get("summary", {})
        timings = doc.get("timings_sec", {})

        # Determine phase from storage format
        sf = cfg.get("storage_format", "")
        phase = _infer_phase(cfg.get("implementation", ""), sf)

        row = {
            "run_id": doc["run_id"],
            "timestamp": doc["timestamp"],
            "phase": phase,
            "implementation": cfg.get("implementation"),
            "language": cfg.get("language"),
            "storage_format": sf,
            "portfolio_scale": cfg.get("portfolio_scale"),
            "portfolio_k": cfg.get("portfolio_k"),
            "universe_size": cfg.get("universe_size"),
            "seed": cfg.get("seed"),
            "repetitions": cfg.get("repetitions"),
            "warmup": cfg.get("warmup"),
            "drop_cache_attempted": cfg.get("drop_cache_attempted"),
            # Timings
            "median_total_sec": smry.get("median_total_sec"),
            "p10_total_sec": smry.get("p10_total_sec"),
            "p90_total_sec": smry.get("p90_total_sec"),
            "median_load_sec": smry.get("median_load_sec"),
            "median_compute_sec": smry.get("median_compute_sec"),
            "throughput": smry.get("throughput_portfolios_per_sec"),
            # Telemetry
            "peak_ram_mb": smry.get("peak_ram_mb"),
            "mean_io_read_mb": smry.get("mean_io_read_mb"),
            "mean_cpu_pct": smry.get("mean_cpu_pct"),
            "peak_cpu_pct": smry.get("peak_cpu_pct"),
            # System
            "cpu_model": hw.get("cpu_model"),
            "cpu_logical_cores": hw.get("cpu_logical_cores"),
            "cpu_physical_cores": hw.get("cpu_physical_cores"),
            "ram_gb": hw.get("ram_gb"),
            "gpu_model": hw.get("gpu_model"),
            "python_version": sw.get("python_version"),
            "numpy_version": sw.get("numpy_version"),
            "pyarrow_version": sw.get("pyarrow_version"),
            "blas": sw.get("blas_implementation"),
            "git_sha": sw.get("implementation_version"),
            "result_checksum": doc.get("result_checksum"),
        }
        summary_rows.append(row)

        # Per-rep telemetry
        for rep_data in doc.get("telemetry_per_rep", []):
            phases = rep_data.get("phases_sec", {})
            telemetry_rows.append({
                "run_id": doc["run_id"],
                "implementation": cfg.get("implementation"),
                "storage_format": sf,
                "portfolio_scale": cfg.get("portfolio_scale"),
                "phase": phase,
                "rep": rep_data.get("rep"),
                "elapsed_sec": rep_data.get("elapsed_sec"),
                "load_prices_sec": phases.get("load_prices"),
                "align_weights_sec": phases.get("align_weights"),
                "compute_metrics_sec": phases.get("compute_metrics"),
                "total_sec": phases.get("total"),
                "io_read_mb": rep_data.get("io_read_mb"),
                "io_read_syscalls": rep_data.get("io_read_syscalls"),
                "cpu_mean_pct": rep_data.get("cpu_mean_pct"),
                "cpu_peak_pct": rep_data.get("cpu_peak_pct"),
                "cpu_p90_pct": rep_data.get("cpu_p90_pct"),
                "rss_peak_mb": rep_data.get("rss_peak_mb"),
                "rss_mean_mb": rep_data.get("rss_mean_mb"),
                "cache_dropped": rep_data.get("cache_dropped"),
            })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["phase", "implementation", "storage_format", "portfolio_scale"])
        .reset_index(drop=True)
    )
    telemetry_df = pd.DataFrame(telemetry_rows)
    return summary_df, telemetry_df


def _infer_phase(implementation: str, storage: str) -> str:
    """Heuristically infer which research phase a run belongs to."""
    # Phase 4 check first — distributed engines always map to phase 4
    if implementation in ("spark_local", "dask_local", "ray_local"):
        return "4_distributed"
    # Phase 3 compute engines — regardless of storage format
    if implementation in (
        "numba_parallel", "cupy_gpu",
        "cpp_openmp", "cpp_blas", "cpp_eigen",
        "cpp_openmp_unroll", "cpp_openmp_tile4", "cpp_openmp_clang",
        "rust_rayon", "rust_rayon_nightly", "rust_ndarray", "rust_faer",
        "fortran_openmp", "julia_loopvec", "go_goroutines", "java_vector_api",
        "polars_engine", "duckdb_sql",
        "pytorch_cpu", "jax_cpu", "numpy_float32",
    ):
        return "3_compute_opt"
    # Phase 2 storage formats (numpy_vectorised engine on parquet/arrow/hdf5)
    if "parquet" in storage or "arrow" in storage or "hdf5" in storage:
        return "2_storage_opt"
    return "1_csv_baseline"


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------

def make_throughput_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot: rows = (phase, implementation, storage_format),
           columns = portfolio_scale,
           values  = median throughput (portfolios/sec).
    """
    pivot = df.pivot_table(
        index=["phase", "implementation", "storage_format"],
        columns="portfolio_scale",
        values="throughput",
        aggfunc="median",
    )
    pivot.columns = [f"N={int(c):,}" for c in pivot.columns]
    return pivot


def make_load_time_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot on median_load_sec — isolates the I/O cost from compute."""
    pivot = df.pivot_table(
        index=["phase", "implementation", "storage_format"],
        columns="portfolio_scale",
        values="median_load_sec",
        aggfunc="median",
    )
    pivot.columns = [f"N={int(c):,}" for c in pivot.columns]
    return pivot


def make_speedup_table(df: pd.DataFrame, baseline_impl: str = "pandas_baseline",
                       baseline_storage: str = "csv_per_stock") -> pd.DataFrame:
    """
    Compute speedup relative to the pandas_baseline / csv_per_stock configuration.
    """
    baseline = df[
        (df["implementation"] == baseline_impl) &
        (df["storage_format"] == baseline_storage)
    ][["portfolio_scale", "throughput"]].rename(columns={"throughput": "baseline_throughput"})

    merged = df.merge(baseline, on="portfolio_scale", how="left")
    merged["speedup_x"] = (merged["throughput"] / merged["baseline_throughput"]).round(1)
    return merged[[
        "phase", "implementation", "storage_format", "portfolio_scale",
        "throughput", "speedup_x", "median_load_sec", "median_compute_sec",
        "mean_io_read_mb", "peak_cpu_pct", "peak_ram_mb",
    ]].sort_values(["portfolio_scale", "speedup_x"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Checksum validation
# ---------------------------------------------------------------------------

def validate_checksums(df: pd.DataFrame) -> bool:
    """
    Verify reproducibility: within each (implementation, storage, scale, seed),
    all runs must produce the same checksum.

    Cross-implementation differences are expected (float accumulation order).
    """
    ok = True
    group_cols = ["implementation", "storage_format", "portfolio_scale", "seed"]
    available = [c for c in group_cols if c in df.columns]

    for key, group in df.groupby(available):
        checksums = group["result_checksum"].dropna().unique()
        label = ", ".join(f"{k}={v}" for k, v in zip(available, key if isinstance(key, tuple) else [key]))
        if len(checksums) > 1:
            log.error(f"Reproducibility failure ({label}): {checksums}")
            ok = False
        elif len(checksums) == 1:
            log.debug(f"Reproducible ({label}): {checksums[0][:16]}...")

    n_impls = df["implementation"].nunique() if "implementation" in df.columns else 0
    if n_impls > 1:
        log.info(
            "Note: Cross-implementation checksum differences are expected "
            "(float accumulation order differs between pandas/numpy/C++/Rust)."
        )
    return ok


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, tel_df: pd.DataFrame) -> None:
    """Print a formatted benchmark report to stdout."""
    SEP = "=" * 72

    print(f"\n{SEP}")
    print("  PORTFOLIO SIMULATOR — BENCHMARK REPORT")
    print(SEP)

    if df.empty:
        print("  No results found. Run benchmarks first.")
        return

    print(f"\n  Total runs   : {len(df)}")
    print(f"  Phases       : {', '.join(sorted(df['phase'].unique()))}")
    print(f"  Engines      : {', '.join(df['implementation'].unique())}")
    print(f"  Storage      : {', '.join(df['storage_format'].unique())}")
    print(f"  Scales       : {', '.join(f'{int(s):,}' for s in sorted(df['portfolio_scale'].unique()))}")

    # Hardware
    hw_row = df.iloc[0]
    print(f"\n  CPU          : {hw_row['cpu_model']}")
    print(f"  Cores        : {hw_row['cpu_logical_cores']} logical / {hw_row['cpu_physical_cores']} physical")
    print(f"  RAM          : {hw_row['ram_gb']} GB")

    # Throughput pivot
    print(f"\n{'-' * 72}")
    print("  THROUGHPUT (portfolios / second)")
    print(f"{'-' * 72}")
    pivot = make_throughput_pivot(df)
    print(pivot.to_string())

    # Load-time pivot (I/O isolation)
    print(f"\n{'-' * 72}")
    print("  PRICE-LOAD TIME (seconds)  — isolates storage I/O cost")
    print(f"{'-' * 72}")
    load_pivot = make_load_time_pivot(df)
    print(load_pivot.to_string())

    # Telemetry summary
    if not tel_df.empty:
        print(f"\n{'-' * 72}")
        print("  TELEMETRY SUMMARY (median across reps)")
        print(f"{'-' * 72}")
        tel_summary = (
            tel_df.groupby(["implementation", "storage_format", "portfolio_scale"])
            .agg(
                io_read_mb=("io_read_mb", "median"),
                cpu_mean_pct=("cpu_mean_pct", "median"),
                cpu_peak_pct=("cpu_peak_pct", "max"),
                rss_peak_mb=("rss_peak_mb", "max"),
            )
            .round(2)
        )
        print(tel_summary.to_string())

    # Speedup table
    if "pandas_baseline" in df["implementation"].values:
        print(f"\n{'-' * 72}")
        print("  SPEEDUP vs pandas_baseline / csv_per_stock")
        print(f"{'-' * 72}")
        speedup = make_speedup_table(df)
        cols = ["implementation", "storage_format", "portfolio_scale",
                "throughput", "speedup_x", "median_load_sec"]
        print(speedup[cols].to_string(index=False))

    # Checksum
    print(f"\n{'-' * 72}")
    print("  REPRODUCIBILITY")
    print(f"{'-' * 72}")
    valid = validate_checksums(df)
    if valid:
        print("  All within-implementation runs produce identical checksums. ✓")
    else:
        print("  WARNING: Reproducibility failures detected — see log above.")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    """Load all results, write CSVs, print report."""
    # Incrementally ingest any new JSON result files into the DuckDB registry
    try:
        from src.benchmark.db import get_connection, ingest_all, export_csv, export_parquet
        con = get_connection()
        n_new = ingest_all(con=con)
        if n_new:
            log.info(f"Ingested {n_new} new run(s) into registry.duckdb")
    except Exception as e:
        log.warning(f"DuckDB registry update skipped: {e}")
        con = None

    df, tel_df = load_all_results()
    print_summary(df, tel_df)

    # Legacy CSVs in results/ (backward compat)
    if not df.empty:
        df.to_csv(RESULTS_DIR / "summary.csv", index=False)
        log.info(f"summary.csv written ({len(df)} rows)")

    if not tel_df.empty:
        tel_df.to_csv(RESULTS_DIR / "telemetry.csv", index=False)
        log.info(f"telemetry.csv written ({len(tel_df)} rows)")

    if not df.empty:
        try:
            pivot = make_throughput_pivot(df)
            pivot.to_csv(RESULTS_DIR / "comparison.csv")
            log.info("comparison.csv written")
        except Exception as e:
            log.warning(f"Could not write comparison.csv: {e}")

    # Publish-ready exports to results/exports/
    if con is not None:
        try:
            export_csv(con=con)
            export_parquet(con=con)
        except Exception as e:
            log.warning(f"Export to results/exports/ skipped: {e}")
        finally:
            con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
