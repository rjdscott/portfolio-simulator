"""
DuckDB Result Registry.

Provides an incremental, queryable index of all benchmark result JSONs.
JSON files remain the immutable source of truth; this registry is a
derived artefact that can be rebuilt at any time by calling ingest_all().

Design
------
- ``results/registry.duckdb``  — DuckDB file, built from JSONs
- ``results/exports/``         — publish-ready Parquet + CSV exports

Deduplication
-------------
A ``config_fingerprint`` is SHA-256 of
(implementation, storage_format, portfolio_scale, portfolio_k, seed, cpu_model).
Re-running the same config marks the old row superseded=TRUE and inserts the new
row.  The ``v_canonical`` view shows only the latest non-superseded row per
fingerprint.

Usage
-----
    from src.benchmark.db import get_connection, ingest_all, export_parquet

    con = get_connection()
    n   = ingest_all(con=con)          # incremental — skips already-ingested JSONs
    export_parquet(con=con)
    export_csv(con=con)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR  = ROOT / "results"
DB_PATH      = RESULTS_DIR / "registry.duckdb"
EXPORTS_DIR  = RESULTS_DIR / "exports"


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id               VARCHAR PRIMARY KEY,
    config_fingerprint   VARCHAR NOT NULL,
    superseded           BOOLEAN DEFAULT FALSE,
    json_path            VARCHAR UNIQUE,
    ingested_at          TIMESTAMP,
    -- identity
    timestamp            TIMESTAMP,
    phase                VARCHAR,
    implementation       VARCHAR,
    language             VARCHAR,
    storage_format       VARCHAR,
    portfolio_scale      BIGINT,
    portfolio_k          INTEGER,
    universe_size        INTEGER,
    seed                 INTEGER,
    repetitions          INTEGER,
    warmup               BOOLEAN,
    drop_cache_attempted BOOLEAN,
    -- hardware
    cpu_model            VARCHAR,
    cpu_logical_cores    INTEGER,
    cpu_physical_cores   INTEGER,
    ram_gb               DOUBLE,
    gpu_model            VARCHAR,
    -- software
    python_version       VARCHAR,
    numpy_version        VARCHAR,
    pyarrow_version      VARCHAR,
    numba_version        VARCHAR,
    blas_implementation  VARCHAR,
    git_sha              VARCHAR,
    omp_num_threads      VARCHAR,
    numba_num_threads    VARCHAR,
    -- summary stats
    median_total_sec     DOUBLE,
    p10_total_sec        DOUBLE,
    p90_total_sec        DOUBLE,
    median_load_sec      DOUBLE,
    median_compute_sec   DOUBLE,
    throughput           DOUBLE,
    peak_ram_mb          DOUBLE,
    mean_io_read_mb      DOUBLE,
    mean_cpu_pct         DOUBLE,
    peak_cpu_pct         DOUBLE,
    -- reproducibility
    result_checksum      VARCHAR,
    notes                VARCHAR
)
"""

_DDL_TELEMETRY = """
CREATE TABLE IF NOT EXISTS telemetry (
    run_id              VARCHAR,
    rep                 INTEGER,
    load_prices_sec     DOUBLE,
    align_weights_sec   DOUBLE,
    compute_metrics_sec DOUBLE,
    total_sec           DOUBLE,
    io_read_mb          DOUBLE,
    cpu_mean_pct        DOUBLE,
    cpu_peak_pct        DOUBLE,
    rss_peak_mb         DOUBLE,
    cache_dropped       BOOLEAN,
    PRIMARY KEY (run_id, rep)
)
"""

_DDL_V_CANONICAL = """
CREATE OR REPLACE VIEW v_canonical AS
    SELECT * FROM runs WHERE superseded = FALSE
"""

_DDL_V_PHASE3 = """
CREATE OR REPLACE VIEW v_phase3 AS
    SELECT implementation, portfolio_scale,
           ROUND(throughput, 0) AS throughput_ps
    FROM v_canonical
    WHERE phase = '3_compute_opt'
    ORDER BY portfolio_scale, throughput_ps DESC
"""

_DDL_V_SPEEDUP = """
CREATE OR REPLACE VIEW v_speedup AS
    SELECT r.implementation, r.portfolio_scale,
           ROUND(r.throughput / b.throughput, 2) AS speedup_x,
           ROUND(r.throughput, 0) AS throughput_ps
    FROM v_canonical r
    JOIN (
        SELECT portfolio_scale, throughput FROM v_canonical
        WHERE implementation = 'numpy_vectorised'
          AND storage_format = 'parquet_wide_uncompressed'
    ) b ON r.portfolio_scale = b.portfolio_scale
    WHERE r.phase = '3_compute_opt'
    ORDER BY r.portfolio_scale, speedup_x DESC
"""


# ---------------------------------------------------------------------------
# Connection + schema bootstrap
# ---------------------------------------------------------------------------

def get_connection(path: Path = DB_PATH):
    """Open the registry DB, apply schema if new, return DuckDB connection."""
    import duckdb

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    _ensure_schema(con)
    return con


def _ensure_schema(con) -> None:
    """Create tables and views if they do not exist yet."""
    con.execute(_DDL_RUNS)
    con.execute(_DDL_TELEMETRY)
    con.execute(_DDL_V_CANONICAL)
    con.execute(_DDL_V_PHASE3)
    con.execute(_DDL_V_SPEEDUP)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _config_fingerprint(doc: dict) -> str:
    """SHA-256 of the config fields that identify a unique benchmark config."""
    cfg = doc.get("config", {})
    hw  = doc.get("hardware", {})
    key = "|".join(str(v) for v in [
        cfg.get("implementation", ""),
        cfg.get("storage_format", ""),
        cfg.get("portfolio_scale", ""),
        cfg.get("portfolio_k", ""),
        cfg.get("seed", ""),
        hw.get("cpu_model", ""),
    ])
    return hashlib.sha256(key.encode()).hexdigest()


def _infer_phase(implementation: str, storage: str) -> str:
    """Heuristically infer which research phase a run belongs to."""
    if implementation in ("spark_local", "dask_local", "ray_local"):
        return "4_distributed"
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
    if "parquet" in storage or "arrow" in storage or "hdf5" in storage:
        return "2_storage_opt"
    return "1_csv_baseline"


def ingest_json(path: Path, con=None) -> str | None:
    """
    Parse one result JSON and insert into the registry.

    Marks any existing row with the same config_fingerprint as superseded=TRUE
    before inserting the new row.  Returns the run_id, or None on error.
    """
    _own_con = con is None
    if _own_con:
        con = get_connection()

    try:
        with open(path) as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Skipping {path.name}: {e}")
        return None

    try:
        run_id = doc["run_id"]
        cfg    = doc.get("config", {})
        hw     = doc.get("hardware", {})
        sw     = doc.get("software", {})
        smry   = doc.get("summary", {})

        fingerprint  = _config_fingerprint(doc)
        storage      = cfg.get("storage_format", "")
        phase        = _infer_phase(cfg.get("implementation", ""), storage)
        ingested_at  = datetime.now(timezone.utc).isoformat()

        # Mark existing rows with same fingerprint as superseded
        con.execute(
            "UPDATE runs SET superseded = TRUE WHERE config_fingerprint = ? AND run_id != ?",
            [fingerprint, run_id],
        )

        # Parse timestamp safely
        ts_raw = doc.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_raw).isoformat() if ts_raw else None
        except ValueError:
            ts = None

        con.execute(
            """
            INSERT INTO runs VALUES (
                ?, ?, FALSE, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?
            )
            """,
            [
                run_id, fingerprint, str(path), ingested_at,
                # identity
                ts, phase,
                cfg.get("implementation"), cfg.get("language"),
                storage,
                cfg.get("portfolio_scale"), cfg.get("portfolio_k"),
                cfg.get("universe_size"), cfg.get("seed"),
                cfg.get("repetitions"), cfg.get("warmup"),
                cfg.get("drop_cache_attempted"),
                # hardware
                hw.get("cpu_model"), hw.get("cpu_logical_cores"),
                hw.get("cpu_physical_cores"), hw.get("ram_gb"),
                hw.get("gpu_model"),
                # software
                sw.get("python_version"), sw.get("numpy_version"),
                sw.get("pyarrow_version"), sw.get("numba_version"),
                sw.get("blas_implementation"),
                sw.get("implementation_version"),   # git SHA
                str(sw.get("omp_num_threads")) if sw.get("omp_num_threads") is not None else None,
                str(sw.get("numba_num_threads")) if sw.get("numba_num_threads") is not None else None,
                # summary stats
                smry.get("median_total_sec"), smry.get("p10_total_sec"),
                smry.get("p90_total_sec"), smry.get("median_load_sec"),
                smry.get("median_compute_sec"),
                smry.get("throughput_portfolios_per_sec"),
                smry.get("peak_ram_mb"), smry.get("mean_io_read_mb"),
                smry.get("mean_cpu_pct"), smry.get("peak_cpu_pct"),
                # reproducibility
                doc.get("result_checksum"), doc.get("notes"),
            ],
        )

        # Per-rep telemetry
        for rep_data in doc.get("telemetry_per_rep", []):
            phases = rep_data.get("phases_sec", {})
            rep    = rep_data.get("rep", 0)
            try:
                con.execute(
                    """
                    INSERT INTO telemetry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        run_id, rep,
                        phases.get("load_prices"),
                        phases.get("align_weights"),
                        phases.get("compute_metrics"),
                        phases.get("total"),
                        rep_data.get("io_read_mb"),
                        rep_data.get("cpu_mean_pct"),
                        rep_data.get("cpu_peak_pct"),
                        rep_data.get("rss_peak_mb"),
                        rep_data.get("cache_dropped"),
                    ],
                )
            except Exception as e:
                log.debug(f"  Telemetry insert skipped for rep {rep}: {e}")

        log.debug(f"  Ingested {path.name} → {run_id[:8]}")
        return run_id

    except Exception as e:
        log.error(f"Failed to ingest {path.name}: {e}")
        return None
    finally:
        if _own_con:
            con.close()


def ingest_all(results_dir: Path = RESULTS_DIR, con=None) -> int:
    """
    Scan results/*.json and ingest any files not already in the registry.

    Returns the count of newly ingested rows.
    """
    _own_con = con is None
    if _own_con:
        con = get_connection()

    try:
        # Fetch already-ingested paths
        existing = set(
            row[0] for row in con.execute("SELECT json_path FROM runs").fetchall()
        )

        new_count = 0
        for path in sorted(results_dir.glob("*.json")):
            if str(path) in existing:
                continue
            result = ingest_json(path, con=con)
            if result is not None:
                new_count += 1

        if new_count:
            log.info(f"Ingested {new_count} new result(s) into registry.")
        else:
            log.debug("Registry up to date — no new results to ingest.")

        return new_count
    finally:
        if _own_con:
            con.close()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_parquet(con=None) -> dict[str, Path]:
    """
    Export v_canonical and telemetry to Parquet files in results/exports/.

    Returns a dict of {name: path} for the files written.
    """
    import duckdb

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    _own_con = con is None
    if _own_con:
        con = get_connection()

    out: dict[str, Path] = {}
    try:
        for name, query in [
            ("summary",   "SELECT * FROM v_canonical ORDER BY phase, implementation, portfolio_scale"),
            ("telemetry", "SELECT t.*, r.implementation, r.storage_format, r.portfolio_scale, r.phase "
                          "FROM telemetry t JOIN v_canonical r USING (run_id)"),
        ]:
            p = EXPORTS_DIR / f"{name}.parquet"
            con.execute(f"COPY ({query}) TO '{p}' (FORMAT PARQUET)")
            log.info(f"Exported {p}")
            out[name] = p

        return out
    finally:
        if _own_con:
            con.close()


def export_csv(con=None) -> dict[str, Path]:
    """
    Export v_canonical to CSV files in results/exports/.

    Also writes a throughput pivot (comparison.csv) for backward compatibility.
    Returns a dict of {name: path}.
    """
    import pandas as pd

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    _own_con = con is None
    if _own_con:
        con = get_connection()

    out: dict[str, Path] = {}
    try:
        # Summary CSV
        df = con.execute("SELECT * FROM v_canonical ORDER BY phase, implementation, portfolio_scale").df()
        p = EXPORTS_DIR / "summary.csv"
        df.to_csv(p, index=False)
        log.info(f"Exported {p} ({len(df)} rows)")
        out["summary"] = p

        # Telemetry CSV
        tel_df = con.execute(
            "SELECT t.*, r.implementation, r.storage_format, r.portfolio_scale, r.phase "
            "FROM telemetry t JOIN v_canonical r USING (run_id)"
        ).df()
        p = EXPORTS_DIR / "telemetry.csv"
        tel_df.to_csv(p, index=False)
        log.info(f"Exported {p} ({len(tel_df)} rows)")
        out["telemetry"] = p

        # Comparison pivot (backward compat)
        if not df.empty and "throughput" in df.columns:
            try:
                pivot = df.pivot_table(
                    index=["phase", "implementation", "storage_format"],
                    columns="portfolio_scale",
                    values="throughput",
                    aggfunc="median",
                )
                pivot.columns = [f"N={int(c):,}" for c in pivot.columns]
                p = EXPORTS_DIR / "comparison.csv"
                pivot.to_csv(p)
                log.info(f"Exported {p}")
                out["comparison"] = p
            except Exception as e:
                log.warning(f"Could not write comparison.csv: {e}")

        return out
    finally:
        if _own_con:
            con.close()
