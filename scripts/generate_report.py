#!/usr/bin/env python3
"""
Generate a single comprehensive markdown results report (RESULTS.md).

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output docs/results/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PARQUET_DIR = ROOT / "data" / "parquet"
PRICES_DIR = ROOT / "data" / "raw" / "prices"
PORTFOLIOS_DIR = ROOT / "data" / "raw" / "portfolios"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results() -> list[dict]:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue

        c = doc["config"]
        s = doc["summary"]
        hw = doc.get("hardware", {})
        sw = doc.get("software", {})
        timings = doc.get("timings_sec", {})

        n = c["portfolio_scale"]
        total_sec = s["median_total_sec"]
        load_sec = s.get("median_load_sec") or 0.0
        compute_sec = s.get("median_compute_sec") or 0.0
        tp = s["throughput_portfolios_per_sec"]

        row = {
            "run_id": doc["run_id"][:8],
            "timestamp": doc["timestamp"],
            "phase": _infer_phase(c["implementation"], c["storage_format"]),
            "implementation": c["implementation"],
            "storage_format": c["storage_format"],
            "N": n,
            "K": c.get("portfolio_k", 15),
            "universe": c.get("universe_size", 100),
            "seed": c.get("seed", 42),
            "total_sec": total_sec,
            "p10_sec": s.get("p10_total_sec", total_sec),
            "p90_sec": s.get("p90_total_sec", total_sec),
            "load_sec": load_sec,
            "compute_sec": compute_sec,
            "load_pct": _pct(load_sec, total_sec),
            "compute_pct": _pct(compute_sec, total_sec),
            "throughput": tp,
            "us_per_portfolio": (total_sec / n * 1e6) if n > 0 else 0,
            "peak_rss_mb": s.get("peak_ram_mb") or 0.0,
            "mean_cpu_pct": s.get("mean_cpu_pct") or 0.0,
            "io_read_mb": s.get("mean_io_read_mb") or 0.0,
            "cpu_logical": hw.get("cpu_logical_cores", 28),
            "cpu_physical": hw.get("cpu_physical_cores", 20),
            "ram_gb": hw.get("ram_gb", 67.2),
            "cpu_model": hw.get("cpu_model", ""),
            "numpy_version": sw.get("numpy_version", ""),
            "pyarrow_version": sw.get("pyarrow_version", ""),
            "checksum": doc.get("result_checksum", ""),
            "all_totals": timings.get("total", []),
        }

        row["variance_pct"] = _variance_pct(row["p10_sec"], row["p90_sec"], total_sec)
        row["throughput_per_core"] = tp / row["cpu_logical"]

        rows.append(row)

    return sorted(rows, key=lambda r: (r["phase"], r["implementation"], r["storage_format"], r["N"]))


def _infer_phase(impl: str, storage: str) -> str:
    if "parquet" in storage or "arrow" in storage or "hdf5" in storage:
        return "2_parquet"
    if impl in ("numba_parallel", "cupy_gpu", "cpp_openmp", "cpp_blas",
                "rust_rayon", "rust_ndarray"):
        return "3_compute"
    if impl in ("spark_local", "dask_local", "ray_local"):
        return "4_distributed"
    return "1_csv"


def _pct(part: float, total: float) -> float:
    return round(part / total * 100, 1) if total > 0 else 0.0


def _variance_pct(p10: float, p90: float, median: float) -> float:
    if not p10 or not p90 or median <= 0:
        return 0.0
    return round((p90 - p10) / median * 100, 1)


# ---------------------------------------------------------------------------
# Storage size helpers
# ---------------------------------------------------------------------------

def get_storage_sizes() -> dict[str, float]:
    sizes: dict[str, float] = {}

    def mb(p: Path) -> float:
        return p.stat().st_size / 1_048_576 if p.exists() else 0.0

    per_stock_total = sum(
        f.stat().st_size for f in PRICES_DIR.glob("*.csv")
        if f.stem not in ("prices_wide", "prices_long")
    ) / 1_048_576
    sizes["CSV per-stock (100 files)"] = per_stock_total
    sizes["CSV wide (1 file)"] = mb(PRICES_DIR / "prices_wide.csv")
    sizes["CSV long (tidy)"] = mb(PRICES_DIR / "prices_long.csv")

    for codec, label in [
        ("snappy",       "Parquet wide (snappy)"),
        ("zstd",         "Parquet wide (zstd)"),
        ("uncompressed", "Parquet wide (uncompressed)"),
    ]:
        sizes[label] = mb(PARQUET_DIR / f"prices_wide_{codec}.parquet")

    per_stock_dir = PARQUET_DIR / "per_stock"
    if per_stock_dir.exists():
        total = sum(f.stat().st_size for f in per_stock_dir.glob("*.parquet")) / 1_048_576
        sizes["Parquet per-stock (snappy, 100 files)"] = total

    return {k: v for k, v in sizes.items() if v > 0}


def get_portfolio_sizes() -> dict[str, float]:
    out = {}
    for f in sorted(PORTFOLIOS_DIR.glob("portfolios_*.csv")):
        label = f.stem.replace("portfolios_", "N=").replace("_", ",")
        out[label] = f.stat().st_size / 1_048_576
    return out


# ---------------------------------------------------------------------------
# ASCII bar chart (code-block only, not inside tables)
# ---------------------------------------------------------------------------

def bar(value: float, max_val: float, width: int = 28) -> str:
    if max_val <= 0:
        return ""
    filled = max(0, min(int(round(value / max_val * width)), width))
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Markdown table builder
# ---------------------------------------------------------------------------

def md_table(headers: list[str], rows: list[list],
             alignments: list[str] | None = None) -> str:
    """
    Render a GFM-compliant markdown table.
    alignments: list of 'l', 'r', 'c' per column (default: first col left, rest right).
    Alignment markers: left = :---, right = ---:, center = :---:
    """
    if not rows:
        return "_No data_\n"
    if alignments is None:
        alignments = ["l"] + ["r"] * (len(headers) - 1)

    col_widths = [
        max(len(str(h)), max(len(str(row[i])) for row in rows))
        for i, h in enumerate(headers)
    ]
    col_widths = [max(w, 3) for w in col_widths]

    def pad(val: str, width: int, align: str) -> str:
        if align == "r":
            return val.rjust(width)
        if align == "c":
            return val.center(width)
        return val.ljust(width)

    def sep(width: int, align: str) -> str:
        dashes = "-" * width
        if align == "r":
            return dashes[:-1] + ":"       # ---:
        if align == "c":
            return ":" + dashes[1:-1] + ":" # :--:
        return ":" + dashes[1:]             # :---

    hdr = "| " + " | ".join(pad(h, col_widths[i], alignments[i])
                             for i, h in enumerate(headers)) + " |"
    div = "| " + " | ".join(sep(col_widths[i], alignments[i])
                             for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(pad(str(row[i]), col_widths[i], alignments[i])
                           for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([hdr, div] + body) + "\n"


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_n(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n // 1_000_000_000}B"
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def _fmt_time(sec: float) -> str:
    if sec == 0:
        return "—"
    if sec < 0.001:
        return f"{sec * 1e6:.0f} µs"
    if sec < 1.0:
        return f"{sec * 1000:.1f} ms"
    if sec < 60:
        return f"{sec:.3f} s"
    return f"{sec / 60:.1f} min"


def _fmt_tp(tp: float) -> str:
    if tp >= 1_000_000:
        return f"{tp / 1_000_000:.2f}M/s"
    if tp >= 1_000:
        return f"{tp / 1_000:.1f}K/s"
    return f"{tp:.0f}/s"


def _fmt_lat(us: float) -> str:
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    if us >= 1:
        return f"{us:.2f} µs"
    return f"{us * 1000:.1f} ns"


def _storage_label(sf: str) -> str:
    return {
        "csv_per_stock":             "CSV per-stock",
        "csv_wide":                  "CSV wide",
        "csv_long":                  "CSV long",
        "parquet_per_stock":         "Parquet per-stock",
        "parquet_wide_snappy":       "Parquet wide (snappy)",
        "parquet_wide_zstd":         "Parquet wide (zstd)",
        "parquet_wide_uncompressed": "Parquet wide (uncompressed)",
        "arrow_ipc":                 "Arrow IPC",
        "hdf5":                      "HDF5",
    }.get(sf, sf)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Pivot helper
# ---------------------------------------------------------------------------

def _pivot(rows: list[dict]) -> tuple[list[str], list[int], dict]:
    storages = sorted({r["storage_format"] for r in rows})
    scales = sorted({r["N"] for r in rows})
    p: dict[str, dict[int, dict]] = {}
    for r in rows:
        p.setdefault(r["storage_format"], {})[r["N"]] = r
    return storages, scales, p


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_environment(rows: list[dict]) -> str:
    r = rows[0]
    sha = _git_sha()

    hw_rows = [
        ["CPU model", r["cpu_model"] or "x86_64"],
        ["Logical cores", str(r["cpu_logical"])],
        ["Physical cores", str(r["cpu_physical"])],
        ["RAM", f"{r['ram_gb']} GB"],
        ["Storage", "Local NVMe SSD"],
        ["GPU", "See Phase 3"],
    ]
    sw_rows = [
        ["NumPy", r["numpy_version"]],
        ["pyarrow", r["pyarrow_version"]],
        ["Git SHA", f"`{sha}`"],
    ]

    lines = [
        "## 1. Environment\n",
        "### Hardware\n",
        md_table(["Component", "Value"], hw_rows, ["l", "l"]),
        "### Software\n",
        md_table(["Package", "Version"], sw_rows, ["l", "l"]),
        "### Benchmark Protocol\n",
        "- **Repetitions**: 5 timed runs + 1 untimed warmup",
        "- **Cache state**: warm OS page cache (cold-cache requires root to flush)",
        "- **Reported metric**: median wall-clock time",
        "- **Seed**: 42  |  K = 15 stocks/portfolio  |  Universe = 100 stocks",
        "- **Metrics computed**: cumulative total return + annualised Sharpe ratio (Rf = 0)",
        "",
    ]
    return "\n".join(lines) + "\n"


def section_storage_sizes() -> str:
    price_sizes = get_storage_sizes()
    port_sizes = get_portfolio_sizes()

    csv_wide_mb = price_sizes.get("CSV wide (1 file)", 2.19)
    max_mb = max(price_sizes.values()) if price_sizes else 1.0

    # Table: Format | Size | vs CSV wide
    price_rows = []
    for label, mb in sorted(price_sizes.items(), key=lambda x: x[1], reverse=True):
        ratio = mb / csv_wide_mb if csv_wide_mb > 0 else 1.0
        price_rows.append([label, f"{mb:.2f} MB", f"{ratio:.1f}x"])

    # Bar chart as code block
    chart_lines = ["```"]
    for label, mb in sorted(price_sizes.items(), key=lambda x: x[1], reverse=True):
        chart_lines.append(f"  {label:<38}  {mb:>6.2f} MB  {bar(mb, max_mb)}")
    chart_lines.append("```")

    port_rows = []
    for label, mb in sorted(port_sizes.items()):
        port_rows.append([label, f"{mb:.1f} MB"])

    lines = [
        "## 2. Data & Storage\n",
        "### Price Data File Sizes (100 stocks × 1,257 trading days)\n",
        md_table(["Format", "Size", "vs CSV wide"], price_rows, ["l", "r", "r"]),
        "\n".join(chart_lines),
        "",
        "> Compression cuts file size 2–4× but adds CPU decode overhead."
        " On local NVMe, uncompressed reads fastest.\n",
        "### Portfolio Weight Matrix Sizes\n",
        md_table(["Scale", "CSV size"], port_rows, ["l", "r"]),
        "> N = 1M (622 MB) fits in 64 GB RAM. N > 10M is generated on-the-fly from seeds.",
        "",
    ]
    return "\n".join(lines) + "\n"


def section_total_time(rows: list[dict]) -> str:
    storages, scales, p = _pivot(rows)

    headers = ["Storage Format"] + [_fmt_n(n) for n in scales]
    align = ["l"] + ["r"] * len(scales)

    time_rows = []
    for sf in storages:
        row = [_storage_label(sf)]
        for n in scales:
            r = p.get(sf, {}).get(n)
            row.append(_fmt_time(r["total_sec"]) if r else "—")
        time_rows.append(row)

    var_rows = []
    for sf in storages:
        row = [_storage_label(sf)]
        for n in scales:
            r = p.get(sf, {}).get(n)
            row.append(f"±{r['variance_pct']:.0f}%" if r else "—")
        var_rows.append(row)

    lines = [
        "## 3. Total Completion Time\n",
        "> End-to-end wall-clock: load prices → compute returns → collect results."
        " Median of 5 repetitions, warm page cache.\n",
        md_table(headers, time_rows, align),
        "### Run-to-run Variance  (p90 − p10) / median\n",
        md_table(headers, var_rows, align),
        "> Values >10% indicate sensitivity to system state (cache, NUMA, scheduling).",
        "",
    ]
    return "\n".join(lines) + "\n"


def section_throughput(rows: list[dict]) -> str:
    storages, scales, p = _pivot(rows)

    headers = ["Storage Format"] + [_fmt_n(n) for n in scales]
    align = ["l"] + ["r"] * len(scales)

    tp_rows = []
    for sf in storages:
        row = [_storage_label(sf)]
        for n in scales:
            r = p.get(sf, {}).get(n)
            row.append(_fmt_tp(r["throughput"]) if r else "—")
        tp_rows.append(row)

    lines = [
        "## 4. Throughput (portfolios / second)\n",
        md_table(headers, tp_rows, align),
        "",
    ]

    # Bar chart at largest scale
    max_n = max(scales)
    chart_data = [(sf, p.get(sf, {}).get(max_n)) for sf in storages]
    chart_data = [(sf, r) for sf, r in chart_data if r]
    if chart_data:
        max_tp = max(r["throughput"] for _, r in chart_data)
        lines += [f"**N = {_fmt_n(max_n)}** (compute-bound — all formats converge)\n", "```"]
        for sf, r in sorted(chart_data, key=lambda x: x[1]["throughput"], reverse=True):
            lines.append(f"  {_storage_label(sf):<28}  {_fmt_tp(r['throughput']):>9}  {bar(r['throughput'], max_tp)}")
        lines += ["```", ""]

    # Bar chart at N=100 (storage-dominated)
    chart_data_100 = [(sf, p.get(sf, {}).get(100)) for sf in storages]
    chart_data_100 = [(sf, r) for sf, r in chart_data_100 if r]
    if chart_data_100:
        max_tp_100 = max(r["throughput"] for _, r in chart_data_100)
        lines += ["**N = 100** (storage-dominated — format matters most here)\n", "```"]
        for sf, r in sorted(chart_data_100, key=lambda x: x[1]["throughput"], reverse=True):
            lines.append(f"  {_storage_label(sf):<28}  {_fmt_tp(r['throughput']):>9}  {bar(r['throughput'], max_tp_100)}")
        lines += ["```", ""]

    return "\n".join(lines) + "\n"


def section_latency(rows: list[dict]) -> str:
    storages, scales, p = _pivot(rows)

    headers = ["Storage Format"] + [_fmt_n(n) for n in scales]
    align = ["l"] + ["r"] * len(scales)

    lat_rows = []
    for sf in storages:
        row = [_storage_label(sf)]
        for n in scales:
            r = p.get(sf, {}).get(n)
            row.append(_fmt_lat(r["us_per_portfolio"]) if r else "—")
        lat_rows.append(row)

    lines = [
        "## 5. Per-Portfolio Latency\n",
        "> Total time / N — includes amortised cost of loading price data.\n",
        md_table(headers, lat_rows, align),
        "> At N = 1M with any wide format, per-portfolio cost is ~5.6 µs."
        " Latency stabilises at N ≥ 100K, confirming the compute-bound regime.",
        "",
    ]
    return "\n".join(lines) + "\n"


def section_load_compute(rows: list[dict]) -> str:
    storages, scales, p = _pivot(rows)

    headers = ["Storage Format"] + [_fmt_n(n) for n in scales]
    align = ["l"] + ["r"] * len(scales)

    load_rows, compute_rows, load_pct_rows = [], [], []
    for sf in storages:
        lr = [_storage_label(sf)]
        cr = [_storage_label(sf)]
        pr = [_storage_label(sf)]
        for n in scales:
            r = p.get(sf, {}).get(n)
            if r:
                lr.append(_fmt_time(r["load_sec"]))
                cr.append(_fmt_time(r["compute_sec"]))
                pr.append(f"{r['load_pct']:.0f}%")
            else:
                lr.append("—"); cr.append("—"); pr.append("—")
        load_rows.append(lr)
        compute_rows.append(cr)
        load_pct_rows.append(pr)

    # Bar chart: load time at N=1K
    n_ref = 1_000
    chart_data = [(sf, p.get(sf, {}).get(n_ref)) for sf in storages]
    chart_data = [(sf, r) for sf, r in chart_data if r]

    lines = [
        "## 6. Load vs Compute Breakdown\n",
        "### Price Load Time (median)\n",
        md_table(headers, load_rows, align),
        "### Compute Time (median)\n",
        md_table(headers, compute_rows, align),
        "### Load as % of Total\n",
        md_table(headers, load_pct_rows, align),
        "> At N = 100 with per-stock CSV, ~98% of time is file I/O."
        " At N ≥ 100K, load drops below 1% — compute dominates regardless of format.",
        "",
    ]

    if chart_data:
        max_load = max(r["load_sec"] for _, r in chart_data)
        lines += ["**Price load time at N = 1K**\n", "```"]
        for sf, r in sorted(chart_data, key=lambda x: x[1]["load_sec"], reverse=True):
            lines.append(f"  {_storage_label(sf):<28}  {_fmt_time(r['load_sec']):>8}  {bar(r['load_sec'], max_load)}")
        lines += ["```", ""]

    return "\n".join(lines) + "\n"


def section_telemetry(rows: list[dict]) -> str:
    storages, scales, p = _pivot(rows)

    headers = ["Storage Format"] + [_fmt_n(n) for n in scales]
    align = ["l"] + ["r"] * len(scales)

    rss_rows, cpu_rows = [], []
    for sf in storages:
        rr = [_storage_label(sf)]
        cr = [_storage_label(sf)]
        for n in scales:
            r = p.get(sf, {}).get(n)
            if r:
                rr.append(f"{r['peak_rss_mb'] / 1024:.1f} GB" if r["peak_rss_mb"] >= 1024
                           else f"{r['peak_rss_mb']:.0f} MB")
                cr.append(f"{r['mean_cpu_pct']:.0f}%")
            else:
                rr.append("—"); cr.append("—")
        rss_rows.append(rr)
        cpu_rows.append(cr)

    lines = [
        "## 7. Telemetry\n",
        "> All runs used warm OS page cache — `io_read_mb = 0` for all configs."
        " Cold-cache benchmarks require root to flush `/proc/sys/vm/drop_caches`.\n",
        "### Peak RAM\n",
        md_table(headers, rss_rows, align),
        "> RAM scales linearly with N."
        " At N = 1M, ~22 GB is consumed (dominated by the 1M × 100 float32 weight matrix).\n",
        "### Mean CPU Utilisation\n",
        md_table(headers, cpu_rows, align),
        "> Near 100% at small N (compute fills the run)."
        " Falls to ~17% at N = 1M because loading the 622 MB portfolio CSV is sequential.",
        "",
    ]
    return "\n".join(lines) + "\n"


def section_speedup(rows: list[dict]) -> str:
    baseline = {r["N"]: r for r in rows
                if r["implementation"] == "pandas_baseline"
                and r["storage_format"] == "csv_per_stock"}
    if not baseline:
        return ""

    scales = sorted(baseline)
    np_rows = [r for r in rows if r["implementation"] == "numpy_vectorised"]
    storages, _, p = _pivot(np_rows)

    headers = ["Storage Format"] + [_fmt_n(n) for n in scales]
    align = ["l"] + ["r"] * len(scales)

    sp_rows = []
    for sf in storages:
        row = [_storage_label(sf)]
        for n in scales:
            r = p.get(sf, {}).get(n)
            b = baseline.get(n)
            if r and b:
                su = b["total_sec"] / r["total_sec"]
                row.append(f"**{su:.1f}×**" if su >= 5 else f"{su:.1f}×")
            else:
                row.append("—")
        sp_rows.append(row)

    lines = [
        "## 8. Speedup vs Baseline\n",
        "> Baseline: `pandas_baseline` + `csv_per_stock`"
        " (Python row loop over 100 individual files). **Bold** = ≥ 5×.\n",
        md_table(headers, sp_rows, align),
        "> At N = 100 the run is entirely I/O-bound:"
        " `parquet_wide_uncompressed` gives **14×** by eliminating 100 file opens."
        " At N = 1M the 5.6 s compute cost swamps the 4–70 ms I/O difference,",
        "> so all formats converge to ~1×.",
        "",
    ]
    return "\n".join(lines) + "\n"


def section_projections(rows: list[dict]) -> str:
    best_1m = min(
        (r for r in rows if r["N"] == 1_000_000 and "wide" in r["storage_format"]),
        key=lambda r: r["total_sec"],
        default=None,
    )
    if best_1m is None:
        return ""

    spp = best_1m["compute_sec"] / 1_000_000  # seconds per portfolio
    load_fixed = 0.005

    proj = []
    for n, label in [
        (1_000_000,       "1M (observed)"),
        (100_000_000,     "100M (projected)"),
        (1_000_000_000,   "1B (projected)"),
    ]:
        t = load_fixed + spp * n
        ram_gb = best_1m["peak_rss_mb"] / 1_000_000 * n / 1024
        proj.append([label, _fmt_time(t), _fmt_tp(n / t), f"{ram_gb:.0f} GB"])

    lines = [
        "## 9. Projections\n",
        "> Extrapolated from N = 1M compute rate (single-threaded NumPy/BLAS)."
        " RAM assumes linear scaling."
        " Phase 3/4 (GPU, distributed) will dramatically reduce these times.\n",
        md_table(["Scale", "Total Time", "Throughput", "Peak RAM (est.)"],
                 proj, ["l", "r", "r", "r"]),
        f"> N = 100M: estimated ~{best_1m['peak_rss_mb'] / 1_000_000 * 100_000_000 / 1024:.0f} GB RAM"
        " — exceeds this machine. Requires seeded batch generation.",
        f"> N = 1B: ~{spp * 1_000_000_000 / 3600:.1f} hours single-threaded."
        " Requires GPU or distributed compute (Phase 3/4).",
        "",
    ]
    return "\n".join(lines) + "\n"


def section_findings() -> str:
    return """\
## 10. Key Findings

### Finding 1 — Storage format only matters at small N

At N ≥ 100K the BLAS matrix multiply (~560 ms) completely dominates price loading
(4–70 ms). Switching formats gives < 2% end-to-end improvement. Optimising storage
before compute is premature at large N.

At N ≤ 1K, storage is everything: `parquet_wide_uncompressed` is **14×** faster than
`csv_per_stock` — all gain comes from eliminating 100 file opens and text parsing.

### Finding 2 — File count is the I/O bottleneck, not file format

`csv_per_stock` and `parquet_per_stock` load in 79 ms vs 71 ms — nearly identical.
The 100 sequential `open()` / `read()` / `close()` syscalls cost ~70 ms regardless
of encoding. **Partition strategy matters more than file format.**

### Finding 3 — Compression is a liability on local NVMe

For a 0.5–1 MB file on local SSD:

| Format                    | Load time | vs uncompressed |
| :------------------------ | --------: | --------------: |
| Parquet wide (uncompressed) | 4.3 ms  | 1.0×            |
| Parquet wide (snappy)       | 7.8 ms  | 1.8×            |
| Parquet wide (zstd)         | 12.4 ms | 2.9×            |

Decompression CPU cost exceeds I/O savings at this file size. On a cloud object
store with limited bandwidth, zstd would likely win.

### Finding 4 — NumPy/BLAS is compute-bound and single-threaded

At N = 1M, mean CPU is 17% despite a 100% peak burst during the matmul.
Loading the 622 MB portfolio CSV is fully sequential. Even with parquet prices
(4 ms load), 99.9% of wall time is `np.dot()`. The 176K portfolios/sec ceiling
is the primary target for Phase 3.

### Finding 5 — RAM is the binding constraint for N ≥ 10M on 64 GB

| N    | Weight matrix (float32) | Observed / estimated RAM |
| :--- | ----------------------: | -----------------------: |
| 1M   | 400 MB                  | ~22 GB (observed)        |
| 10M  | 4 GB                    | ~50 GB (estimated)       |
| 100M | 40 GB                   | > 64 GB — OOM            |

At N > ~30M the weight matrix cannot be fully materialised. Phase 3 will use
seeded batch generation.

---

## 11. Hypothesis Status

| ID | Hypothesis | Status | Evidence |
| :- | :--------- | :----: | :------- |
| H1 | CSV-per-stock is worst for cross-sectional ops | ✅ Confirmed | 100 file opens = 70 ms fixed overhead; 14× slower than best Parquet |
| H2 | NumPy matmul ≥ 10× faster than pandas loop at 1M | ✅ Confirmed | Pure compute step is >100× faster; pandas loop dominated by Python overhead |
| H3 | Parquet ≥ 5× faster than CSV | ⚠️ Regime-dependent | True at small N (14.1×); false at N ≥ 100K (1.0×) |
| H4 | GPU dominates at N ≥ 100K | ⏳ Pending Phase 3 | Matmul at 1M takes 5.6 s on 1 CPU thread — GPU expected to cut to ms |
| H5 | No single-machine solution < 60 s at N ≥ 100M | ⏳ Pending Phase 3/4 | Projected ~94 min single-threaded; GPU could reduce to seconds |
| H6 | Memory bandwidth is binding at 1B | ⏳ Pending Phase 4 | RAM projections indicate OOM before FLOP ceiling on this machine |

"""


def section_open_questions() -> str:
    return """\
## 12. Open Questions

1. **Cold-cache I/O**: all benchmarks ran with warm OS page cache (`io_read_mb = 0`
   for all formats). True cold-read numbers require `sudo` to flush the page cache.

2. **Break-even N for Parquet vs CSV**: the crossover appears between N = 1K and
   N = 100K. The exact point has not been measured.

3. **Arrow IPC (zero-copy)**: memory-mapping the returns matrix would reduce load time
   to near-zero and eliminate the `pd.read_parquet` + `.to_numpy()` copy. Likely the
   optimal format for Phase 3 GPU benchmarks.

4. **Column pruning**: we always load all 100 tickers. Parquet's columnar format would
   give larger gains with sector-constrained portfolios that read a subset of tickers.

5. **Portfolio CSV load time**: at N = 1M, reading the 622 MB weight CSV takes ~4.6 s
   and is not yet isolated as a separate phase. It is larger than the price-load cost.

"""


# ---------------------------------------------------------------------------
# Appendix
# ---------------------------------------------------------------------------

def section_appendix(rows: list[dict]) -> str:
    app_rows = []
    for r in rows:
        phase = r["phase"].replace("_", " ")
        app_rows.append([
            phase,
            _storage_label(r["storage_format"]),
            _fmt_n(r["N"]),
            _fmt_time(r["total_sec"]),
            _fmt_tp(r["throughput"]),
            _fmt_time(r["load_sec"]),
            _fmt_time(r["compute_sec"]),
            f"{r['peak_rss_mb'] / 1024:.1f} GB" if r["peak_rss_mb"] >= 1024
            else f"{r['peak_rss_mb']:.0f} MB",
        ])

    headers = ["Phase", "Storage", "N", "Total", "Throughput", "Load", "Compute", "Peak RAM"]
    align = ["l", "l", "r", "r", "r", "r", "r", "r"]

    lines = [
        "---\n",
        "## Appendix — Raw Results\n",
        "> One row per configuration. Timing = median of 5 repetitions.\n",
        md_table(headers, app_rows, align),
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_report(rows: list[dict]) -> str:
    if not rows:
        return "# No benchmark results found.\n\nRun benchmarks first.\n"

    phases = sorted({r["phase"] for r in rows})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sha = _git_sha()
    scales = sorted({r["N"] for r in rows})

    header = f"""\
# Portfolio Return Simulator — Results

> **Generated**: {now}  |  **Git**: `{sha}`  |  **Runs**: {len(rows)}

Benchmarks cover {len(phases)} phase(s), portfolio scales {_fmt_n(min(scales))} – {_fmt_n(max(scales))}.
Metrics: annualised Sharpe ratio + cumulative return over 1,257 trading days,
K = 15 stocks per portfolio, 100-stock universe (top S&P 500 by market cap, 2020–2024).

---

## Contents

1. [Environment](#1-environment)
2. [Data & Storage](#2-data--storage)
3. [Total Completion Time](#3-total-completion-time)
4. [Throughput](#4-throughput-portfolios--second)
5. [Per-Portfolio Latency](#5-per-portfolio-latency)
6. [Load vs Compute Breakdown](#6-load-vs-compute-breakdown)
7. [Telemetry](#7-telemetry)
8. [Speedup vs Baseline](#8-speedup-vs-baseline)
9. [Projections](#9-projections)
10. [Key Findings](#10-key-findings)
11. [Hypothesis Status](#11-hypothesis-status)
12. [Open Questions](#12-open-questions)

---

"""

    np_rows = [r for r in rows if r["implementation"] == "numpy_vectorised"]

    return (
        header
        + section_environment(rows)
        + section_storage_sizes()
        + section_total_time(np_rows)
        + section_throughput(np_rows)
        + section_latency(np_rows)
        + section_load_compute(np_rows)
        + section_telemetry(np_rows)
        + section_speedup(rows)
        + section_projections(np_rows)
        + section_findings()
        + section_open_questions()
        + section_appendix(rows)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate RESULTS.md from benchmark JSON files.")
    parser.add_argument("--output", default=str(ROOT / "RESULTS.md"))
    args = parser.parse_args()

    rows = load_results()
    if not rows:
        print("ERROR: No result JSON files found. Run benchmarks first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} benchmark runs.")
    report_md = build_report(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_md, encoding="utf-8")
    print(f"Report written: {out}  ({len(report_md):,} chars)")


if __name__ == "__main__":
    main()
