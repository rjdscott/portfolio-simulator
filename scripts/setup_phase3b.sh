#!/usr/bin/env bash
# =============================================================================
# setup_phase3b.sh — One-shot environment setup for Phase 3b engines
#
# Installs and builds all 7 Phase 3b compute engines:
#   1. polars_engine       — uv pip install polars
#   2. duckdb_sql          — uv pip install duckdb
#   3. rust_rayon_nightly  — rustup nightly + cargo build
#   4. fortran_openmp      — apt install gfortran + cmake build
#   5. julia_loopvec       — juliaup + juliacall + Pkg.instantiate
#   6. go_goroutines       — user-local Go binary + go build -buildmode=c-shared
#   7. java_vector_api     — javac + jar (Java 21+ required)
#
# Usage:
#   bash scripts/setup_phase3b.sh
#
# The script is idempotent — safe to re-run.  Each step checks whether the
# required artifact already exists before building.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================"
echo "Portfolio Simulator — Phase 3b environment setup"
echo "Repository: $REPO_ROOT"
echo "================================================================"

# ---------------------------------------------------------------------------
# Activate Python venv
# ---------------------------------------------------------------------------
if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: Python venv not found. Run: uv venv --python 3.11 && uv pip install -e ."
    exit 1
fi
source .venv/bin/activate
echo "[OK] Python venv activated: $(python --version)"

# ---------------------------------------------------------------------------
# 1. Polars + DuckDB (Python packages)
# ---------------------------------------------------------------------------
echo ""
echo "--- [1/7] Polars + DuckDB ---"
python -c "import polars" 2>/dev/null && echo "[OK] polars already installed" || {
    echo "Installing polars..."
    uv pip install polars
}
python -c "import duckdb" 2>/dev/null && echo "[OK] duckdb already installed" || {
    echo "Installing duckdb..."
    uv pip install duckdb
}

# ---------------------------------------------------------------------------
# 2. JPype (Java bridge)
# ---------------------------------------------------------------------------
echo ""
echo "--- [2/7] JPype1 (Java bridge) ---"
python -c "import jpype" 2>/dev/null && echo "[OK] jpype1 already installed" || {
    echo "Installing jpype1..."
    uv pip install jpype1
}

# ---------------------------------------------------------------------------
# 3. Rust nightly engine
# ---------------------------------------------------------------------------
echo ""
echo "--- [3/7] Rust nightly (fadd_fast) ---"
NIGHTLY_SO="implementations/rust/rayon_nightly/target/release/libportfolio_rayon_nightly.so"
if [[ -f "$NIGHTLY_SO" ]]; then
    echo "[OK] $NIGHTLY_SO already built"
else
    RUSTUP="${HOME}/.cargo/bin/rustup"
    CARGO="${HOME}/.cargo/bin/cargo"
    if [[ ! -x "$RUSTUP" ]]; then
        echo "ERROR: rustup not found at $RUSTUP. Install via: https://rustup.rs"
        exit 1
    fi
    echo "Installing Rust nightly toolchain..."
    "$RUSTUP" toolchain install nightly
    echo "Building Rust nightly library..."
    "$CARGO" +nightly build --release \
        --manifest-path implementations/rust/rayon_nightly/Cargo.toml
    echo "[OK] Built: $NIGHTLY_SO"
fi

# ---------------------------------------------------------------------------
# 4. FORTRAN OpenMP engine
# ---------------------------------------------------------------------------
echo ""
echo "--- [4/7] FORTRAN OpenMP ---"
FORTRAN_SO="implementations/fortran/openmp/build/libportfolio_fortran.so"
if [[ -f "$FORTRAN_SO" ]]; then
    echo "[OK] $FORTRAN_SO already built"
else
    if ! command -v gfortran &>/dev/null; then
        echo "WARNING: gfortran not found."
        echo "  Install with: sudo apt-get install gfortran"
        echo "  Skipping FORTRAN build — fortran_openmp engine will be unavailable."
    else
        echo "Building FORTRAN library (CMake)..."
        cmake -S implementations/fortran/openmp \
              -B implementations/fortran/openmp/build \
              -DCMAKE_BUILD_TYPE=Release \
              --fresh -Wno-dev
        cmake --build implementations/fortran/openmp/build --parallel
        echo "[OK] Built: $FORTRAN_SO"
    fi
fi

# ---------------------------------------------------------------------------
# 5. Julia LoopVectorization engine
# ---------------------------------------------------------------------------
echo ""
echo "--- [5/7] Julia + LoopVectorization.jl ---"
python -c "import juliacall" 2>/dev/null && echo "[OK] juliacall already installed" || {
    echo "Installing juliacall..."
    uv pip install juliacall
}
# Check Julia binary
JULIA_BIN=""
for candidate in julia "${HOME}/.juliaup/bin/julia" "${HOME}/.local/bin/julia"; do
    if command -v "$candidate" &>/dev/null 2>&1; then
        JULIA_BIN="$candidate"
        break
    fi
done
if [[ -z "$JULIA_BIN" ]]; then
    echo "Julia not found in PATH. Installing via juliaup (user-local, no sudo)..."
    curl -fsSL https://install.julialang.org | sh -s -- --yes
    # Source the updated shell config
    export PATH="${HOME}/.juliaup/bin:$PATH"
    JULIA_BIN="julia"
fi
echo "[OK] Julia: $($JULIA_BIN --version)"

echo "Installing Julia dependencies (LoopVectorization.jl)..."
"$JULIA_BIN" --project=implementations/julia -e '
    using Pkg
    Pkg.instantiate()
    using LoopVectorization
    println("LoopVectorization loaded OK")
'
echo "[OK] Julia project instantiated"

# ---------------------------------------------------------------------------
# 6. Go goroutines engine
# ---------------------------------------------------------------------------
echo ""
echo "--- [6/7] Go goroutines ---"
GO_SO="implementations/go/goroutines/build/libportfolio_go.so"
if [[ -f "$GO_SO" ]]; then
    echo "[OK] $GO_SO already built"
else
    # Find Go: user-local install, then PATH
    GO_BIN=""
    for candidate in go "${HOME}/.local/go-sdk/go/bin/go"; do
        if command -v "$candidate" &>/dev/null 2>&1; then
            GO_BIN="$candidate"
            break
        fi
    done
    if [[ -z "$GO_BIN" ]]; then
        echo "Go not found. Downloading Go 1.22 to ~/.local/go-sdk (no sudo)..."
        mkdir -p "${HOME}/.local/go-sdk"
        curl -sL https://go.dev/dl/go1.22.12.linux-amd64.tar.gz \
            -o /tmp/go.tar.gz
        tar -xzf /tmp/go.tar.gz -C "${HOME}/.local/go-sdk"
        rm /tmp/go.tar.gz
        GO_BIN="${HOME}/.local/go-sdk/go/bin/go"
    fi
    echo "Go version: $($GO_BIN version)"
    echo "Building Go shared library..."
    mkdir -p implementations/go/goroutines/build
    (cd implementations/go/goroutines && \
     "$GO_BIN" build -buildmode=c-shared \
                     -o build/libportfolio_go.so \
                     portfolio_compute.go)
    echo "[OK] Built: $GO_SO"
fi

# ---------------------------------------------------------------------------
# 7. Java Vector API engine
# ---------------------------------------------------------------------------
echo ""
echo "--- [7/7] Java Vector API ---"
JAVA_JAR="implementations/java/vector_api/dist/portfolio_vector_api.jar"
if [[ -f "$JAVA_JAR" ]]; then
    echo "[OK] $JAVA_JAR already built"
else
    if ! command -v javac &>/dev/null; then
        echo "ERROR: javac not found. Install Java 21+ and add to PATH."
        echo "  Ubuntu: sudo apt-get install openjdk-21-jdk"
        exit 1
    fi
    JAVA_VER=$(java -version 2>&1 | head -1 | sed 's/.*version "\([0-9]*\).*/\1/')
    if [[ "${JAVA_VER:-0}" -lt 21 ]]; then
        echo "ERROR: Java 21+ required. Found Java ${JAVA_VER}."
        exit 1
    fi
    echo "Building Java JAR (Java $JAVA_VER, Vector API)..."
    make -C implementations/java/vector_api build
    echo "[OK] Built: $JAVA_JAR"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Phase 3b setup complete."
echo ""
echo "Smoke test (single rep, each engine):"
echo "  python scripts/run_benchmark.py --engine polars_engine      --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache"
echo "  python scripts/run_benchmark.py --engine duckdb_sql         --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache"
echo "  python scripts/run_benchmark.py --engine rust_rayon_nightly --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache"
echo "  python scripts/run_benchmark.py --engine fortran_openmp     --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache"
echo "  python scripts/run_benchmark.py --engine julia_loopvec      --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache"
echo "  python scripts/run_benchmark.py --engine go_goroutines      --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache"
echo "  python scripts/run_benchmark.py --engine java_vector_api    --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache"
echo ""
echo "Full Phase 3b suite (28 configs):"
echo "  python scripts/run_benchmark.py --phase 3b --no-drop-cache"
echo "================================================================"
