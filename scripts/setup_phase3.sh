#!/usr/bin/env bash
# setup_phase3.sh — Install all Phase 3 dependencies and build native extensions.
#
# Run from the project root:
#   bash scripts/setup_phase3.sh
#
# What this does:
#   1. Installs Numba (Python JIT)
#   2. Attempts to install CuPy (GPU) — skipped gracefully if no CUDA
#   3. Builds the C++ OpenMP shared library
#   4. Ensures Rust is installed, then builds the Rust/Rayon shared library
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Phase 3 Setup ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# ---------------------------------------------------------------------------
# 1. Python dependencies
# ---------------------------------------------------------------------------
echo "--- Installing Python dependencies ---"
uv pip install "numba>=0.59.0"

# CuPy: try cuda12x, fall back gracefully
echo "--- Attempting CuPy install (requires CUDA 12.x GPU) ---"
uv pip install "cupy-cuda12x" || {
    echo "  CuPy cuda12x not available, trying cuda11x..."
    uv pip install "cupy-cuda11x" || echo "  CuPy skipped — no compatible CUDA toolkit found."
}

echo ""

# ---------------------------------------------------------------------------
# 2. C++ OpenMP shared library
# ---------------------------------------------------------------------------
echo "--- Building C++ OpenMP library ---"
cmake -S "$PROJECT_ROOT/implementations/cpp/openmp" \
      -B "$PROJECT_ROOT/implementations/cpp/openmp/build" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build "$PROJECT_ROOT/implementations/cpp/openmp/build" --parallel
echo "  Built: implementations/cpp/openmp/build/libportfolio_openmp.so"
echo ""

# ---------------------------------------------------------------------------
# 3. Rust / Rayon shared library
# ---------------------------------------------------------------------------
echo "--- Checking Rust toolchain ---"
if ! command -v rustc &>/dev/null; then
    echo "  Rust not found — installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
else
    rustc --version
fi

echo "--- Building Rust/Rayon library ---"
make -C "$PROJECT_ROOT/implementations/rust/rayon/"
echo "  Built: implementations/rust/rayon/target/release/libportfolio_rayon.so"
echo ""

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "=== Phase 3 setup complete ==="
echo ""
echo "Smoke tests:"
echo "  python scripts/run_benchmark.py --engine numba_parallel --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache"
echo "  python scripts/run_benchmark.py --engine cpp_openmp     --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache"
echo "  python scripts/run_benchmark.py --engine rust_rayon     --storage parquet_wide_uncompressed --scale 1K --reps 1 --no-drop-cache"
echo ""
echo "Full Phase 3 suite:"
echo "  python scripts/run_benchmark.py --phase 3"
echo ""
echo "Rebuild C++ only:"
echo "  cmake --build implementations/cpp/openmp/build --parallel"
