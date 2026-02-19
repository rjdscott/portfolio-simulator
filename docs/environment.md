# Environment Specification

This document records the hardware and software environment for all benchmarks.
Update this file after each new benchmark machine or major software change.

---

## Hardware — Primary Benchmark Machine

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i7 (20 logical cores) |
| RAM | 64 GB |
| GPU | (run `nvidia-smi` and update here) |
| GPU VRAM | (update here) |
| Storage | Local NVMe SSD |
| OS | Linux (kernel version: run `uname -r`) |

## Estimated Compute Budget

| Operation | Estimate |
|----------|---------|
| 1M portfolio matmul (NumPy/BLAS, 20 threads) | ~1–5 sec |
| 1M portfolio matmul (CuPy, GPU) | < 0.5 sec (estimated) |
| 1M portfolio matmul (C++/OpenMP) | < 0.5 sec (estimated) |
| Loading wide CSV (100 stocks × 1258 days) | ~0.05 sec |
| Loading per-stock CSVs (100 files) | ~0.2–0.5 sec |

---

## Software Environment

Run the following to capture your environment:

```bash
# Python environment
pip freeze > docs/pip_freeze.txt

# System
uname -a > docs/system_info.txt
lscpu >> docs/system_info.txt
free -h >> docs/system_info.txt

# GPU
nvidia-smi >> docs/system_info.txt

# NumPy BLAS configuration
python -c "import numpy; numpy.show_config()"
```

### Python
- Python ≥ 3.11
- NumPy ≥ 1.26 (with MKL or OpenBLAS BLAS backend)
- pandas ≥ 2.2

### C++ (Phase 3)
- GCC ≥ 13 or Clang ≥ 17
- CMake ≥ 3.20
- OpenBLAS or MKL (CBLAS interface)
- OpenMP (comes with GCC)
- CUDA Toolkit ≥ 12.0 (for GPU variants)

### Rust (Phase 3)
- Rust stable ≥ 1.78 (`rustup update`)
- Cargo (comes with Rust)
- blas-src with system BLAS

### Kotlin / JVM (Phase 3)
- JDK ≥ 21 (LTS, virtual threads support)
- Kotlin ≥ 2.0
- Gradle ≥ 8.0

### Distributed (Phase 4)
- Apache Spark ≥ 3.5
- PySpark ≥ 3.5
- Apache Arrow ≥ 15.0

---

## Docker Environment (Spark Cluster)

See `docker/` for docker-compose configuration.

```bash
# Start local Spark cluster
docker-compose -f docker/spark-cluster.yml up -d

# Check cluster health
docker-compose -f docker/spark-cluster.yml ps
```

Spark UI: http://localhost:8080
Worker UI: http://localhost:8081 (per worker)

---

## Reproducibility Checklist

Before each benchmark session:
- [ ] Update `docs/environment.md` with current GPU/driver versions
- [ ] Ensure no other CPU-intensive processes are running
- [ ] Check available RAM: `free -h`
- [ ] For cold-cache benchmarks: run `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`
- [ ] Verify data manifest matches expected fetch parameters
- [ ] Record Git commit SHA in results (automatic via `runner.py`)
