/**
 * portfolio_compute.cpp — C++ Eigen compute kernel.
 *
 * Uses Eigen's header-only linear algebra library for the (N×U) × (U×T)
 * matmul (W * R^T), then computes per-portfolio cumulative return and
 * annualised Sharpe ratio using Eigen array operations.
 *
 * Parallelism: OpenMP #pragma omp parallel for over N portfolios.
 * Eigen's internal BLAS-backed matmul is also multi-threaded via OpenBLAS
 * or MKL, but for large N the outer OpenMP loop dominates.
 *
 * Compile flags: -O3 -march=native -ffast-math -funroll-loops
 * These flags (especially -ffast-math) are critical for SIMD vectorisation
 * of the FP reduction in the Eigen matmul inner kernels.
 *
 * ABI: same extern "C" signature as cpp_openmp so the Python ctypes wrapper
 * is interchangeable.
 *
 * Build:
 *   cmake -S implementations/cpp/eigen -B implementations/cpp/eigen/build \
 *         -DCMAKE_BUILD_TYPE=Release
 *   cmake --build implementations/cpp/eigen/build --parallel
 *
 * Prerequisite:
 *   sudo apt install libeigen3-dev
 */

#include <cmath>
#include <cstdint>

#include <Eigen/Dense>

#ifdef _WIN32
#  define EXPORT __declspec(dllexport)
#else
#  define EXPORT __attribute__((visibility("default")))
#endif

using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapConst = Eigen::Map<const MatrixXdRowMajor>;
using Map      = Eigen::Map<MatrixXdRowMajor>;

static constexpr double SQRT_252 = 15.874507866387544;  // sqrt(252)

extern "C" {

/**
 * compute_portfolio_metrics
 *
 * @param r        Pointer to (T × U) float64 log-returns matrix (C-order)
 * @param w        Pointer to (N × U) float64 portfolio weights matrix (C-order)
 * @param results  Pointer to (N × 2) float64 output [cum_return, sharpe]
 * @param n        Number of portfolios
 * @param t        Number of trading days
 * @param u        Universe size (number of stocks)
 */
EXPORT void compute_portfolio_metrics(
    const double* __restrict__ r,
    const double* __restrict__ w,
    double*       __restrict__ results,
    int64_t n,
    int64_t t,
    int64_t u
) {
    // Wrap raw C pointers as Eigen maps (zero copy)
    MapConst R(r, t, u);  // (T, U)
    MapConst W(w, n, u);  // (N, U)

    // Compute (N, T) daily portfolio log-returns: port_returns = W * R^T
    // Eigen uses BLAS-backed GEMM; -ffast-math enables FMA and reassociation.
    MatrixXdRowMajor port_returns = W * R.transpose();  // (N, T)

    // Per-portfolio statistics using Eigen array operations
    // These loops are embarrassingly parallel — OpenMP over N
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        // Row view of daily portfolio returns
        auto row = port_returns.row(i);  // 1 × T

        // Cumulative return: expm1(sum of log-returns)
        double log_sum = row.sum();
        results[i * 2] = std::expm1(log_sum);

        // Annualised Sharpe (ddof=1): mean / std * sqrt(252)
        double mean_r = log_sum / static_cast<double>(t);
        double var_r  = (row.array() - mean_r).square().sum() / static_cast<double>(t - 1);

        if (var_r > 0.0) {
            results[i * 2 + 1] = mean_r / std::sqrt(var_r) * SQRT_252;
        } else {
            results[i * 2 + 1] = 0.0;
        }
    }
}

}  // extern "C"
