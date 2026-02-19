/**
 * portfolio_compute.cpp — OpenMP parallel portfolio return computation.
 *
 * Compiled to a shared library (libportfolio_openmp.so) and called from
 * Python via ctypes. No Python headers required.
 *
 * Algorithm
 * ---------
 * For each portfolio i in [0, N):
 *   1. Compute daily portfolio log-return: port_r[t] = sum_u(W[i,u] * R[t,u])
 *   2. Accumulate Welford online mean + M2 for variance (ddof=1, single pass)
 *   3. cum_return = expm1(sum of port_r)
 *   4. sharpe = mean / sqrt(M2/(T-1)) * sqrt(252)  (if variance > 0)
 *
 * Compile flags: -O3 -march=native -fopenmp -shared -fPIC -std=c++17
 *
 * The outer prange is embarrassingly parallel — each thread owns a disjoint
 * range of portfolio indices with no shared mutable state.
 */

#include <cmath>
#include <cstdint>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static constexpr double TRADING_DAYS_PER_YEAR = 252.0;

/**
 * compute_portfolio_metrics
 *
 * Parameters (C layout, row-major)
 * ---------------------------------
 * R       : double[T * U] — log returns, row = day, col = stock
 * W       : double[N * U] — portfolio weights, row = portfolio, col = stock
 * results : double[N * 2] — output: [cum_return, sharpe] per portfolio
 * N       : number of portfolios
 * T       : number of trading days
 * U       : universe size (number of stocks)
 */
extern "C" EXPORT void compute_portfolio_metrics(
    const double* __restrict__ R,       // (T, U)
    const double* __restrict__ W,       // (N, U)
    double*       __restrict__ results, // (N, 2) output
    int64_t N,
    int64_t T,
    int64_t U
) {
    const double sqrt_annual = std::sqrt(TRADING_DAYS_PER_YEAR);

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        const double* __restrict__ w = W + i * U;  // weights for portfolio i

        // Welford online algorithm for mean and variance
        double count = 0.0;
        double mean  = 0.0;
        double M2    = 0.0;
        double log_sum = 0.0;

        for (int64_t t = 0; t < T; ++t) {
            const double* __restrict__ r = R + t * U;  // returns row for day t

            // Weighted dot product — explicitly SIMD-vectorised over U stocks.
            // -ffast-math allows the compiler to reorder this FP reduction and
            // use AVX2 FMA instructions (8 doubles/cycle on Haswell+).
            double port_r = 0.0;
#pragma omp simd reduction(+:port_r)
            for (int64_t u = 0; u < U; ++u) {
                port_r += w[u] * r[u];
            }

            log_sum += port_r;

            // Welford update
            count += 1.0;
            double delta  = port_r - mean;
            mean         += delta / count;
            double delta2 = port_r - mean;
            M2           += delta * delta2;
        }

        // Cumulative return: e^(sum of log returns) - 1
        double cum_return = std::expm1(log_sum);

        // Annualised Sharpe (ddof=1)
        double sharpe = 0.0;
        if (count > 1.0 && M2 > 0.0) {
            double variance = M2 / (count - 1.0);
            double std_r    = std::sqrt(variance);
            sharpe = (mean / std_r) * sqrt_annual;
        }

        results[i * 2 + 0] = cum_return;
        results[i * 2 + 1] = sharpe;
    }
}
