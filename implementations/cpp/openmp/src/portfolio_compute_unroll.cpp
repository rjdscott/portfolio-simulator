/**
 * portfolio_compute_unroll.cpp — 8-wide accumulator unroll + day tiling.
 *
 * Engine: cpp_openmp_unroll
 *
 * Optimizations over the baseline cpp_openmp engine:
 *
 * 1. 8-wide scalar accumulator unroll
 *    The baseline uses `#pragma omp simd reduction(+:port_r)` which generates a
 *    single ymm accumulator chain. FMA latency (4–5 cycles on Intel) means the
 *    dot product over U=100 stocks runs latency-bound:
 *      25 SIMD rounds × 4-cycle latency = 100 cycles
 *
 *    Eight independent scalar accumulators (a0..a7) let GCC emit 2 independent
 *    ymm FMA chains (with -ffast-math) that fill both AVX2 FMA execution ports:
 *      8 chains ÷ 4 doubles/ymm = 2 ymm chains × 4-cycle latency ≈ 13 cycles
 *    Expected 5–7× speedup on the dot-product kernel.
 *
 * 2. Day-loop tiling (tile B_t = 40 days)
 *    The R matrix row for day t is U=100 doubles = 800 bytes. With a 40-day tile
 *    that is 32 KB — matching the L1d cache size. All R rows in a tile stay warm
 *    in L1 for the duration of the tile's work, eliminating repeated cache-line
 *    refills across the outer portfolio loop iteration.
 *    (With the embarrassingly-parallel outer i-loop each thread processes its
 *    own range of portfolios, so the reuse benefit is within one portfolio's
 *    T-iterations.)
 *
 * Compile flags: -O3 -march=native -ffast-math -funroll-loops -fopenmp -shared -fPIC
 */

#include <cmath>
#include <cstdint>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static constexpr double TRADING_DAYS_PER_YEAR = 252.0;
static constexpr int64_t DAY_TILE = 40;   // 40 × 100 × 8 bytes = 32 KB ≈ L1d

/**
 * compute_portfolio_metrics_unroll
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
extern "C" EXPORT void compute_portfolio_metrics_unroll(
    const double* __restrict__ R,       // (T, U)
    const double* __restrict__ W,       // (N, U)
    double*       __restrict__ results, // (N, 2) output
    int64_t N,
    int64_t T,
    int64_t U
) {
    const double sqrt_annual = std::sqrt(TRADING_DAYS_PER_YEAR);
    const int64_t U8 = (U / 8) * 8;   // largest multiple of 8 ≤ U

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        const double* __restrict__ w = W + i * U;

        // Welford state
        double count   = 0.0;
        double mean    = 0.0;
        double M2      = 0.0;
        double log_sum = 0.0;

        // Day-tiled outer loop
        for (int64_t t0 = 0; t0 < T; t0 += DAY_TILE) {
            const int64_t t_end = (t0 + DAY_TILE < T) ? t0 + DAY_TILE : T;

            for (int64_t t = t0; t < t_end; ++t) {
                const double* __restrict__ r = R + t * U;

                // 8-wide scalar accumulator unroll.
                // With -ffast-math, GCC packs these into 2 independent ymm FMA
                // chains, saturating both AVX2 FMA execution ports.
                double a0=0.0, a1=0.0, a2=0.0, a3=0.0;
                double a4=0.0, a5=0.0, a6=0.0, a7=0.0;

                for (int64_t u = 0; u < U8; u += 8) {
                    a0 += w[u+0] * r[u+0];
                    a1 += w[u+1] * r[u+1];
                    a2 += w[u+2] * r[u+2];
                    a3 += w[u+3] * r[u+3];
                    a4 += w[u+4] * r[u+4];
                    a5 += w[u+5] * r[u+5];
                    a6 += w[u+6] * r[u+6];
                    a7 += w[u+7] * r[u+7];
                }
                // Remainder (when U % 8 != 0)
                for (int64_t u = U8; u < U; ++u) {
                    a0 += w[u] * r[u];
                }

                // Tree reduction to minimise rounding error
                double port_r = ((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7));

                log_sum += port_r;

                // Welford update
                count += 1.0;
                double delta  = port_r - mean;
                mean         += delta / count;
                double delta2 = port_r - mean;
                M2           += delta * delta2;
            }
        }

        double cum_return = std::expm1(log_sum);

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
