/**
 * portfolio_compute_tile4.cpp — 4-portfolio AVX2 tile with transposed W.
 *
 * Engine: cpp_openmp_tile4
 *
 * Key insight: instead of vectorizing over U stocks for one portfolio at a
 * time, vectorize across 4 portfolios simultaneously using AVX2 (256-bit)
 * registers holding 4 doubles.
 *
 * Layout change
 * -------------
 * W is transposed to shape (U, N) before the call (done in Python):
 *   W_T[u * N + i] = weight of stock u in portfolio i
 *
 * This lets us broadcast r[t, u] (a scalar) across 4 portfolios, then load
 * 4 weights from W_T[u*N + i .. u*N + i+3] into a single ymm register and
 * compute a 4-way FMA. Every R value is reused 4× per outer i-iteration,
 * reducing unique R cache-line fetches by 4× relative to per-portfolio
 * processing.
 *
 * Inner kernel (per AVX2 FMA):
 *   port_r[0..3] += _mm256_fmadd_pd(w_v, r_bc, port_r_v)
 * where:
 *   r_bc = _mm256_set1_pd(R[t, u])          — broadcast scalar
 *   w_v  = _mm256_loadu_pd(W_T + u*N + i)   — 4 weights in one load
 *
 * Welford mean/variance is maintained in parallel AVX2 registers over T days,
 * then reduced to scalars at the end.
 *
 * Tail handling: when N % 4 != 0, the remainder portfolios are processed
 * one at a time with the scalar 8-wide unroll kernel.
 *
 * Compile flags: -O3 -march=native -ffast-math -funroll-loops -fopenmp
 *                -shared -fPIC -mavx2 -mfma
 */

#include <cmath>
#include <cstdint>
#include <immintrin.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

static constexpr double TRADING_DAYS_PER_YEAR = 252.0;

// ---------------------------------------------------------------------------
// Scalar fallback: 8-wide unroll for remainder portfolios (N % 4 != 0)
// ---------------------------------------------------------------------------
static inline void compute_scalar(
    const double* __restrict__ R,
    const double* __restrict__ W_T,   // (U, N) transposed
    double*       __restrict__ out,
    int64_t i,
    int64_t N,
    int64_t T,
    int64_t U,
    double sqrt_annual
) {
    const int64_t U8 = (U / 8) * 8;

    double count   = 0.0;
    double mean    = 0.0;
    double M2      = 0.0;
    double log_sum = 0.0;

    for (int64_t t = 0; t < T; ++t) {
        const double* __restrict__ r = R + t * U;
        double a0=0.0,a1=0.0,a2=0.0,a3=0.0,a4=0.0,a5=0.0,a6=0.0,a7=0.0;

        for (int64_t u = 0; u < U8; u += 8) {
            a0 += W_T[(u+0)*N + i] * r[u+0];
            a1 += W_T[(u+1)*N + i] * r[u+1];
            a2 += W_T[(u+2)*N + i] * r[u+2];
            a3 += W_T[(u+3)*N + i] * r[u+3];
            a4 += W_T[(u+4)*N + i] * r[u+4];
            a5 += W_T[(u+5)*N + i] * r[u+5];
            a6 += W_T[(u+6)*N + i] * r[u+6];
            a7 += W_T[(u+7)*N + i] * r[u+7];
        }
        for (int64_t u = U8; u < U; ++u) {
            a0 += W_T[u*N + i] * r[u];
        }
        double port_r = ((a0+a4)+(a1+a5)) + ((a2+a6)+(a3+a7));

        log_sum += port_r;
        count   += 1.0;
        double delta  = port_r - mean;
        mean         += delta / count;
        double delta2 = port_r - mean;
        M2           += delta * delta2;
    }

    double cum_return = std::expm1(log_sum);
    double sharpe = 0.0;
    if (count > 1.0 && M2 > 0.0) {
        sharpe = (mean / std::sqrt(M2 / (count - 1.0))) * sqrt_annual;
    }
    out[i * 2 + 0] = cum_return;
    out[i * 2 + 1] = sharpe;
}

// ---------------------------------------------------------------------------
// AVX2 horizontal sum of 4 doubles in a ymm register
// ---------------------------------------------------------------------------
static inline double hsum_pd(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_hadd_pd(s, s));
}

// ---------------------------------------------------------------------------
// AVX2 Welford update for 4 portfolios simultaneously
// ---------------------------------------------------------------------------
static inline void welford_update_avx2(
    __m256d& count,
    __m256d& mean,
    __m256d& M2,
    __m256d port_r
) {
    // count += 1.0 (broadcast, done outside for performance)
    __m256d delta  = _mm256_sub_pd(port_r, mean);
    mean           = _mm256_add_pd(mean, _mm256_div_pd(delta, count));
    __m256d delta2 = _mm256_sub_pd(port_r, mean);
    M2             = _mm256_add_pd(M2, _mm256_mul_pd(delta, delta2));
}

/**
 * compute_portfolio_metrics_tile4
 *
 * Parameters
 * ----------
 * R       : double[T * U]  — log returns (T rows × U cols, row-major)
 * W_T     : double[U * N]  — TRANSPOSED weights (U rows × N cols, row-major)
 *                            Python: W_T = np.ascontiguousarray(weights.T)
 * results : double[N * 2]  — output: [cum_return, sharpe] per portfolio
 * N       : number of portfolios
 * T       : number of trading days
 * U       : universe size (number of stocks)
 */
extern "C" EXPORT void compute_portfolio_metrics_tile4(
    const double* __restrict__ R,       // (T, U)
    const double* __restrict__ W_T,     // (U, N) — transposed
    double*       __restrict__ results, // (N, 2)
    int64_t N,
    int64_t T,
    int64_t U
) {
    const double sqrt_annual = std::sqrt(TRADING_DAYS_PER_YEAR);
    const __m256d one    = _mm256_set1_pd(1.0);
    const __m256d sqrt_a = _mm256_set1_pd(sqrt_annual);
    const __m256d zero   = _mm256_setzero_pd();

    const int64_t N4 = (N / 4) * 4;   // largest multiple of 4

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N4; i += 4) {
        // Welford state — 4 portfolios in parallel ymm registers
        __m256d count   = zero;
        __m256d mean    = zero;
        __m256d M2      = zero;
        __m256d log_sum = zero;

        for (int64_t t = 0; t < T; ++t) {
            const double* __restrict__ r = R + t * U;
            __m256d port_r = zero;

            // Inner loop: broadcast r[u] and load 4 weights for portfolios i..i+3
            for (int64_t u = 0; u < U; ++u) {
                __m256d r_bc = _mm256_set1_pd(r[u]);                // broadcast
                __m256d w_v  = _mm256_loadu_pd(W_T + u * N + i);   // 4 weights
                port_r = _mm256_fmadd_pd(w_v, r_bc, port_r);       // FMA
            }

            log_sum = _mm256_add_pd(log_sum, port_r);

            // Welford update (vectorised across 4 portfolios)
            count = _mm256_add_pd(count, one);
            welford_update_avx2(count, mean, M2, port_r);
        }

        // Reduce and write results for portfolios i..i+3
        alignas(32) double vcum[4], vmean[4], vM2[4], vcnt[4];
        _mm256_store_pd(vcum,  log_sum);
        _mm256_store_pd(vmean, mean);
        _mm256_store_pd(vM2,   M2);
        _mm256_store_pd(vcnt,  count);

        for (int k = 0; k < 4; ++k) {
            double cum_return = std::expm1(vcum[k]);
            double sharpe = 0.0;
            if (vcnt[k] > 1.0 && vM2[k] > 0.0) {
                sharpe = (vmean[k] / std::sqrt(vM2[k] / (vcnt[k] - 1.0))) * sqrt_annual;
            }
            results[(i + k) * 2 + 0] = cum_return;
            results[(i + k) * 2 + 1] = sharpe;
        }
    }

    // Scalar tail: handle N % 4 remaining portfolios
    for (int64_t i = N4; i < N; ++i) {
        compute_scalar(R, W_T, results, i, N, T, U, sqrt_annual);
    }
}
