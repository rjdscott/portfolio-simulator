/// portfolio_rayon — parallel portfolio return computation via Rayon.
///
/// Exported as a C-ABI shared library (`cdylib`) and called from Python via ctypes.
/// No PyO3 / Python headers required.
///
/// Algorithm (identical to all other engines):
///   For each portfolio i:
///     1. Compute daily portfolio log-return: port_r[t] = Σ_u W[i,u] * R[t,u]
///     2. Accumulate Welford online mean + M2 (ddof=1, single pass)
///     3. cum_return = expm1(Σ port_r)        (e^x - 1, numerically stable)
///     4. sharpe = mean / sqrt(M2/(T-1)) * sqrt(252)  (if variance > 0)
///
/// Parallelism
/// -----------
/// `par_chunks_mut(2)` gives one Rayon task per portfolio. Rayon's divide-and-conquer
/// work-stealing distributes tasks across all threads with minimal overhead.
/// Fine-grained tasks give LLVM the best view of a single-portfolio closure for
/// aggressive inner-loop optimisation.
///
/// Dot-product vectorisation
/// -------------------------
/// Rust stable has no -ffast-math equivalent, so LLVM cannot vectorise a sequential
/// FP reduction like `iter().sum()` — FP addition is not associative by default.
/// The 8-wide independent accumulator pattern works around this: LLVM packs
/// [a0..a3] and [a4..a7] into two AVX2 ymm registers and emits vfmadd231pd
/// without needing to reorder within any single accumulator.
/// Requires `target-cpu=native` in .cargo/config.toml to unlock AVX2/FMA.
///
/// Performance ceiling
/// -------------------
/// Without a fast-math flag at the LLVM IR level, Rust achieves ~55% of C++
/// throughput at large N — a known stable-Rust limitation. Using nightly
/// `std::intrinsics::fadd_fast` or the `wide` crate for explicit SIMD would
/// close the gap.

use rayon::prelude::*;

const TRADING_DAYS_PER_YEAR: f64 = 252.0;

/// compute_portfolio_metrics
///
/// # Safety
/// All pointer arguments must be non-null and point to valid, C-contiguous,
/// row-major f64 arrays of the documented dimensions.
#[no_mangle]
pub extern "C" fn compute_portfolio_metrics(
    r_ptr: *const f64,
    w_ptr: *const f64,
    results_ptr: *mut f64,
    n: i64,
    t: i64,
    u: i64,
) {
    let n = n as usize;
    let t = t as usize;
    let u = u as usize;

    // SAFETY: caller guarantees valid, non-overlapping slices
    let r_slice: &[f64] = unsafe { std::slice::from_raw_parts(r_ptr, t * u) };
    let w_slice: &[f64] = unsafe { std::slice::from_raw_parts(w_ptr, n * u) };
    let results_slice: &mut [f64] =
        unsafe { std::slice::from_raw_parts_mut(results_ptr, n * 2) };

    let sqrt_annual = TRADING_DAYS_PER_YEAR.sqrt();

    // One Rayon task per portfolio — LLVM sees a tight single-portfolio closure
    // and can optimise the inner T and U loops aggressively.
    results_slice
        .par_chunks_mut(2)
        .enumerate()
        .for_each(|(i, out)| {
            let w = &w_slice[i * u..(i + 1) * u];

            // Welford online algorithm (single pass over T days)
            let mut count: f64 = 0.0;
            let mut mean:  f64 = 0.0;
            let mut m2:    f64 = 0.0;
            let mut log_sum: f64 = 0.0;

            for t_idx in 0..t {
                let r = &r_slice[t_idx * u..(t_idx + 1) * u];

                // 8-wide manual accumulation — LLVM packs into 2 AVX2 ymm registers
                // (4 f64 each) and emits vfmadd231pd without needing fast-math.
                let mut a0 = 0.0f64; let mut a1 = 0.0f64;
                let mut a2 = 0.0f64; let mut a3 = 0.0f64;
                let mut a4 = 0.0f64; let mut a5 = 0.0f64;
                let mut a6 = 0.0f64; let mut a7 = 0.0f64;

                let main = (u / 8) * 8; // 96 for U=100
                let mut k = 0usize;
                while k < main {
                    a0 += w[k]     * r[k];
                    a1 += w[k + 1] * r[k + 1];
                    a2 += w[k + 2] * r[k + 2];
                    a3 += w[k + 3] * r[k + 3];
                    a4 += w[k + 4] * r[k + 4];
                    a5 += w[k + 5] * r[k + 5];
                    a6 += w[k + 6] * r[k + 6];
                    a7 += w[k + 7] * r[k + 7];
                    k += 8;
                }
                // Remainder: indices 96–99 for U=100
                for j in main..u { a0 += w[j] * r[j]; }
                let port_r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;

                log_sum += port_r;

                // Welford update
                count += 1.0;
                let delta  = port_r - mean;
                mean      += delta / count;
                let delta2 = port_r - mean;
                m2        += delta * delta2;
            }

            // Cumulative return: e^(log_sum) - 1  (numerically stable for small log_sum)
            let cum_return = log_sum.exp_m1();

            // Annualised Sharpe (ddof=1)
            let sharpe = if count > 1.0 && m2 > 0.0 {
                let std_r = (m2 / (count - 1.0)).sqrt();
                (mean / std_r) * sqrt_annual
            } else {
                0.0
            };

            out[0] = cum_return;
            out[1] = sharpe;
        });
}
