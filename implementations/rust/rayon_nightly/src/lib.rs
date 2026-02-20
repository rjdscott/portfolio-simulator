//! Portfolio metrics kernel — Rust nightly with fadd_fast.
//!
//! This crate is identical to the stable `portfolio_rayon` crate except that the
//! inner dot-product reduction uses `std::intrinsics::fadd_fast`, which annotates
//! each addition with the LLVM `fast` flag (equivalent to `-ffast-math` for that
//! op).  This allows LLVM to reassociate the reduction and emit vectorised
//! `vfmadd231pd` instructions — the same path that GCC takes with `-ffast-math`.
//!
//! Expected impact: closes the ~1.9× gap between Rust stable and C++/OpenMP.
//!
//! Build:
//!   cargo +nightly build --release
//! Output:
//!   target/release/libportfolio_rayon_nightly.so

#![feature(core_intrinsics)]

use rayon::prelude::*;
use std::intrinsics::fadd_fast;
use std::slice;

/// Compute cumulative return and annualised Sharpe ratio for N portfolios.
///
/// # Arguments
/// * `r_ptr`       — pointer to (T × U) f64 array of log-returns (C-order, row-major)
/// * `w_ptr`       — pointer to (N × U) f64 array of portfolio weights (C-order)
/// * `results_ptr` — pointer to (N × 2) f64 output array
/// * `n`           — number of portfolios
/// * `t`           — number of trading days
/// * `u`           — universe size (number of stocks)
#[no_mangle]
pub extern "C" fn compute_portfolio_metrics(
    r_ptr:       *const f64,
    w_ptr:       *const f64,
    results_ptr: *mut   f64,
    n: i64,
    t: i64,
    u: i64,
) {
    let (n, t, u) = (n as usize, t as usize, u as usize);

    let r       = unsafe { slice::from_raw_parts(r_ptr,       t * u) };
    let w       = unsafe { slice::from_raw_parts(w_ptr,       n * u) };
    let results = unsafe { slice::from_raw_parts_mut(results_ptr, n * 2) };

    // One Rayon task per portfolio — work-stealing over N portfolios.
    results
        .par_chunks_mut(2)
        .enumerate()
        .for_each(|(i, out)| {
            let wi = &w[i * u..(i + 1) * u];

            let mut log_sum: f64 = 0.0;
            let mut mean:    f64 = 0.0;
            let mut m2:      f64 = 0.0;

            for t_idx in 0..t {
                let r_row = &r[t_idx * u..(t_idx + 1) * u];

                // Dot product with fadd_fast: allows LLVM to vectorise the reduction.
                // Each fadd_fast tells LLVM this addition has the 'fast' flag, enabling
                // reassociation and FMA emission (equivalent to -ffast-math for this op).
                let port_r: f64 = unsafe {
                    let mut acc = 0.0_f64;
                    for k in 0..u {
                        acc = fadd_fast(acc, wi[k] * r_row[k]);
                    }
                    acc
                };

                // Welford online mean and variance (ddof=1)
                log_sum += port_r;
                let count = (t_idx + 1) as f64;
                let delta = port_r - mean;
                mean += delta / count;
                let delta2 = port_r - mean;
                m2 += delta * delta2;
            }

            // Cumulative return: expm1(sum of log-returns)
            out[0] = log_sum.exp() - 1.0;

            // Annualised Sharpe (ddof=1): mean / std * sqrt(252)
            if t > 1 && m2 > 0.0 {
                let variance = m2 / (t - 1) as f64;
                let std_r    = variance.sqrt();
                out[1] = mean / std_r * 252_f64.sqrt();
            } else {
                out[1] = 0.0;
            }
        });
}
