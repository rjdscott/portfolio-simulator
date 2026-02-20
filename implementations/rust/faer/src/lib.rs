//! Portfolio metrics kernel — Rust + faer linear algebra library.
//!
//! faer is a pure-Rust, BLAS-level matrix library that auto-parallelises
//! its GEMM kernel via Rayon.  This crate uses faer's matmul for the heavy
//! (N×U) × (U×T) inner product, then Rayon for per-portfolio statistics.
//!
//! Memory layout note: faer uses column-major storage internally.
//! The C-ABI arrays are row-major, so we convert them via `Mat::from_fn`
//! (one copy each for R and W).  The per-portfolio port_returns matrix is
//! (N, T) in faer's column-major order, but we access elements by index.
//!
//! Build:
//!   cargo build --release
//! Output:
//!   target/release/libportfolio_faer.so

use faer::linalg::matmul;
use faer::prelude::*;
use faer::Parallelism;
use rayon::prelude::*;
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
    let (n, t, u) = (n as usize, t as usize, u as usize);

    let r_slice = unsafe { slice::from_raw_parts(r_ptr, t * u) };
    let w_slice = unsafe { slice::from_raw_parts(w_ptr, n * u) };
    let results = unsafe { slice::from_raw_parts_mut(results_ptr, n * 2) };

    // Copy inputs into faer column-major matrices (from row-major C arrays).
    // R is small (~1 MB for T=1257, U=100) — negligible copy overhead.
    // W can be large at N=1M (800 MB) — same memory footprint as numpy_vectorised.
    let r_mat = Mat::<f64>::from_fn(t, u, |i, j| r_slice[i * u + j]);
    let w_mat = Mat::<f64>::from_fn(n, u, |i, j| w_slice[i * u + j]);

    // Compute W * R^T = (N, T) using faer's BLAS-optimised GEMM.
    // Parallelism::Rayon(0) uses all available threads automatically.
    let mut port_returns = Mat::<f64>::zeros(n, t);
    matmul::matmul(
        port_returns.as_mut(),
        w_mat.as_ref(),
        r_mat.transpose(),
        None,     // alpha: None → output = beta * lhs * rhs (no accumulation)
        1.0f64,   // beta
        Parallelism::Rayon(0),
    );

    // Per-portfolio Welford statistics using Rayon parallel iteration.
    // One task per portfolio — work-stealing over N.
    results
        .par_chunks_mut(2)
        .enumerate()
        .for_each(|(i, out)| {
            let mut log_sum: f64 = 0.0;
            let mut mean: f64 = 0.0;
            let mut m2: f64 = 0.0;

            for t_idx in 0..t {
                let pr = port_returns[(i, t_idx)];
                log_sum += pr;
                let count = (t_idx + 1) as f64;
                let delta = pr - mean;
                mean += delta / count;
                let delta2 = pr - mean;
                m2 += delta * delta2;
            }

            // Cumulative return: expm1(sum of log-returns)
            out[0] = log_sum.exp_m1();

            // Annualised Sharpe (ddof=1): mean / std * sqrt(252)
            out[1] = if t > 1 && m2 > 0.0 {
                let std_r = (m2 / (t - 1) as f64).sqrt();
                mean / std_r * 252_f64.sqrt()
            } else {
                0.0
            };
        });
}
