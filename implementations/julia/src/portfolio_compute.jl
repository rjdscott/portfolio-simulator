"""
Portfolio metrics kernel — Julia with LoopVectorization.jl (@turbo).

Computes cumulative return and annualised Sharpe ratio for N portfolios.
Uses Threads.@threads for outer parallelism (over portfolios) and @turbo
for inner SIMD vectorisation (the dot-product over U stocks).

@turbo (LoopVectorization.jl) applies polyhedral loop analysis and emits
SIMD intrinsics directly, without the -ffast-math language-level restriction
that limits Rust stable.  This is the Julia equivalent of Numba's fastmath=True.

Memory layout: the Python caller transposes numpy arrays to (U, T) and (U, N)
in Fortran-order (column-major) and passes raw data pointers + dimensions.
unsafe_wrap turns these into native Julia Matrix{Float64} objects — zero-copy,
correct strides, and fully compatible with @turbo's check_args.

Interface: compute_portfolio_metrics(r_ptr, w_ptr, U, T, N) -> Matrix{Float64}
All pointer arguments are Python integers (Int64 on 64-bit systems).
"""

using LoopVectorization
using Base.Threads

const SQRT_252 = sqrt(252.0)

"""
    compute_portfolio_metrics(r_ptr, w_ptr, U, T, N) -> Matrix{Float64}

Compute cumulative return and annualised Sharpe for each of the N portfolios.

# Arguments
- `r_ptr`: raw data pointer (Int) to a (U × T) Fortran-order float64 array
- `w_ptr`: raw data pointer (Int) to a (U × N) Fortran-order float64 array
- `U`: number of stocks (universe size)
- `T`: number of trading days
- `N`: number of portfolios

# Returns
- `(N × 2)` Float64 matrix: column 1 = cumulative return, column 2 = annualised Sharpe

# Notes
unsafe_wrap creates native Julia Matrix{Float64} views onto the numpy data
without copying.  The arrays must remain alive on the Python side for the
duration of this call (guaranteed by the caller).
"""
function compute_portfolio_metrics(
    r_ptr::Int, w_ptr::Int,
    U::Int, T::Int, N::Int,
)
    # Wrap raw pointers as native Julia column-major matrices.
    # Data is (U, T) and (U, N) Fortran-order → unit stride on dim-1 (u axis).
    returns_jl = unsafe_wrap(Matrix{Float64}, Ptr{Float64}(r_ptr), (U, T))
    weights_jl = unsafe_wrap(Matrix{Float64}, Ptr{Float64}(w_ptr), (U, N))

    out = Matrix{Float64}(undef, 2, N)  # (2, N) — transposed for column-major efficiency

    @threads for i in 1:N
        log_sum = 0.0
        mean_r  = 0.0
        m2      = 0.0

        for t in 1:T
            # Dot product: portfolio i vs returns day t.
            # Both accesses have unit stride on u — SIMD-friendly for @turbo.
            port_r = 0.0
            @turbo for u in 1:U
                port_r += weights_jl[u, i] * returns_jl[u, t]
            end

            log_sum += port_r

            # Welford online mean and M2 (ddof=1)
            count   = Float64(t)
            delta   = port_r - mean_r
            mean_r += delta / count
            delta2  = port_r - mean_r
            m2     += delta * delta2
        end

        # Cumulative return: exp(sum of log-returns) - 1
        out[1, i] = exp(log_sum) - 1.0

        # Annualised Sharpe (ddof=1)
        if T > 1 && m2 > 0.0
            std_r     = sqrt(m2 / (T - 1))
            out[2, i] = mean_r / std_r * SQRT_252
        else
            out[2, i] = 0.0
        end
    end

    # Return (N, 2) to match the convention of all other engines
    return Matrix(out')
end
