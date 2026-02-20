! Portfolio metrics kernel — FORTRAN 2008 + OpenMP + ISO_C_BINDING
!
! Computes cumulative return and annualised Sharpe ratio for N portfolios.
! Uses Welford's online algorithm for numerically stable variance (ddof=1).
!
! Memory layout: all arrays are C-order (row-major) float64, passed as raw
! C pointers via ISO_C_BINDING.  FORTRAN's column-major storage convention
! is irrelevant here — we use linear index arithmetic throughout.
!
! Build:
!   gfortran -O3 -march=native -ffast-math -funroll-loops -fopenmp \
!            -shared -fPIC -o libportfolio_fortran.so \
!            src/portfolio_compute.f90
!
! Or via CMake:
!   cmake -S implementations/fortran/openmp \
!         -B implementations/fortran/openmp/build \
!         -DCMAKE_BUILD_TYPE=Release
!   cmake --build implementations/fortran/openmp/build --parallel

subroutine compute_portfolio_metrics(r_ptr, w_ptr, out_ptr, N, T, U) BIND(C)
    use ISO_C_BINDING
    implicit none

    integer(C_INT64_T), value :: N, T, U
    type(C_PTR),        value :: r_ptr, w_ptr, out_ptr

    real(C_DOUBLE), pointer :: r(:), w(:), out(:)

    integer(C_INT64_T) :: i, t_idx, u_idx, cnt
    real(C_DOUBLE)     :: port_r, log_sum, mean_r, m2, delta, delta2

    call C_F_POINTER(r_ptr,   r,   [T * U])
    call C_F_POINTER(w_ptr,   w,   [N * U])
    call C_F_POINTER(out_ptr, out, [N * 2])

    !$OMP PARALLEL DO DEFAULT(NONE) &
    !$OMP&    SHARED(r, w, out, N, T, U) &
    !$OMP&    PRIVATE(i, t_idx, u_idx, cnt, port_r, log_sum, mean_r, m2, delta, delta2) &
    !$OMP&    SCHEDULE(static)
    do i = 0, N - 1

        log_sum = 0.0_C_DOUBLE
        mean_r  = 0.0_C_DOUBLE
        m2      = 0.0_C_DOUBLE
        cnt     = 0

        do t_idx = 0, T - 1

            ! Dot product: portfolio i against returns row t_idx
            ! Both arrays are C-order row-major; index as r(t_idx*U + u_idx)
            port_r = 0.0_C_DOUBLE
            !$OMP SIMD REDUCTION(+:port_r)
            do u_idx = 0, U - 1
                port_r = port_r + w(i * U + u_idx + 1) * r(t_idx * U + u_idx + 1)
            end do

            log_sum = log_sum + port_r

            ! Welford online mean and M2 (ddof=1)
            cnt    = cnt + 1
            delta  = port_r - mean_r
            mean_r = mean_r + delta / real(cnt, C_DOUBLE)
            delta2 = port_r - mean_r
            m2     = m2 + delta * delta2

        end do

        ! Cumulative return: exp(sum of log-returns) - 1
        out(i * 2 + 1) = exp(log_sum) - 1.0_C_DOUBLE

        ! Annualised Sharpe (ddof=1): mean / sqrt(M2/(T-1)) * sqrt(252)
        if (T > 1 .and. m2 > 0.0_C_DOUBLE) then
            out(i * 2 + 2) = mean_r / sqrt(m2 / real(T - 1, C_DOUBLE)) * sqrt(252.0_C_DOUBLE)
        else
            out(i * 2 + 2) = 0.0_C_DOUBLE
        end if

    end do
    !$OMP END PARALLEL DO

end subroutine compute_portfolio_metrics
